"""VOICEVOX ENGINE の CPU affinity を設定する。"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

CpuAffinityMode = Literal["auto", "disabled"]
CpuAffinityStatus = Literal["applied", "disabled", "unsupported"]

_LOGGER = logging.getLogger(__name__)
_CPU_AFFINITY_MODES = ("auto", "disabled")
_LINUX_AFFINITY_RETRY_COUNT = 3


class _LinuxThreadAffinityRace(Exception):
    """Linux の TID 列挙競合。"""


@dataclass(frozen=True)
class CpuAffinityConfiguration:
    """CPU affinity の設定結果。"""

    mode: CpuAffinityMode
    status: CpuAffinityStatus
    original_cpu_set: tuple[int, ...] | None
    requested_cpu_set: tuple[int, ...] | None
    configured_cpu_set: tuple[int, ...] | None
    excluded_cpu: int | None
    cpu_num_threads: int | None
    reason: str


def parse_cpu_affinity_mode(value: str | None) -> CpuAffinityMode:
    """CPU affinity の設定値を解析する。"""
    if value is None:
        return "auto"
    if value == "auto":
        return "auto"
    if value == "disabled":
        return "disabled"
    raise ValueError(
        "VV_CPU_AFFINITY_MODE は auto または disabled を指定してください。"
    )


def configure_cpu_affinity(
    mode: CpuAffinityMode, cpu_num_threads: int | None
) -> CpuAffinityConfiguration:
    """CPU affinity を設定し、Core に渡すスレッド数を決定する。"""
    if mode not in _CPU_AFFINITY_MODES:
        raise ValueError(f"想定外の CPU affinity mode です: {mode}")
    if cpu_num_threads is not None and cpu_num_threads < 0:
        raise ValueError("cpu_num_threads は0以上で指定してください。")

    if mode == "disabled":
        configuration = CpuAffinityConfiguration(
            mode=mode,
            status="disabled",
            original_cpu_set=None,
            requested_cpu_set=None,
            configured_cpu_set=None,
            excluded_cpu=None,
            cpu_num_threads=cpu_num_threads,
            reason="CPU affinity は無効化されています。",
        )
        _log_configuration(configuration)
        return configuration

    system = platform.system()
    if system == "Linux":
        configuration = _configure_linux(cpu_num_threads)
    elif system == "Windows":
        configuration = _configure_windows(cpu_num_threads)
    elif system == "Darwin":
        configuration = CpuAffinityConfiguration(
            mode=mode,
            status="unsupported",
            original_cpu_set=None,
            requested_cpu_set=None,
            configured_cpu_set=None,
            excluded_cpu=None,
            cpu_num_threads=cpu_num_threads,
            reason="macOS では論理 CPU ID の affinity を設定できません。",
        )
    else:
        raise RuntimeError(f"対応していない OS です: {system}")

    _log_configuration(configuration)
    return configuration


def _configure_linux(cpu_num_threads: int | None) -> CpuAffinityConfiguration:
    original_cpu_set = tuple(sorted(_linux_sched_getaffinity(0)))
    if len(original_cpu_set) <= 1:
        excluded_cpu = original_cpu_set[0] if len(original_cpu_set) == 1 else None
        return CpuAffinityConfiguration(
            mode="auto",
            status="unsupported",
            original_cpu_set=original_cpu_set,
            requested_cpu_set=(),
            configured_cpu_set=original_cpu_set,
            excluded_cpu=excluded_cpu,
            cpu_num_threads=cpu_num_threads,
            reason="利用可能な論理 CPU が二つ未満のため affinity を設定できません。",
        )

    excluded_cpu = original_cpu_set[-1]
    requested_cpu_set = original_cpu_set[:-1]
    _set_linux_thread_affinity(requested_cpu_set)
    return CpuAffinityConfiguration(
        mode="auto",
        status="applied",
        original_cpu_set=original_cpu_set,
        requested_cpu_set=requested_cpu_set,
        configured_cpu_set=requested_cpu_set,
        excluded_cpu=excluded_cpu,
        cpu_num_threads=_normalized_cpu_num_threads(
            cpu_num_threads, len(requested_cpu_set)
        ),
        reason="最大 CPU ID を除外しました。",
    )


def _linux_thread_ids() -> tuple[int, ...]:
    return tuple(sorted(int(path.name) for path in Path("/proc/self/task").iterdir()))


def _set_linux_thread_affinity(requested_cpu_set: tuple[int, ...]) -> None:
    for attempt in range(_LINUX_AFFINITY_RETRY_COUNT):
        try:
            thread_ids = _linux_thread_ids()
            requested_cpu_mask = set(requested_cpu_set)
            for thread_id in thread_ids:
                _linux_sched_setaffinity(thread_id, requested_cpu_mask)

            configured_thread_ids = _linux_thread_ids()
            if configured_thread_ids != thread_ids:
                raise _LinuxThreadAffinityRace()
            for thread_id in configured_thread_ids:
                configured_cpu_set = tuple(sorted(_linux_sched_getaffinity(thread_id)))
                if configured_cpu_set != requested_cpu_set:
                    raise RuntimeError(
                        "Linux の CPU affinity 設定後の CPU 集合が要求集合と一致しません。"
                    )

            final_thread_ids = _linux_thread_ids()
            if final_thread_ids != configured_thread_ids:
                raise _LinuxThreadAffinityRace()
            return
        except (
            _LinuxThreadAffinityRace,
            FileNotFoundError,
            ProcessLookupError,
        ) as error:
            if attempt == _LINUX_AFFINITY_RETRY_COUNT - 1:
                raise RuntimeError(
                    "Linux の全 TID への CPU affinity 設定が安定しません。"
                ) from error
    raise RuntimeError("Linux の全 TID への CPU affinity 設定に失敗しました。")


def _linux_sched_getaffinity(thread_id: int) -> set[int]:
    sched_getaffinity = cast(Callable[[int], set[int]], vars(os)["sched_getaffinity"])
    return sched_getaffinity(thread_id)


def _linux_sched_setaffinity(thread_id: int, cpu_set: set[int]) -> None:
    sched_setaffinity = cast(
        Callable[[int, set[int]], None], vars(os)["sched_setaffinity"]
    )
    sched_setaffinity(thread_id, cpu_set)


def _configure_windows(cpu_num_threads: int | None) -> CpuAffinityConfiguration:
    kernel32 = _windows_kernel32()
    processor_group_count = int(kernel32.GetActiveProcessorGroupCount())
    if processor_group_count > 1:
        return CpuAffinityConfiguration(
            mode="auto",
            status="unsupported",
            original_cpu_set=None,
            requested_cpu_set=None,
            configured_cpu_set=None,
            excluded_cpu=None,
            cpu_num_threads=cpu_num_threads,
            reason="複数の processor group を持つ Windows 環境は未対応です。",
        )
    if processor_group_count != 1:
        raise RuntimeError("Windows の processor group 数を取得できません。")

    original_cpu_set = _windows_process_cpu_set(kernel32)
    if len(original_cpu_set) <= 1:
        excluded_cpu = original_cpu_set[0] if len(original_cpu_set) == 1 else None
        return CpuAffinityConfiguration(
            mode="auto",
            status="unsupported",
            original_cpu_set=original_cpu_set,
            requested_cpu_set=(),
            configured_cpu_set=original_cpu_set,
            excluded_cpu=excluded_cpu,
            cpu_num_threads=cpu_num_threads,
            reason="利用可能な論理 CPU が二つ未満のため affinity を設定できません。",
        )

    excluded_cpu = original_cpu_set[-1]
    requested_cpu_set = original_cpu_set[:-1]
    _windows_set_process_cpu_set(kernel32, requested_cpu_set)
    configured_cpu_set = _windows_process_cpu_set(kernel32)
    if configured_cpu_set != requested_cpu_set:
        raise RuntimeError(
            "Windows の CPU affinity 設定後の CPU 集合が要求集合と一致しません。"
        )
    return CpuAffinityConfiguration(
        mode="auto",
        status="applied",
        original_cpu_set=original_cpu_set,
        requested_cpu_set=requested_cpu_set,
        configured_cpu_set=configured_cpu_set,
        excluded_cpu=excluded_cpu,
        cpu_num_threads=_normalized_cpu_num_threads(
            cpu_num_threads, len(requested_cpu_set)
        ),
        reason="最大 CPU ID を除外しました。",
    )


def _normalized_cpu_num_threads(
    cpu_num_threads: int | None, requested_cpu_count: int
) -> int:
    if cpu_num_threads is None or cpu_num_threads == 0:
        return requested_cpu_count
    return cpu_num_threads


def _log_configuration(configuration: CpuAffinityConfiguration) -> None:
    payload = {
        "event": "cpu_affinity",
        "mode": configuration.mode,
        "state": configuration.status,
        "original_cpu_set": _cpu_set_for_log(configuration.original_cpu_set),
        "requested_cpu_set": _cpu_set_for_log(configuration.requested_cpu_set),
        "configured_cpu_set": _cpu_set_for_log(configuration.configured_cpu_set),
        "excluded_cpu": configuration.excluded_cpu,
        "cpu_num_threads": configuration.cpu_num_threads,
        "reason": configuration.reason,
    }
    _LOGGER.warning(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _cpu_set_for_log(cpu_set: tuple[int, ...] | None) -> list[int] | None:
    if cpu_set is None:
        return None
    return list(cpu_set)


def _windows_kernel32() -> ctypes.CDLL:
    kernel32 = ctypes.CDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetActiveProcessorGroupCount.argtypes = []
    kernel32.GetActiveProcessorGroupCount.restype = ctypes.c_ushort
    kernel32.GetLastError.argtypes = []
    kernel32.GetLastError.restype = ctypes.c_uint32
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.GetProcessAffinityMask.restype = ctypes.c_int
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    return kernel32


def _windows_error(kernel32: ctypes.CDLL) -> OSError:
    error_code = int(kernel32.GetLastError())
    return OSError(error_code, f"Windows API エラー: {error_code}")


def _windows_process_cpu_set(kernel32: ctypes.CDLL) -> tuple[int, ...]:
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if (
        kernel32.GetProcessAffinityMask(
            kernel32.GetCurrentProcess(),
            ctypes.byref(process_mask),
            ctypes.byref(system_mask),
        )
        == 0
    ):
        raise _windows_error(kernel32)
    return tuple(
        bit
        for bit in range(ctypes.sizeof(ctypes.c_size_t) * 8)
        if process_mask.value & (1 << bit) != 0
    )


def _windows_set_process_cpu_set(
    kernel32: ctypes.CDLL, cpu_set: tuple[int, ...]
) -> None:
    mask = 0
    bit_count = ctypes.sizeof(ctypes.c_size_t) * 8
    for cpu in cpu_set:
        if cpu < 0 or cpu >= bit_count:
            raise ValueError(f"CPU ID が process affinity mask の範囲外です: {cpu}")
        mask |= 1 << cpu
    if (
        kernel32.SetProcessAffinityMask(
            kernel32.GetCurrentProcess(), ctypes.c_size_t(mask)
        )
        == 0
    ):
        raise _windows_error(kernel32)
