"""CPU affinity utility のテスト。"""

import ctypes
import json
import logging
from unittest.mock import Mock

import pytest

from voicevox_engine.utility import cpu_affinity


def test_parse_cpu_affinity_mode_defaults_to_auto() -> None:
    """未指定の CPU affinity mode は auto になる。"""
    assert cpu_affinity.parse_cpu_affinity_mode(None) == "auto"


@pytest.mark.parametrize("mode", ["auto", "disabled"])
def test_parse_cpu_affinity_mode_accepts_supported_values(mode: str) -> None:
    """許可された CPU affinity mode を解析できる。"""
    assert cpu_affinity.parse_cpu_affinity_mode(mode) == mode


def test_parse_cpu_affinity_mode_rejects_unknown_value() -> None:
    """未定義の CPU affinity mode は拒否される。"""
    with pytest.raises(ValueError, match="VV_CPU_AFFINITY_MODE"):
        cpu_affinity.parse_cpu_affinity_mode("unexpected")


def test_configure_cpu_affinity_disabled_does_not_change_cpu_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """disabled では CPU affinity を変更しない。"""
    with caplog.at_level(logging.WARNING):
        configuration = cpu_affinity.configure_cpu_affinity("disabled", 0)

    assert configuration.status == "disabled"
    assert configuration.cpu_num_threads == 0
    assert json.loads(caplog.records[-1].getMessage()) == {
        "configured_cpu_set": None,
        "cpu_num_threads": 0,
        "event": "cpu_affinity",
        "excluded_cpu": None,
        "mode": "disabled",
        "original_cpu_set": None,
        "reason": "CPU affinity は無効化されています。",
        "requested_cpu_set": None,
        "state": "disabled",
    }


def test_configure_cpu_affinity_linux_excludes_maximum_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux では有効 CPU 集合の最大 ID を除外する。"""
    masks = iter(({1, 4, 9}, {1, 4}, {1, 4}))
    thread_ids = (101, 102)
    get_calls: list[int] = []
    set_calls: list[tuple[int, set[int]]] = []
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_thread_ids",
        lambda: thread_ids,
    )

    def get_affinity(pid: int) -> set[int]:
        """CPU affinity API の呼び出しを記録する。"""
        get_calls.append(pid)
        return next(masks)

    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_sched_getaffinity",
        get_affinity,
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_sched_setaffinity",
        lambda pid, mask: set_calls.append((pid, set(mask))),
    )

    configuration = cpu_affinity.configure_cpu_affinity("auto", 0)

    assert configuration.original_cpu_set == (1, 4, 9)
    assert configuration.requested_cpu_set == (1, 4)
    assert configuration.configured_cpu_set == (1, 4)
    assert configuration.excluded_cpu == 9
    assert configuration.cpu_num_threads == 2
    assert set_calls == [(101, {1, 4}), (102, {1, 4})]
    assert get_calls == [0, 101, 102]


def test_configure_cpu_affinity_linux_preserves_positive_thread_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux では正の cpu_num_threads を維持する。"""
    masks = iter(({0, 2, 6}, {0, 2}))
    thread_ids = (101,)
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_thread_ids",
        lambda: thread_ids,
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_sched_getaffinity",
        lambda pid: next(masks),
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_sched_setaffinity",
        lambda pid, mask: None,
    )

    configuration = cpu_affinity.configure_cpu_affinity("auto", 1)

    assert configuration.cpu_num_threads == 1


def test_configure_cpu_affinity_linux_rejects_mismatched_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux の CPU 集合読み戻し不一致は例外になる。"""
    masks = iter(({0, 2, 6}, {0, 2, 6}))
    thread_ids = (101,)
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_thread_ids",
        lambda: thread_ids,
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_sched_getaffinity",
        lambda pid: next(masks),
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_sched_setaffinity",
        lambda pid, mask: None,
    )

    with pytest.raises(RuntimeError, match="要求集合と一致しません"):
        cpu_affinity.configure_cpu_affinity("auto", None)


def test_configure_cpu_affinity_linux_retries_when_new_thread_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux では新しい TID が現れた場合に設定を再試行する。"""
    thread_id_sequences = iter(
        (
            (101, 102),
            (101, 102, 103),
            (101, 102, 103),
            (101, 102, 103),
            (101, 102, 103),
        )
    )
    masks = iter(({0, 2, 6}, {0, 2}, {0, 2}, {0, 2}))
    get_calls: list[int] = []
    set_calls: list[tuple[int, set[int]]] = []
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_thread_ids",
        lambda: next(thread_id_sequences),
    )

    def get_affinity(pid: int) -> set[int]:
        """CPU affinity API の呼び出しを記録する。"""
        get_calls.append(pid)
        return next(masks)

    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_sched_getaffinity",
        get_affinity,
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_sched_setaffinity",
        lambda pid, mask: set_calls.append((pid, set(mask))),
    )

    configuration = cpu_affinity.configure_cpu_affinity("auto", None)

    assert configuration.configured_cpu_set == (0, 2)
    assert set_calls == [
        (101, {0, 2}),
        (102, {0, 2}),
        (101, {0, 2}),
        (102, {0, 2}),
        (103, {0, 2}),
    ]
    assert get_calls == [0, 101, 102, 103]


def test_configure_cpu_affinity_linux_rejects_unstable_thread_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux では三回の再試行で TID 集合が安定しなければ例外になる。"""
    thread_id_sequences = iter(
        (
            (101,),
            (101, 102),
            (101, 102),
            (101, 102, 103),
            (101, 102, 103),
            (101, 102, 103, 104),
        )
    )
    masks = iter(({0, 2, 6},))
    set_calls: list[tuple[int, set[int]]] = []
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_thread_ids",
        lambda: next(thread_id_sequences),
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_sched_getaffinity",
        lambda pid: next(masks),
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._linux_sched_setaffinity",
        lambda pid, mask: set_calls.append((pid, set(mask))),
    )

    with pytest.raises(RuntimeError, match="安定しません"):
        cpu_affinity.configure_cpu_affinity("auto", None)

    assert set_calls == [
        (101, {0, 2}),
        (101, {0, 2}),
        (102, {0, 2}),
        (101, {0, 2}),
        (102, {0, 2}),
        (103, {0, 2}),
    ]


def test_configure_cpu_affinity_darwin_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS では affinity を設定せず非対応として継続する。"""
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity.platform.system",
        lambda: "Darwin",
    )

    configuration = cpu_affinity.configure_cpu_affinity("auto", None)

    assert configuration.status == "unsupported"
    assert configuration.cpu_num_threads is None


def test_configure_cpu_affinity_windows_excludes_maximum_cpu_and_uses_c_size_t(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows では最大 CPU を除外し c_size_t の mask を渡す。"""
    kernel32 = Mock()
    kernel32.GetActiveProcessorGroupCount.return_value = 1
    kernel32.GetCurrentProcess.return_value = object()
    kernel32.SetProcessAffinityMask.return_value = 1
    process_cpu_sets = iter(((1, 3, 7), (1, 3)))
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity.platform.system",
        lambda: "Windows",
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._windows_process_cpu_set",
        lambda _kernel32: next(process_cpu_sets),
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._windows_kernel32",
        lambda: kernel32,
    )

    configuration = cpu_affinity.configure_cpu_affinity("auto", 0)

    assert configuration.original_cpu_set == (1, 3, 7)
    assert configuration.requested_cpu_set == (1, 3)
    assert configuration.configured_cpu_set == (1, 3)
    assert configuration.excluded_cpu == 7
    assert configuration.cpu_num_threads == 2
    affinity_call = kernel32.SetProcessAffinityMask.call_args
    assert affinity_call is not None
    affinity_mask = affinity_call.args[1]
    assert isinstance(affinity_mask, ctypes.c_size_t)
    assert affinity_mask.value == 0b00001010


def test_configure_cpu_affinity_windows_multiple_groups_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """複数の Windows processor group は警告付き非対応になる。"""
    kernel32 = Mock()
    kernel32.GetActiveProcessorGroupCount.return_value = 2
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity.platform.system",
        lambda: "Windows",
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._windows_kernel32",
        lambda: kernel32,
    )

    with caplog.at_level(logging.WARNING):
        configuration = cpu_affinity.configure_cpu_affinity("auto", None)

    assert configuration.status == "unsupported"
    assert not kernel32.GetProcessAffinityMask.called
    assert not kernel32.SetProcessAffinityMask.called
    assert json.loads(caplog.records[-1].getMessage())["state"] == "unsupported"


def test_configure_cpu_affinity_windows_rejects_mismatched_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows の CPU 集合読み戻し不一致は例外になる。"""
    kernel32 = Mock()
    kernel32.GetActiveProcessorGroupCount.return_value = 1
    kernel32.GetCurrentProcess.return_value = object()
    kernel32.SetProcessAffinityMask.return_value = 1
    process_cpu_sets = iter(((0, 4, 6), (0, 4, 6)))
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity.platform.system",
        lambda: "Windows",
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._windows_process_cpu_set",
        lambda _kernel32: next(process_cpu_sets),
    )
    monkeypatch.setattr(
        "voicevox_engine.utility.cpu_affinity._windows_kernel32",
        lambda: kernel32,
    )

    with pytest.raises(RuntimeError, match="要求集合と一致しません"):
        cpu_affinity.configure_cpu_affinity("auto", None)
