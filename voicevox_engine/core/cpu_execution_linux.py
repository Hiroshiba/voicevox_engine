"""LinuxのIntelハイブリッドCPU構成を取得し、CPU affinityを操作する。"""

import errno
import os
import platform
from pathlib import Path

from voicevox_engine.core.cpu_execution import (
    HybridCpuTopology,
    LinuxCpuExecutionPlan,
)

_CPU_CORE_CPUS_PATH = Path("/sys/devices/cpu_core/cpus")
_CPU_ATOM_CPUS_PATH = Path("/sys/devices/cpu_atom/cpus")
_ONLINE_CPUS_PATH = Path("/sys/devices/system/cpu/online")
_CPU_SYSFS_PATH = Path("/sys/devices/system/cpu")
_TASK_PATH = Path("/proc/self/task")
_MAX_RETRIES = 3
_X86_MACHINES = {
    "amd64",
    "i386",
    "i486",
    "i586",
    "i686",
    "x86",
    "x86_64",
}


class _LinuxAffinityRace(Exception):
    pass


def _parse_cpu_list(value: str) -> set[int]:
    text = value.strip()
    if len(text) == 0:
        raise ValueError("LinuxのCPUリストが空です。")
    result: set[int] = set()
    for item in text.split(","):
        if not item.isdecimal():
            if item.count("-") != 1:
                raise ValueError("LinuxのCPUリストの形式が不正です。")
            start_text, end_text = item.split("-")
            if not start_text.isdecimal() or not end_text.isdecimal():
                raise ValueError("LinuxのCPUリストの形式が不正です。")
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError("LinuxのCPUリストの範囲が不正です。")
            values = set(range(start, end + 1))
        else:
            values = {int(item)}
        if result & values:
            raise ValueError("LinuxのCPUリストに重複があります。")
        result.update(values)
    return result


def _read_cpu_list(path: Path) -> set[int]:
    try:
        value = path.read_text(encoding="ascii")
    except OSError as error:
        if _is_esrch(error):
            raise _LinuxAffinityRace from error
        raise
    return _parse_cpu_list(value)


def _is_x86() -> bool:
    return platform.machine().lower() in _X86_MACHINES


def _is_esrch(error: BaseException) -> bool:
    return (
        isinstance(error, ProcessLookupError)
        or getattr(error, "errno", None) == errno.ESRCH
    )


def _list_thread_ids() -> tuple[int, ...]:
    try:
        entries = tuple(_TASK_PATH.iterdir())
    except OSError as error:
        if _is_esrch(error):
            raise _LinuxAffinityRace from error
        raise
    thread_ids: list[int] = []
    for entry in entries:
        if not entry.name.isdecimal():
            raise ValueError("LinuxのプロセスTID一覧に不正な名前があります。")
        thread_ids.append(int(entry.name))
    if len(thread_ids) == 0:
        raise RuntimeError("LinuxのプロセスTIDが存在しません。")
    return tuple(sorted(thread_ids))


def _get_thread_affinity(thread_id: int) -> frozenset[int]:
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is None:
        raise RuntimeError("Linuxのsched_getaffinityが利用できません。")
    try:
        affinity = frozenset(int(cpu_id) for cpu_id in sched_getaffinity(thread_id))
    except OSError:
        raise
    if len(affinity) == 0:
        raise ValueError(f"LinuxのTID {thread_id} のCPU affinityが空です。")
    if any(cpu_id < 0 for cpu_id in affinity):
        raise ValueError(f"LinuxのTID {thread_id} のCPU affinityが不正です。")
    return affinity


def _set_thread_affinity(thread_id: int, affinity: frozenset[int]) -> None:
    sched_setaffinity = getattr(os, "sched_setaffinity", None)
    if sched_setaffinity is None:
        raise RuntimeError("Linuxのsched_setaffinityが利用できません。")
    sched_setaffinity(thread_id, affinity)


def _read_thread_affinities(
    thread_ids: tuple[int, ...],
) -> dict[int, frozenset[int]]:
    affinities: dict[int, frozenset[int]] = {}
    for thread_id in thread_ids:
        try:
            affinities[thread_id] = _get_thread_affinity(thread_id)
        except OSError as error:
            if _is_esrch(error):
                raise _LinuxAffinityRace from error
            raise
    return affinities


def _snapshot_detection_state() -> tuple[set[int], dict[int, frozenset[int]]]:
    online_before = _read_cpu_list(_ONLINE_CPUS_PATH)
    thread_ids_before = _list_thread_ids()
    try:
        affinities = _read_thread_affinities(thread_ids_before)
    except _LinuxAffinityRace:
        raise
    online_after = _read_cpu_list(_ONLINE_CPUS_PATH)
    thread_ids_after = _list_thread_ids()
    if online_before != online_after or thread_ids_before != thread_ids_after:
        raise _LinuxAffinityRace
    return online_after, affinities


def _ensure_detection_state_is_stable(
    p_cpus: set[int],
    e_cpus: set[int],
    online_cpus: set[int],
) -> None:
    if _CPU_CORE_CPUS_PATH.exists() is False or _CPU_ATOM_CPUS_PATH.exists() is False:
        raise _LinuxAffinityRace
    if _read_cpu_list(_CPU_CORE_CPUS_PATH) != p_cpus:
        raise _LinuxAffinityRace
    if _read_cpu_list(_CPU_ATOM_CPUS_PATH) != e_cpus:
        raise _LinuxAffinityRace
    if _read_cpu_list(_ONLINE_CPUS_PATH) != online_cpus:
        raise _LinuxAffinityRace


def _read_thread_siblings(cpu_id: int) -> frozenset[int]:
    path = _CPU_SYSFS_PATH / f"cpu{cpu_id}" / "topology" / "thread_siblings_list"
    try:
        siblings = frozenset(_parse_cpu_list(path.read_text(encoding="ascii")))
    except OSError as error:
        if _is_esrch(error):
            raise _LinuxAffinityRace from error
        raise
    if cpu_id not in siblings:
        raise ValueError(
            f"LinuxのCPU {cpu_id} のthread_siblings_listに自身がありません。"
        )
    return siblings


def _build_physical_cores(
    available_cpus: set[int],
    p_cpus: set[int],
    e_cpus: set[int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    sibling_lists = {
        cpu_id: _read_thread_siblings(cpu_id) for cpu_id in sorted(available_cpus)
    }
    for _cpu_id, siblings in sibling_lists.items():
        sibling_classes = {
            "p" if sibling in p_cpus else "e"
            for sibling in siblings
            if sibling in p_cpus or sibling in e_cpus
        }
        if len(sibling_classes) > 1:
            raise ValueError("Linuxのthread_siblings_listがP/Eコアをまたいでいます。")
        available_siblings = siblings & available_cpus
        for sibling in available_siblings:
            if sibling not in sibling_lists:
                raise ValueError("Linuxのthread_siblings_listの対応が不完全です。")
            if sibling_lists[sibling] & available_cpus != available_siblings:
                raise ValueError("Linuxのthread_siblings_listが相互に不整合です。")
    p_cores: set[frozenset[int]] = set()
    e_cores: set[frozenset[int]] = set()
    for cpu_id, siblings in sibling_lists.items():
        core = frozenset(siblings & available_cpus)
        if cpu_id in p_cpus:
            p_cores.add(core)
        elif cpu_id in e_cpus:
            e_cores.add(core)
        else:
            raise ValueError(f"LinuxのCPU {cpu_id} のP/E分類がありません。")
    return (
        tuple(sorted(tuple(sorted(core)) for core in p_cores)),
        tuple(sorted(tuple(sorted(core)) for core in e_cores)),
    )


def _detect_linux_hybrid_cpu_topology_once() -> HybridCpuTopology | None:
    core_exists = _CPU_CORE_CPUS_PATH.exists()
    atom_exists = _CPU_ATOM_CPUS_PATH.exists()
    if core_exists is False or atom_exists is False:
        if core_exists is False and atom_exists is False:
            return None
        raise ValueError("Linuxのcpu_core/cpusとcpu_atom/cpusの片方がありません。")
    p_cpus = _read_cpu_list(_CPU_CORE_CPUS_PATH)
    e_cpus = _read_cpu_list(_CPU_ATOM_CPUS_PATH)
    if p_cpus & e_cpus:
        raise ValueError("Linuxのcpu_core/cpusとcpu_atom/cpusに重複があります。")
    online_cpus, thread_affinities = _snapshot_detection_state()
    available_cpus = set(online_cpus)
    for affinity in thread_affinities.values():
        available_cpus.intersection_update(affinity)
    if len(available_cpus) == 0:
        _ensure_detection_state_is_stable(p_cpus, e_cpus, online_cpus)
        return None
    unknown_cpus = available_cpus - p_cpus - e_cpus
    if unknown_cpus:
        raise ValueError("Linuxの利用可能CPUにP/E分類のないCPUがあります。")
    available_p_cpus = available_cpus & p_cpus
    available_e_cpus = available_cpus & e_cpus
    if len(available_p_cpus) == 0 or len(available_e_cpus) == 0:
        _ensure_detection_state_is_stable(p_cpus, e_cpus, online_cpus)
        return None
    p_cores, e_cores = _build_physical_cores(
        available_cpus,
        p_cpus,
        e_cpus,
    )
    _ensure_detection_state_is_stable(p_cpus, e_cpus, online_cpus)
    if len(p_cores) == 0 or len(e_cores) == 0:
        return None
    return HybridCpuTopology(p_cores, (e_cores,))


def detect_linux_hybrid_cpu_topology() -> HybridCpuTopology | None:
    """LinuxのsysfsとTID affinityからIntel P/Eコア構成を検出する。"""
    if not _is_x86():
        return None
    for _ in range(_MAX_RETRIES):
        try:
            return _detect_linux_hybrid_cpu_topology_once()
        except _LinuxAffinityRace:
            continue
    raise RuntimeError("LinuxのCPU構成が安定しないため、P/Eコアを検出できません。")


def _plan_cpu_ids(plan: LinuxCpuExecutionPlan) -> frozenset[int]:
    if (
        isinstance(plan.cpu_num_threads, bool)
        or not isinstance(plan.cpu_num_threads, int)
        or plan.cpu_num_threads < 1
    ):
        raise ValueError("LinuxのCPUスレッド数は1以上の整数で指定してください。")
    if len(plan.logical_cpu_ids) == 0:
        raise ValueError("Linuxの対象論理CPUが空です。")
    if any(
        isinstance(cpu_id, bool) or not isinstance(cpu_id, int) or cpu_id < 0
        for cpu_id in plan.logical_cpu_ids
    ):
        raise ValueError("Linuxの対象論理CPU IDが不正です。")
    target = frozenset(plan.logical_cpu_ids)
    if len(target) != len(plan.logical_cpu_ids):
        raise ValueError("Linuxの対象論理CPU IDに重複があります。")
    if len(target) != plan.cpu_num_threads:
        raise ValueError("LinuxのCPUスレッド数と論理CPU IDの数が一致しません。")
    return target


def _rollback_thread_affinities(
    original_affinities: dict[int, frozenset[int]],
    original_error: Exception,
) -> None:
    try:
        for thread_id, affinity in original_affinities.items():
            try:
                _set_thread_affinity(thread_id, affinity)
                if _get_thread_affinity(thread_id) != affinity:
                    raise RuntimeError(
                        f"LinuxのTID {thread_id} のaffinityを復元できません。"
                    )
            except OSError as error:
                if _is_esrch(error):
                    continue
                raise
    except Exception as rollback_error:
        raise RuntimeError(
            "LinuxのCPU affinity適用に失敗し、ロールバック後の状態を確認できません。"
            f" 元のエラー: {original_error}"
        ) from rollback_error


def apply_linux_cpu_execution_plan(plan: LinuxCpuExecutionPlan) -> None:
    """LinuxのCPU実行計画をプロセス内の全TIDへ適用する。"""
    target = _plan_cpu_ids(plan)
    original_affinities: dict[int, frozenset[int]] = {}
    mutation_started = False
    try:
        for _ in range(_MAX_RETRIES):
            try:
                thread_ids = _list_thread_ids()
            except _LinuxAffinityRace:
                continue
            try:
                current_affinities = _read_thread_affinities(thread_ids)
            except _LinuxAffinityRace:
                continue
            for thread_id, affinity in current_affinities.items():
                if thread_id not in original_affinities:
                    original_affinities[thread_id] = affinity
                if not target.issubset(affinity):
                    raise ValueError(
                        f"LinuxのTID {thread_id} の既存affinityが対象CPU集合を許可していません。"
                    )
            try:
                for thread_id, affinity in current_affinities.items():
                    if affinity != target:
                        mutation_started = True
                        _set_thread_affinity(thread_id, target)
            except OSError as error:
                if _is_esrch(error):
                    continue
                raise
            try:
                final_thread_ids = _list_thread_ids()
            except _LinuxAffinityRace:
                continue
            if final_thread_ids != thread_ids:
                continue
            try:
                final_affinities = _read_thread_affinities(final_thread_ids)
            except _LinuxAffinityRace:
                continue
            if all(affinity == target for affinity in final_affinities.values()):
                return
        raise RuntimeError(
            "LinuxのプロセスTIDが安定しないため、CPU affinityを適用できません。"
        )
    except Exception as error:
        if mutation_started:
            _rollback_thread_affinities(original_affinities, error)
        raise


def validate_linux_cpu_execution_plan(plan: LinuxCpuExecutionPlan) -> None:
    """Linuxの全TIDのaffinityがCPU実行計画と一致することを検証する。"""
    target = _plan_cpu_ids(plan)
    for _ in range(_MAX_RETRIES):
        try:
            thread_ids = _list_thread_ids()
        except _LinuxAffinityRace:
            continue
        try:
            affinities = _read_thread_affinities(thread_ids)
        except _LinuxAffinityRace:
            continue
        try:
            final_thread_ids = _list_thread_ids()
        except _LinuxAffinityRace:
            continue
        if final_thread_ids != thread_ids:
            continue
        if any(affinity != target for affinity in affinities.values()):
            raise RuntimeError(
                "Linuxの全TIDのCPU affinityがCPU実行計画と一致しません。"
            )
        return
    raise RuntimeError(
        "LinuxのプロセスTIDが安定しないため、CPU affinityを検証できません。"
    )
