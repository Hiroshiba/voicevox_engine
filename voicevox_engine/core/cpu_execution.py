"""CPU実行計画の値と選択処理"""

import platform
from dataclasses import dataclass

import psutil

from voicevox_engine.utility.error_utility import UnreachableError

type PhysicalCore = tuple[int, ...]
type ECoreTier = tuple[PhysicalCore, ...]


@dataclass(frozen=True)
class HybridCpuTopology:
    """Pコアと高性能順に並べたEコア性能階層の論理CPU構成を表す。"""

    p_cores: tuple[PhysicalCore, ...]
    e_core_tiers: tuple[ECoreTier, ...]


@dataclass(frozen=True)
class LegacyCpuExecutionPlan:
    """レガシーCPU実行計画を表す。"""

    cpu_num_threads: int


@dataclass(frozen=True)
class WindowsCpuExecutionPlan:
    """WindowsのCPU実行計画を表す。"""

    cpu_num_threads: int
    logical_processor_indices: tuple[int, ...]


@dataclass(frozen=True)
class LinuxCpuExecutionPlan:
    """LinuxのCPU実行計画を表す。"""

    cpu_num_threads: int
    logical_cpu_ids: tuple[int, ...]


type CpuExecutionPlan = (
    LegacyCpuExecutionPlan | WindowsCpuExecutionPlan | LinuxCpuExecutionPlan
)


def _validate_cpu_num_threads(cpu_num_threads: int | None) -> int | None:
    if cpu_num_threads is None:
        return None
    if isinstance(cpu_num_threads, bool) or not isinstance(cpu_num_threads, int):
        raise ValueError("cpu_num_threadsは整数、None、または0で指定してください。")
    if cpu_num_threads == 0:
        return None
    if not 1 <= cpu_num_threads <= 65535:
        raise ValueError("cpu_num_threadsは0、または1以上65535以下で指定してください。")
    return cpu_num_threads


def _validate_cpu_count(cpu_count: int | None, name: str) -> int | None:
    if cpu_count is None:
        return None
    if isinstance(cpu_count, bool) or not isinstance(cpu_count, int) or cpu_count < 1:
        raise ValueError(f"{name}はNoneまたは1以上の整数で指定してください。")
    return cpu_count


def _normalize_physical_core(core: PhysicalCore, name: str) -> PhysicalCore:
    if len(core) == 0:
        raise ValueError(f"{name}には論理CPUが1つ以上必要です。")
    if any(
        isinstance(logical_cpu_id, bool)
        or not isinstance(logical_cpu_id, int)
        or logical_cpu_id < 0
        for logical_cpu_id in core
    ):
        raise ValueError(f"{name}の論理CPU IDは0以上の整数で指定してください。")
    normalized_core = tuple(sorted(core))
    if len(set(normalized_core)) != len(normalized_core):
        raise ValueError(f"{name}の論理CPU IDに重複があります。")
    return normalized_core


def _normalize_topology(
    topology: HybridCpuTopology,
) -> tuple[tuple[PhysicalCore, ...], tuple[ECoreTier, ...]]:
    if len(topology.p_cores) == 0:
        raise ValueError("Pコアが1つ以上必要です。")
    p_cores = tuple(
        sorted(
            (_normalize_physical_core(core, "Pコア") for core in topology.p_cores),
            key=lambda core: core[0],
        )
    )
    if len(topology.e_core_tiers) == 0:
        raise ValueError("Eコアの性能階層が1つ以上必要です。")
    e_core_tiers: list[ECoreTier] = []
    for tier_index, tier in enumerate(topology.e_core_tiers):
        if len(tier) == 0:
            raise ValueError(
                f"Eコア性能階層{tier_index}には物理コアが1つ以上必要です。"
            )
        e_core_tiers.append(
            tuple(
                sorted(
                    (
                        _normalize_physical_core(
                            core, f"Eコア性能階層{tier_index}の物理コア"
                        )
                        for core in tier
                    ),
                    key=lambda core: core[0],
                )
            )
        )
    normalized_e_core_tiers = tuple(e_core_tiers)
    logical_cpu_ids = tuple(
        logical_cpu_id for core in p_cores for logical_cpu_id in core
    ) + tuple(
        logical_cpu_id
        for tier in normalized_e_core_tiers
        for core in tier
        for logical_cpu_id in core
    )
    if len(set(logical_cpu_ids)) != len(logical_cpu_ids):
        raise ValueError("論理CPU IDに重複があります。")
    return p_cores, normalized_e_core_tiers


def _ordered_hybrid_logical_cpu_ids(
    p_cores: tuple[PhysicalCore, ...],
    e_core_tiers: tuple[ECoreTier, ...],
) -> tuple[int, ...]:
    p_primary_ids = tuple(core[0] for core in p_cores)
    p_sibling_ids = tuple(
        logical_cpu_id for core in p_cores for logical_cpu_id in core[1:]
    )
    e_ids_list: list[int] = []
    for tier in e_core_tiers:
        e_ids_list.extend(core[0] for core in tier)
        e_ids_list.extend(
            logical_cpu_id for core in tier for logical_cpu_id in core[1:]
        )
    e_ids = tuple(e_ids_list)
    return p_primary_ids + p_sibling_ids + e_ids


def _resolve_hybrid_cpu_num_threads(
    requested_cpu_num_threads: int | None,
    p_cores: tuple[PhysicalCore, ...],
    ordered_logical_cpu_ids: tuple[int, ...],
) -> int:
    if requested_cpu_num_threads is not None:
        if requested_cpu_num_threads > len(ordered_logical_cpu_ids):
            raise ValueError(
                "指定されたCPUスレッド数が利用可能な論理CPU数を超えています。"
            )
        return requested_cpu_num_threads

    p_physical_count = len(p_cores)
    if p_physical_count == 1:
        raise ValueError("Pコアが1つの場合はCPUスレッド数を自動決定できません。")
    p_logical_count = sum(len(core) for core in p_cores)
    if p_logical_count > p_physical_count:
        base_cpu_num_threads = p_physical_count
    elif p_logical_count == p_physical_count:
        base_cpu_num_threads = p_logical_count // 2
    else:
        raise ValueError("Pコアの論理CPU数が物理コア数を下回っています。")
    return min(base_cpu_num_threads, p_physical_count - 1)


def create_legacy_cpu_execution_plan(
    cpu_num_threads: int | None,
    logical_cpu_count: int | None,
    physical_cpu_count: int | None,
) -> LegacyCpuExecutionPlan:
    """レガシーCPU実行計画を生成する。"""
    validated_cpu_num_threads = _validate_cpu_num_threads(cpu_num_threads)
    if validated_cpu_num_threads is not None:
        return LegacyCpuExecutionPlan(validated_cpu_num_threads)
    logical_cpu_count = _validate_cpu_count(logical_cpu_count, "logical_cpu_count")
    physical_cpu_count = _validate_cpu_count(physical_cpu_count, "physical_cpu_count")
    if (
        logical_cpu_count is not None
        and physical_cpu_count is not None
        and physical_cpu_count > logical_cpu_count
    ):
        raise ValueError("物理CPU数が論理CPU数を超えています。")
    if logical_cpu_count is None:
        resolved_cpu_num_threads = 0
    elif physical_cpu_count is not None and physical_cpu_count != logical_cpu_count:
        resolved_cpu_num_threads = 0
    else:
        resolved_cpu_num_threads = logical_cpu_count // 2
    return LegacyCpuExecutionPlan(resolved_cpu_num_threads)


def _create_hybrid_cpu_execution_values(
    cpu_num_threads: int | None,
    topology: HybridCpuTopology,
) -> tuple[int, tuple[int, ...]]:
    validated_cpu_num_threads = _validate_cpu_num_threads(cpu_num_threads)
    p_cores, e_core_tiers = _normalize_topology(topology)
    ordered_logical_cpu_ids = _ordered_hybrid_logical_cpu_ids(p_cores, e_core_tiers)
    resolved_cpu_num_threads = _resolve_hybrid_cpu_num_threads(
        validated_cpu_num_threads,
        p_cores,
        ordered_logical_cpu_ids,
    )
    return resolved_cpu_num_threads, ordered_logical_cpu_ids[:resolved_cpu_num_threads]


def create_windows_cpu_execution_plan(
    cpu_num_threads: int | None,
    topology: HybridCpuTopology,
) -> WindowsCpuExecutionPlan:
    """WindowsのCPU実行計画を生成する。"""
    resolved_cpu_num_threads, logical_processor_indices = (
        _create_hybrid_cpu_execution_values(cpu_num_threads, topology)
    )
    return WindowsCpuExecutionPlan(
        resolved_cpu_num_threads,
        logical_processor_indices,
    )


def create_linux_cpu_execution_plan(
    cpu_num_threads: int | None,
    topology: HybridCpuTopology,
) -> LinuxCpuExecutionPlan:
    """LinuxのCPU実行計画を生成する。"""
    resolved_cpu_num_threads, logical_cpu_ids = _create_hybrid_cpu_execution_values(
        cpu_num_threads, topology
    )
    return LinuxCpuExecutionPlan(resolved_cpu_num_threads, logical_cpu_ids)


def _create_legacy_cpu_execution_plan_from_system(
    cpu_num_threads: int | None,
) -> LegacyCpuExecutionPlan:
    if cpu_num_threads is not None:
        return LegacyCpuExecutionPlan(cpu_num_threads)
    return create_legacy_cpu_execution_plan(
        cpu_num_threads,
        psutil.cpu_count(logical=True),
        psutil.cpu_count(logical=False),
    )


def create_cpu_execution_plan(
    cpu_num_threads: int | None,
) -> CpuExecutionPlan:
    """実行環境に応じたCPU実行計画を生成する。"""
    validated_cpu_num_threads = _validate_cpu_num_threads(cpu_num_threads)
    system = platform.system()
    if system == "Windows":
        from voicevox_engine.core.cpu_execution_windows import (
            detect_windows_hybrid_cpu_topology,
        )

        topology = detect_windows_hybrid_cpu_topology()
        if topology is None:
            return _create_legacy_cpu_execution_plan_from_system(
                validated_cpu_num_threads
            )
        return create_windows_cpu_execution_plan(validated_cpu_num_threads, topology)
    if system == "Linux":
        from voicevox_engine.core.cpu_execution_linux import (
            detect_linux_hybrid_cpu_topology,
        )

        topology = detect_linux_hybrid_cpu_topology()
        if topology is None:
            return _create_legacy_cpu_execution_plan_from_system(
                validated_cpu_num_threads
            )
        return create_linux_cpu_execution_plan(validated_cpu_num_threads, topology)
    if system == "Darwin":
        return _create_legacy_cpu_execution_plan_from_system(validated_cpu_num_threads)
    raise RuntimeError(f"対応していないOSです: {system}")


def apply_cpu_execution_plan(plan: CpuExecutionPlan) -> None:
    """CPU実行計画を現在のプロセスへ適用する。"""
    if isinstance(plan, LegacyCpuExecutionPlan):
        return
    if isinstance(plan, WindowsCpuExecutionPlan):
        from voicevox_engine.core.cpu_execution_windows import (
            apply_windows_cpu_execution_plan,
        )

        apply_windows_cpu_execution_plan(plan)
        return
    if isinstance(plan, LinuxCpuExecutionPlan):
        from voicevox_engine.core.cpu_execution_linux import (
            apply_linux_cpu_execution_plan,
        )

        apply_linux_cpu_execution_plan(plan)
        return
    raise UnreachableError("CPU実行計画の型が不正です。")


def validate_cpu_execution_plan(plan: CpuExecutionPlan) -> None:
    """CPU実行計画が現在のCPU affinityへ反映されていることを検証する。"""
    if isinstance(plan, LegacyCpuExecutionPlan):
        return
    if isinstance(plan, WindowsCpuExecutionPlan):
        from voicevox_engine.core.cpu_execution_windows import (
            validate_windows_cpu_execution_plan,
        )

        validate_windows_cpu_execution_plan(plan)
        return
    if isinstance(plan, LinuxCpuExecutionPlan):
        from voicevox_engine.core.cpu_execution_linux import (
            validate_linux_cpu_execution_plan,
        )

        validate_linux_cpu_execution_plan(plan)
        return
    raise UnreachableError("CPU実行計画の型が不正です。")
