"""`cpu_execution.py` のテスト"""

import pickle

import pytest

from voicevox_engine.core.cpu_execution import (
    HybridCpuTopology,
    LegacyCpuExecutionPlan,
    LinuxCpuExecutionPlan,
    WindowsCpuExecutionPlan,
    create_legacy_cpu_execution_plan,
    create_linux_cpu_execution_plan,
    create_windows_cpu_execution_plan,
)


def _create_topology(
    p_cores: tuple[tuple[int, ...], ...],
    e_core_tiers: tuple[tuple[tuple[int, ...], ...], ...],
) -> HybridCpuTopology:
    return HybridCpuTopology(p_cores, e_core_tiers)


@pytest.mark.parametrize(
    ("cpu_num_threads", "logical_cpu_count", "physical_cpu_count", "expected"),
    [
        (None, 8, 8, 4),
        (0, 8, 8, 4),
        (None, 9, 9, 4),
        (None, 8, 4, 0),
        (None, 8, None, 4),
        (None, None, None, 0),
        (4, 8, 4, 4),
        (65535, None, None, 65535),
    ],
)
def test_create_legacy_cpu_execution_plan(
    cpu_num_threads: int | None,
    logical_cpu_count: int | None,
    physical_cpu_count: int | None,
    expected: int,
) -> None:
    """レガシーCPU実行計画を生成できる。"""
    plan = create_legacy_cpu_execution_plan(
        cpu_num_threads,
        logical_cpu_count,
        physical_cpu_count,
    )

    assert plan == LegacyCpuExecutionPlan(expected)


@pytest.mark.parametrize("cpu_num_threads", [True, False, -1, 65536])
def test_create_legacy_cpu_execution_plan_rejects_invalid_value(
    cpu_num_threads: int,
) -> None:
    """レガシーCPU実行計画が不正なCPUスレッド数を拒否する。"""
    with pytest.raises(ValueError, match="cpu_num_threads"):
        create_legacy_cpu_execution_plan(cpu_num_threads, 8, 8)


@pytest.mark.parametrize(
    ("logical_cpu_count", "physical_cpu_count"),
    [
        (0, None),
        (-1, None),
        (True, None),
        ("8", None),
        (None, 0),
        (None, -1),
        (None, False),
        (None, "8"),
        (8, 9),
    ],
)
def test_create_legacy_cpu_execution_plan_rejects_invalid_cpu_counts(
    logical_cpu_count: int | None,
    physical_cpu_count: int | None,
) -> None:
    """レガシーCPU実行計画が不正なCPU数を拒否する。"""
    with pytest.raises(ValueError, match="CPU|cpu_count"):
        create_legacy_cpu_execution_plan(
            None,
            logical_cpu_count,
            physical_cpu_count,
        )


@pytest.mark.parametrize("cpu_num_threads", [1, 65535])
def test_create_legacy_cpu_execution_plan_explicit_value_ignores_cpu_counts(
    cpu_num_threads: int,
) -> None:
    """レガシーCPU実行計画が明示値ならCPU数を参照しない。"""
    plan = create_legacy_cpu_execution_plan(cpu_num_threads, 0, 0)

    assert plan == LegacyCpuExecutionPlan(cpu_num_threads)


def test_create_windows_cpu_execution_plan_auto_uses_distinct_p_cores() -> None:
    """Windowsの自動計画が異なるPコアから論理CPUを選ぶ。"""
    topology = _create_topology(
        tuple((core_id, core_id + 100) for core_id in range(8)),
        (((1000,),),),
    )

    plan = create_windows_cpu_execution_plan(None, topology)

    assert plan == WindowsCpuExecutionPlan(7, (0, 1, 2, 3, 4, 5, 6))


@pytest.mark.parametrize(
    ("p_cores", "expected_cpu_num_threads", "expected_logical_cpu_ids"),
    [
        (((0,), (1,), (2,), (3,)), 2, (0, 1)),
        (((10, 11), (20, 21), (30,)), 2, (10, 20)),
    ],
)
def test_create_linux_cpu_execution_plan_auto(
    p_cores: tuple[tuple[int, ...], ...],
    expected_cpu_num_threads: int,
    expected_logical_cpu_ids: tuple[int, ...],
) -> None:
    """Linuxの自動計画がPコア数に応じた論理CPUを選ぶ。"""
    topology = _create_topology(p_cores, (((1000,),),))

    plan = create_linux_cpu_execution_plan(0, topology)

    assert plan == LinuxCpuExecutionPlan(
        expected_cpu_num_threads,
        expected_logical_cpu_ids,
    )


def test_create_linux_cpu_execution_plan_auto_rejects_one_p_core() -> None:
    """Linuxの自動計画がPコア一つの構成を拒否する。"""
    topology = _create_topology(((10, 20),), (((30,),),))

    with pytest.raises(ValueError, match="Pコア"):
        create_linux_cpu_execution_plan(None, topology)


@pytest.mark.parametrize("e_core_tiers", [(), ((),)])
def test_create_linux_cpu_execution_plan_rejects_missing_e_cores(
    e_core_tiers: tuple[tuple[tuple[int, ...], ...], ...],
) -> None:
    """Linuxの計画がEコアのない構成を拒否する。"""
    topology = _create_topology(((10,), (20,)), e_core_tiers)

    with pytest.raises(ValueError, match="Eコア"):
        create_linux_cpu_execution_plan(1, topology)


def test_create_windows_cpu_execution_plan_selects_p_then_e() -> None:
    """Windowsの明示計画がPコアとEコアを性能順に選ぶ。"""
    topology = _create_topology(
        ((21, 5, 13), (40, 8)),
        (
            ((100, 64, 88), (120, 112)),
            ((200, 184),),
        ),
    )

    plan = create_windows_cpu_execution_plan(11, topology)

    assert plan == WindowsCpuExecutionPlan(
        11,
        (5, 8, 13, 21, 40, 64, 112, 88, 100, 120, 184),
    )


def test_create_windows_cpu_execution_plan_sorts_physical_cores() -> None:
    """Windowsの計画が物理コアの入力順によらず同じ結果になる。"""
    topology = _create_topology(
        ((40, 8), (21, 5, 13)),
        (
            ((120, 112), (100, 64, 88)),
            ((200, 184),),
        ),
    )

    plan = create_windows_cpu_execution_plan(11, topology)

    assert plan == WindowsCpuExecutionPlan(
        11,
        (5, 8, 13, 21, 40, 64, 112, 88, 100, 120, 184),
    )


def test_create_linux_cpu_execution_plan_allows_explicit_one_p_core() -> None:
    """Linuxの明示計画がPコア一つの構成を受け入れる。"""
    topology = _create_topology(((10, 20),), (((30,),),))

    plan = create_linux_cpu_execution_plan(2, topology)

    assert plan == LinuxCpuExecutionPlan(2, (10, 20))


@pytest.mark.parametrize("p_cores", [((0,), (0,)), ((), (1,))])
def test_create_windows_cpu_execution_plan_rejects_invalid_p_cores(
    p_cores: tuple[tuple[int, ...], ...],
) -> None:
    """Windowsの計画が重複または空のPコアを拒否する。"""
    topology = _create_topology(p_cores, (((100,),),))

    with pytest.raises(ValueError, match="Pコア|論理CPU ID"):
        create_windows_cpu_execution_plan(1, topology)


def test_create_linux_cpu_execution_plan_supports_non_contiguous_ids() -> None:
    """Linuxの計画が非連続な論理CPU IDと部分的なSMTを扱う。"""
    topology = _create_topology(
        ((17, 3), (42,)),
        (((99,),),),
    )

    plan = create_linux_cpu_execution_plan(3, topology)

    assert plan == LinuxCpuExecutionPlan(3, (3, 42, 17))


@pytest.mark.parametrize("cpu_num_threads", [65535, 65536])
def test_create_linux_cpu_execution_plan_validates_capacity(
    cpu_num_threads: int,
) -> None:
    """Linuxの計画が指定値と利用可能な論理CPU数を検証する。"""
    topology = _create_topology(((0,), (1,)), (((100,),),))

    if cpu_num_threads == 65535:
        with pytest.raises(ValueError, match="利用可能"):
            create_linux_cpu_execution_plan(cpu_num_threads, topology)
    else:
        with pytest.raises(ValueError, match="cpu_num_threads"):
            create_linux_cpu_execution_plan(cpu_num_threads, topology)


@pytest.mark.parametrize(
    "plan",
    [
        LegacyCpuExecutionPlan(4),
        WindowsCpuExecutionPlan(2, (0, 1)),
        LinuxCpuExecutionPlan(2, (10, 20)),
    ],
)
def test_cpu_execution_plan_is_pickleable(
    plan: LegacyCpuExecutionPlan | WindowsCpuExecutionPlan | LinuxCpuExecutionPlan,
) -> None:
    """CPU実行計画がpickle化できる。"""
    assert pickle.loads(pickle.dumps(plan)) == plan
