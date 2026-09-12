"""`cancellable_engine.py` のテスト"""

from unittest.mock import MagicMock, patch

import pytest

from voicevox_engine.cancellable_engine import (
    CancellableEngine,
    start_synthesis_subprocess,
)
from voicevox_engine.core.cpu_execution import LegacyCpuExecutionPlan


def test_cancellable_engine_reuses_cpu_execution_plan_for_new_processes() -> None:
    """キャンセル可能エンジンが初回と再生成の子プロセスへ同じ計画を渡す。"""
    cpu_execution_plan = LegacyCpuExecutionPlan(3)
    processes = [MagicMock(), MagicMock(), MagicMock()]
    connections = [
        (MagicMock(), MagicMock()),
        (MagicMock(), MagicMock()),
        (MagicMock(), MagicMock()),
    ]
    processes[0].is_alive.return_value = False
    request = MagicMock()

    with patch(
        "voicevox_engine.cancellable_engine.Process", side_effect=processes
    ) as process:
        with patch("voicevox_engine.cancellable_engine.Pipe", side_effect=connections):
            engine = CancellableEngine(
                init_processes=2,
                use_gpu=False,
                enable_mock=True,
                cpu_execution_plan=cpu_execution_plan,
            )
            engine._finalize_con(request, processes[0], connections[0][0])

    assert engine.cpu_execution_plan is cpu_execution_plan
    assert process.call_count == 3
    assert all(
        call.kwargs["kwargs"]["cpu_execution_plan"] is cpu_execution_plan
        for call in process.call_args_list
    )


def test_start_synthesis_subprocess_initialization_order() -> None:
    """子プロセスが同じ計画を適用してから初期化と検証を行う。"""
    cpu_execution_plan = LegacyCpuExecutionPlan(3)
    connection = MagicMock()
    connection.recv.side_effect = RuntimeError("テスト終了")
    core_manager = MagicMock()
    tts_engines = MagicMock()
    tts_engines.versions.return_value = ["0.0.1"]
    events: list[str] = []

    def apply_plan(plan: LegacyCpuExecutionPlan) -> None:
        assert plan is cpu_execution_plan
        events.append("apply")

    def initialize(**kwargs: object) -> MagicMock:
        assert kwargs["cpu_num_threads"] == cpu_execution_plan.cpu_num_threads
        events.append("initialize")
        return core_manager

    def validate_plan(plan: LegacyCpuExecutionPlan) -> None:
        assert plan is cpu_execution_plan
        events.append("validate")

    with patch(
        "voicevox_engine.cancellable_engine.apply_cpu_execution_plan",
        side_effect=apply_plan,
    ):
        with patch(
            "voicevox_engine.cancellable_engine.initialize_cores",
            side_effect=initialize,
        ):
            with patch(
                "voicevox_engine.cancellable_engine.validate_cpu_execution_plan",
                side_effect=validate_plan,
            ):
                with patch(
                    "voicevox_engine.cancellable_engine.make_tts_engines_from_cores",
                    return_value=tts_engines,
                ):
                    with pytest.raises(RuntimeError, match="テスト終了"):
                        start_synthesis_subprocess(
                            use_gpu=False,
                            voicelib_dirs=None,
                            voicevox_dir=None,
                            runtime_dirs=None,
                            cpu_execution_plan=cpu_execution_plan,
                            enable_mock=True,
                            connection=connection,
                        )

    assert events == ["apply", "initialize", "validate"]
    connection.close.assert_called_once_with()
