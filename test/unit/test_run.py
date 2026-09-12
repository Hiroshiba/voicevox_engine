"""`run.py` のテスト"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import run
from voicevox_engine.core.cpu_execution import LegacyCpuExecutionPlan
from voicevox_engine.setting.model import CorsPolicyMode


def test_main_reuses_cpu_execution_plan_for_core_and_cancellable_engine(
    tmp_path: Path,
) -> None:
    """mainが同じCPU実行計画を初期化とキャンセル可能エンジンへ渡す。"""
    args = run._CLIArgs(
        host=None,
        port=None,
        use_gpu=False,
        voicevox_dir=tmp_path,
        voicelib_dirs=None,
        runtime_dirs=None,
        enable_mock=True,
        enable_cancellable_synthesis=True,
        init_processes=1,
        load_all_models=False,
        cpu_num_threads=None,
        output_log_utf8=False,
        cors_policy_mode=CorsPolicyMode.localapps,
        allow_origins=None,
        setting_file=tmp_path / "setting.yml",
        preset_file=tmp_path / "preset.yml",
        disable_mutable_api=False,
    )
    envs = run.Envs(
        output_log_utf8=False,
        cpu_num_threads=None,
        env_preset_path=None,
        disable_mutable_api=False,
        host=None,
        port=None,
        use_gpu=False,
    )
    cpu_execution_plan = LegacyCpuExecutionPlan(3)
    core_manager = MagicMock()
    tts_engines = MagicMock()
    tts_engines.versions.return_value = ["0.0.1"]
    song_engines = MagicMock()
    song_engines.versions.return_value = ["0.0.1"]
    settings = MagicMock()
    settings.cors_policy_mode = CorsPolicyMode.localapps
    settings.allow_origin = None
    engine_manifest = MagicMock()
    events: list[str] = []

    def create_plan(cpu_num_threads: int | None) -> LegacyCpuExecutionPlan:
        assert cpu_num_threads is None
        events.append("create")
        return cpu_execution_plan

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

    def create_cancellable_engine(**kwargs: object) -> MagicMock:
        assert kwargs["cpu_execution_plan"] is cpu_execution_plan
        events.append("cancellable")
        return MagicMock()

    with patch.object(run, "multiprocessing") as multiprocessing:
        with patch.object(run, "read_environment_variables", return_value=envs):
            with patch.object(run, "read_cli_arguments", return_value=args):
                with patch.object(
                    run, "create_cpu_execution_plan", side_effect=create_plan
                ) as create_plan_mock:
                    with patch.object(
                        run, "apply_cpu_execution_plan", side_effect=apply_plan
                    ) as apply_plan_mock:
                        with patch.object(
                            run, "initialize_cores", side_effect=initialize
                        ) as initialize_mock:
                            with patch.object(
                                run,
                                "validate_cpu_execution_plan",
                                side_effect=validate_plan,
                            ) as validate_plan_mock:
                                with patch.object(
                                    run,
                                    "CancellableEngine",
                                    side_effect=create_cancellable_engine,
                                ) as cancellable_engine_mock:
                                    with patch.object(
                                        run,
                                        "make_tts_engines_from_cores",
                                        return_value=tts_engines,
                                    ):
                                        with patch.object(
                                            run,
                                            "make_song_engines_from_cores",
                                            return_value=song_engines,
                                        ):
                                            with patch.object(
                                                run,
                                                "SettingHandler",
                                                return_value=MagicMock(
                                                    load=MagicMock(
                                                        return_value=settings
                                                    )
                                                ),
                                            ):
                                                with patch.object(
                                                    run,
                                                    "PresetManager",
                                                    return_value=MagicMock(),
                                                ):
                                                    with patch.object(
                                                        run,
                                                        "UserDictionary",
                                                        return_value=MagicMock(),
                                                    ):
                                                        with patch.object(
                                                            run,
                                                            "load_manifest",
                                                            return_value=engine_manifest,
                                                        ):
                                                            with patch.object(
                                                                run,
                                                                "LibraryManager",
                                                                return_value=MagicMock(),
                                                            ):
                                                                with patch.object(
                                                                    run,
                                                                    "engine_manifest_path",
                                                                    return_value=tmp_path
                                                                    / "manifest.json",
                                                                ):
                                                                    with patch.object(
                                                                        run,
                                                                        "get_save_dir",
                                                                        return_value=tmp_path,
                                                                    ):
                                                                        with patch.object(
                                                                            run,
                                                                            "engine_root",
                                                                            return_value=tmp_path,
                                                                        ):
                                                                            with patch.object(
                                                                                run,
                                                                                "generate_app",
                                                                                return_value=MagicMock(),
                                                                            ):
                                                                                with (
                                                                                    patch(
                                                                                        "run.uvicorn.run"
                                                                                    ) as uvicorn_run
                                                                                ):
                                                                                    run.main()

    multiprocessing.freeze_support.assert_called_once_with()
    create_plan_mock.assert_called_once_with(None)
    apply_plan_mock.assert_called_once_with(cpu_execution_plan)
    validate_plan_mock.assert_called_once_with(cpu_execution_plan)
    initialize_mock.assert_called_once()
    cancellable_engine_mock.assert_called_once()
    uvicorn_run.assert_called_once()
    assert events == ["create", "apply", "initialize", "validate", "cancellable"]
