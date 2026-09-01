"""VOICEVOX ENGINE の CPU affinity を検証する。"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from subprocess import Popen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import psutil

_MODES = ("default", "thread-count", "affinity")
_LAUNCH_FORMATS = ("python", "pyinstaller")
_VERIFICATION_FAILURE_EXIT_CODE = 1
_UNSUPPORTED_EXIT_CODE = 70
_INCONCLUSIVE_EXIT_CODE = 71
_REQUEST_TIMEOUT_SEC = 120.0
_TEXT = "こんにちは、音声合成の CPU affinity 検証です"
_CPU_AFFINITY_LOG_MARKER = re.compile(
    r'"event"\s*:\s*"cpu_affinity"|event=cpu_affinity'
)


def main() -> int:
    """CPU affinity の検証を実行する。"""
    parser = _create_parser()
    args = parser.parse_args()
    engine_command = list(args.engine_command)
    if engine_command and engine_command[0] == "--":
        engine_command = engine_command[1:]
    if not engine_command:
        parser.error("Engine command を指定してください。例: -- python run.py")

    output_path: Path = args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path: Path = args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = _run_trial(
            mode=args.mode,
            launch_format=args.launch_format,
            duration_sec=args.duration_sec,
            warmup_count=args.warmup_count,
            startup_timeout_sec=args.startup_timeout_sec,
            log_path=log_path,
            engine_command=engine_command,
            engine_cwd=_engine_cwd(args.launch_format, args.engine_cwd),
        )
    except _VerificationError as error:
        result = _failure_result(
            base=_initial_result(args.mode, args.launch_format),
            details=error.details,
            status="failed",
            reason=error.reason,
        )
        _write_result(output_path, result)
        return _VERIFICATION_FAILURE_EXIT_CODE
    except _UnsupportedError as error:
        result = _failure_result(
            base=_initial_result(args.mode, args.launch_format),
            details=error.details,
            status="unsupported",
            reason=error.reason,
        )
        _write_result(output_path, result)
        return _UNSUPPORTED_EXIT_CODE
    except _InconclusiveError as error:
        result = _failure_result(
            base=_initial_result(args.mode, args.launch_format),
            details=error.details,
            status="inconclusive",
            reason=error.reason,
        )
        _write_result(output_path, result)
        return _INCONCLUSIVE_EXIT_CODE

    _write_result(output_path, result)
    if result["status"] == "verified":
        return 0
    return _INCONCLUSIVE_EXIT_CODE


class _UnsupportedError(Exception):
    def __init__(self, reason: str, details: dict[str, object]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


class _VerificationError(Exception):
    def __init__(self, reason: str, details: dict[str, object]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


class _InconclusiveError(Exception):
    def __init__(self, reason: str, details: dict[str, object]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


class _LinuxCpuSampler:
    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cpu-sampler")
        self._samples: list[dict[str, object]] = []
        self._previous_stats: dict[str, tuple[int, int]] = {}
        self._error: BaseException | None = None

    def start(self) -> None:
        self._previous_stats = _linux_thread_stats(self._pid)
        self._thread.start()

    def stop(self) -> list[dict[str, object]]:
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError(
                "Linux の CPU サンプル取得スレッドを停止できませんでした。"
            )
        if self._error is not None:
            raise RuntimeError(
                "Linux の CPU サンプル取得に失敗しました。"
            ) from self._error
        return list(self._samples)

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(0.01):
                self._record()
        except BaseException as error:
            self._error = error

    def _record(self) -> None:
        current_stats = _linux_thread_stats(self._pid)
        active_thread_cpu: dict[str, int] = {}
        for tid, (ticks, processor) in current_stats.items():
            previous = self._previous_stats.get(tid)
            if previous is None:
                continue
            previous_ticks = previous[0]
            if ticks < previous_ticks:
                raise RuntimeError(f"TID {tid} の CPU 時間 tick が逆行しました。")
            if ticks > previous_ticks:
                active_thread_cpu[tid] = processor
        self._previous_stats = current_stats
        if len(active_thread_cpu) == 0:
            return
        self._samples.append(
            {
                "monotonic_sec": time.monotonic(),
                "thread_cpu": active_thread_cpu,
            }
        )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VOICEVOX ENGINE の CPU affinity を検証します。"
    )
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--startup-timeout-sec", type=float, default=180.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--engine-cwd", type=Path)
    parser.add_argument("--launch-format", choices=_LAUNCH_FORMATS, default="python")
    parser.add_argument("engine_command", nargs=argparse.REMAINDER)
    return parser


def _engine_cwd(launch_format: str, engine_cwd: Path | None) -> Path:
    if launch_format not in _LAUNCH_FORMATS:
        raise ValueError(f"想定外の launch-format です: {launch_format}")
    if engine_cwd is not None:
        return engine_cwd
    repository_root = Path(__file__).resolve().parents[1]
    if launch_format == "python":
        return repository_root
    return repository_root / "dist" / "run"


def _run_trial(
    mode: str,
    launch_format: str,
    duration_sec: float,
    warmup_count: int,
    startup_timeout_sec: float,
    log_path: Path,
    engine_command: list[str],
    engine_cwd: Path,
) -> dict[str, object]:
    if mode not in _MODES:
        raise ValueError(f"想定外の mode です: {mode}")
    if duration_sec <= 0:
        raise ValueError("duration-sec は正数で指定してください。")
    if warmup_count < 0:
        raise ValueError("warmup-count は0以上で指定してください。")
    if startup_timeout_sec <= 0:
        raise ValueError("startup-timeout-sec は正数で指定してください。")
    if launch_format not in _LAUNCH_FORMATS:
        raise ValueError(f"想定外の launch-format です: {launch_format}")
    result = _initial_result(mode, launch_format)
    result["engine_cwd"] = str(engine_cwd)
    system = platform.system()
    observer_cpu_set, observer_thread_masks = _read_observer_state()
    _store_observer_state(
        result,
        "before_launch",
        observer_cpu_set,
        observer_thread_masks,
    )
    original_cpu_set = observer_cpu_set
    if original_cpu_set is None:
        raise RuntimeError("検証開始時の observer CPU 集合を取得できません。")
    _verify_observer_state(
        result,
        observer_cpu_set,
        observer_thread_masks,
        original_cpu_set,
        "起動前",
    )
    result["original_cpu_set"] = original_cpu_set
    if len(original_cpu_set) <= 1:
        raise _UnsupportedError(
            "使用可能な論理 CPU が一つしかないため検証できません。", result
        )

    excluded_cpu = max(original_cpu_set)
    requested_cpu_set = [cpu for cpu in original_cpu_set if cpu != excluded_cpu]
    result["excluded_cpu"] = excluded_cpu
    result["requested_cpu_set"] = requested_cpu_set if mode != "default" else None
    command_cpu_num_threads = len(requested_cpu_set) if mode == "thread-count" else None
    result["command_cpu_num_threads"] = command_cpu_num_threads

    port = _find_free_port()
    result["port"] = port
    command = _build_engine_command(engine_command, port, command_cpu_num_threads)
    child_env = os.environ.copy()
    child_env.pop("VV_USE_GPU", None)
    child_env.pop("VV_CPU_NUM_THREADS", None)
    if mode == "default" or mode == "thread-count":
        child_env["VV_CPU_AFFINITY_MODE"] = "disabled"
    else:
        child_env.pop("VV_CPU_AFFINITY_MODE", None)

    process: subprocess.Popen[bytes] | None = None
    engine_pid: int | None = None
    sampler: _LinuxCpuSampler | None = None
    try:
        with open(log_path, "w", encoding="utf-8") as log_stream:
            process = Popen(
                command,
                cwd=engine_cwd,
                env=child_env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            result["launcher_pid"] = process.pid
            _capture_observer_state(
                result,
                "after_popen",
                original_cpu_set,
            )
            result["engine_version"] = _wait_for_version(
                process, port, startup_timeout_sec
            )
            engine_pid = _find_engine_process_id(process.pid, engine_command)
            result["engine_pid"] = engine_pid
            result["cpu_affinity_event"] = _wait_for_cpu_affinity_event(
                log_path,
                process,
                startup_timeout_sec,
            )
            engine_cpu_num_threads = _verify_cpu_affinity_event(
                result,
                result["cpu_affinity_event"],
                mode,
                system,
                original_cpu_set,
                requested_cpu_set,
                command_cpu_num_threads,
            )
            result["engine_cpu_num_threads"] = engine_cpu_num_threads
            result["cpu_num_threads"] = engine_cpu_num_threads
            thread_masks_after_core_initialization = _read_thread_masks(engine_pid)
            result["thread_masks_after_core_initialization"] = (
                thread_masks_after_core_initialization
            )
            result["configured_cpu_set"] = _configured_cpu_set(
                engine_pid, thread_masks_after_core_initialization
            )
            _capture_observer_state(
                result,
                "after_core_initialization",
                original_cpu_set,
            )

            query = _audio_query(port)
            for _ in range(warmup_count):
                _synthesis(port, query)
            result["thread_masks_before_measurement"] = _read_thread_masks(engine_pid)

            if system != "Darwin":
                thread_cpu_before = _thread_cpu_times(engine_pid)
            process_cpu_before = _process_cpu_times(engine_pid)
            logical_cpu_before = _logical_cpu_times()
            inference_durations: list[float] = []
            inference_started = time.monotonic()
            sampler = _start_linux_sampler(engine_pid)
            while (
                time.monotonic() - inference_started < duration_sec
                or len(inference_durations) == 0
            ):
                request_started = time.monotonic()
                _synthesis(port, query)
                inference_durations.append(time.monotonic() - request_started)
            linux_samples = _stop_linux_sampler(sampler)
            sampler = None
            result["thread_masks_after_measurement"] = _read_thread_masks(engine_pid)
            _capture_observer_state(
                result,
                "after_measurement",
                original_cpu_set,
            )
            if system != "Darwin":
                thread_cpu_after = _thread_cpu_times(engine_pid)
            process_cpu_after = _process_cpu_times(engine_pid)
            logical_cpu_after = _logical_cpu_times()

            if system == "Darwin":
                result["thread_cpu_time"] = _unsupported_thread_cpu_time()
            else:
                result["thread_cpu_time"] = _thread_cpu_time_result(
                    thread_cpu_before,
                    thread_cpu_after,
                )
            result["process_cpu_time"] = _process_cpu_time_result(
                process_cpu_before, process_cpu_after
            )
            result["logical_cpu_time_delta"] = _logical_cpu_time_delta(
                logical_cpu_before, logical_cpu_after
            )
            result["linux_thread_execution_cpu_samples"] = linux_samples
            result["inference_count"] = len(inference_durations)
            result["inference_time_sec"] = {
                "total": sum(inference_durations),
                "minimum": min(inference_durations),
                "maximum": max(inference_durations),
                "per_request": inference_durations,
            }
            if mode == "affinity" and system == "Linux":
                _verify_linux_affinity(result, requested_cpu_set)
            if mode == "affinity" and system == "Windows":
                _verify_windows_affinity(result, requested_cpu_set)
            if mode == "affinity" and system == "Darwin":
                raise _UnsupportedError(
                    "macOS の CPU affinity は診断上 unsupported です。合成は成功しました。",
                    result,
                )
            result["status"] = "verified"
            result["verified"] = True
            result["unsupported"] = False
            result["inconclusive"] = False
            result["reason"] = _verified_reason(mode)
    finally:
        if sampler is not None:
            _stop_linux_sampler(sampler)
        if engine_pid is not None and process is not None and engine_pid != process.pid:
            _terminate_process(engine_pid)
        if process is not None:
            _terminate_process(process.pid)
            process.wait()
            _capture_observer_state(
                result,
                "after_engine_exit",
                original_cpu_set,
            )
    return result


def _initial_result(mode: str, launch_format: str) -> dict[str, object]:
    if mode not in _MODES:
        raise ValueError(f"想定外の mode です: {mode}")
    if launch_format not in _LAUNCH_FORMATS:
        raise ValueError(f"想定外の launch-format です: {launch_format}")
    return {
        "environment": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "github_actions": _github_actions_environment(),
        },
        "mode": mode,
        "launch_format": launch_format,
        "engine_cwd": None,
        "engine_version": None,
        "status": "inconclusive",
        "verified": False,
        "unsupported": False,
        "inconclusive": True,
        "original_cpu_set": None,
        "requested_cpu_set": None,
        "configured_cpu_set": None,
        "excluded_cpu": None,
        "command_cpu_num_threads": None,
        "engine_cpu_num_threads": None,
        "cpu_num_threads": None,
        "thread_masks_after_core_initialization": {},
        "thread_masks_before_measurement": {},
        "thread_masks_after_measurement": {},
        "thread_cpu_time": (
            _unsupported_thread_cpu_time() if platform.system() == "Darwin" else {}
        ),
        "process_cpu_time": {},
        "logical_cpu_time_delta": {},
        "linux_thread_execution_cpu_samples": [],
        "inference_count": 0,
        "inference_time_sec": {},
        "cpu_affinity_event": None,
        "observer_cpu_sets": {},
        "observer_thread_masks": {},
    }


def _github_actions_environment() -> dict[str, str]:
    variables = ("RUNNER_OS", "RUNNER_ARCH", "RUNNER_NAME", "ImageOS")
    environment: dict[str, str] = {}
    for variable in variables:
        value = os.environ.get(variable)
        if value is not None:
            environment[variable] = value
    return environment


def _failure_result(
    base: dict[str, object],
    details: dict[str, object],
    status: str,
    reason: str,
) -> dict[str, object]:
    result = {**base, **details}
    result["status"] = status
    result["verified"] = False
    result["unsupported"] = status == "unsupported"
    result["inconclusive"] = status == "inconclusive"
    result["reason"] = reason
    return result


def _write_result(output_path: Path, result: dict[str, object]) -> None:
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_observer_state() -> tuple[list[int] | None, dict[str, list[int]] | None]:
    system = platform.system()
    if system == "Linux":
        return (
            sorted(psutil.Process(os.getpid()).cpu_affinity()),
            _linux_thread_masks(os.getpid()),
        )
    if system == "Windows":
        return _windows_process_cpu_set(os.getpid()), None
    if system == "Darwin":
        cpu_count = os.cpu_count()
        if cpu_count is None:
            raise RuntimeError("macOS の論理 CPU 数を取得できませんでした。")
        return list(range(cpu_count)), None
    raise RuntimeError(f"対応していない OS です: {system}")


def _store_observer_state(
    result: dict[str, object],
    phase: str,
    cpu_set: list[int] | None,
    thread_masks: dict[str, list[int]] | None,
) -> None:
    cpu_sets = result["observer_cpu_sets"]
    if not isinstance(cpu_sets, dict):
        raise RuntimeError("observer CPU 集合の保存先が不正です。")
    cpu_sets[phase] = cpu_set
    observer_thread_masks = result["observer_thread_masks"]
    if not isinstance(observer_thread_masks, dict):
        raise RuntimeError("observer thread mask の保存先が不正です。")
    observer_thread_masks[phase] = thread_masks


def _capture_observer_state(
    result: dict[str, object], phase: str, expected_cpu_set: list[int]
) -> None:
    cpu_set, thread_masks = _read_observer_state()
    _store_observer_state(result, phase, cpu_set, thread_masks)
    _verify_observer_state(
        result,
        cpu_set,
        thread_masks,
        expected_cpu_set,
        phase,
    )


def _verify_observer_state(
    result: dict[str, object],
    cpu_set: list[int] | None,
    thread_masks: dict[str, list[int]] | None,
    expected_cpu_set: list[int],
    phase: str,
) -> None:
    if cpu_set != expected_cpu_set:
        raise _VerificationError(
            f"observer の CPU 集合が{phase}に起動前と一致しません。",
            result,
        )
    if platform.system() != "Linux":
        return
    if thread_masks is None or len(thread_masks) == 0:
        raise _VerificationError(
            f"observer の全 TID mask を{phase}に取得できません。", result
        )
    for mask in thread_masks.values():
        if mask != expected_cpu_set:
            raise _VerificationError(
                f"observer の TID mask が{phase}に起動前と一致しません。",
                result,
            )


def _build_engine_command(
    engine_command: list[str], port: int, cpu_num_threads: int | None
) -> list[str]:
    managed_options = {
        "--host",
        "--port",
        "--load_all_models",
        "--cpu_num_threads",
    }
    for argument in engine_command:
        option = argument.split("=", 1)[0]
        if option in managed_options:
            raise ValueError(f"Engine command に {option} を指定しないでください。")
    command = list(engine_command)
    command.extend(
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--load_all_models",
        ]
    )
    if cpu_num_threads is not None:
        command.extend(["--cpu_num_threads", str(cpu_num_threads)])
    return command


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_version(
    process: subprocess.Popen[bytes], port: int, timeout_sec: float
) -> str:
    url = f"http://127.0.0.1:{port}/version"
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Engine が起動前に終了しました。終了コード: {process.returncode}"
            )
        try:
            response = _http_request(url=url, method="GET", body=None)
            if response:
                try:
                    version = json.loads(response)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        "Engine の version 応答を JSON として解釈できません。"
                    ) from error
                if not isinstance(version, str) or not version:
                    raise RuntimeError("Engine の version 応答が文字列ではありません。")
                return version
            last_error = RuntimeError("Engine の version 応答が空でした。")
        except (HTTPError, OSError, URLError) as error:
            last_error = error
        time.sleep(0.5)
    if last_error is not None:
        raise RuntimeError(
            "Engine の起動を待機中にタイムアウトしました。"
        ) from last_error
    raise RuntimeError("Engine の起動を待機中にタイムアウトしました。")


def _wait_for_cpu_affinity_event(
    log_path: Path,
    process: subprocess.Popen[bytes],
    timeout_sec: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        events = _read_cpu_affinity_events(log_path)
        if len(events) != 0:
            return events[-1]
        if process.poll() is not None:
            raise RuntimeError(
                "Engine のログに cpu_affinity 診断イベントがありません。"
            )
        time.sleep(0.1)
    raise RuntimeError(
        "Engine の cpu_affinity 診断イベントを待機中にタイムアウトしました。"
    )


def _read_cpu_affinity_events(log_path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if _CPU_AFFINITY_LOG_MARKER.search(line) is None:
            continue
        json_start = line.find("{")
        if json_start < 0:
            raise RuntimeError(
                "cpu_affinity 診断 marker を含むログ行に JSON がありません。"
            )
        payload = json.loads(line[json_start:])
        if not isinstance(payload, dict):
            raise RuntimeError(
                "cpu_affinity 診断イベントが JSON オブジェクトではありません。"
            )
        if payload.get("event") != "cpu_affinity":
            raise RuntimeError("cpu_affinity 診断イベントの marker が不正です。")
        events.append(payload)
    return events


def _verify_cpu_affinity_event(
    result: dict[str, object],
    event: object,
    mode: str,
    system: str,
    original_cpu_set: list[int],
    requested_cpu_set: list[int],
    command_cpu_num_threads: int | None,
) -> int | None:
    if not isinstance(event, dict):
        raise _VerificationError(
            "Engine の cpu_affinity 診断イベントが JSON オブジェクトではありません。",
            result,
        )
    if event.get("event") != "cpu_affinity":
        raise _VerificationError(
            "Engine の cpu_affinity 診断イベントの event が不正です。", result
        )
    if mode == "default" or mode == "thread-count":
        expected_mode = "disabled"
        expected_state = "disabled"
        expected_original_cpu_set: list[int] | None = None
        expected_requested_cpu_set: list[int] | None = None
        expected_configured_cpu_set: list[int] | None = None
        expected_excluded_cpu: int | None = None
        expected_cpu_num_threads = command_cpu_num_threads
    elif mode == "affinity":
        expected_mode = "auto"
        if system == "Darwin":
            expected_state = "unsupported"
            expected_original_cpu_set = None
            expected_requested_cpu_set = None
            expected_configured_cpu_set = None
            expected_excluded_cpu = None
            expected_cpu_num_threads = None
        else:
            expected_state = "applied"
            expected_original_cpu_set = original_cpu_set
            expected_requested_cpu_set = requested_cpu_set
            expected_configured_cpu_set = requested_cpu_set
            expected_excluded_cpu = max(original_cpu_set)
            expected_cpu_num_threads = len(requested_cpu_set)
    else:
        raise ValueError(f"想定外の mode です: {mode}")

    expected_values: dict[str, object] = {
        "mode": expected_mode,
        "state": expected_state,
        "original_cpu_set": expected_original_cpu_set,
        "requested_cpu_set": expected_requested_cpu_set,
        "configured_cpu_set": expected_configured_cpu_set,
        "excluded_cpu": expected_excluded_cpu,
        "cpu_num_threads": expected_cpu_num_threads,
    }
    for field, expected in expected_values.items():
        if field not in event:
            raise _VerificationError(
                f"Engine の cpu_affinity 診断イベントに {field} がありません。", result
            )
        actual = event[field]
        if field.endswith("cpu_set"):
            actual = _validated_cpu_set(actual, field, result)
        elif field == "excluded_cpu":
            if actual is not None and (
                not isinstance(actual, int) or isinstance(actual, bool)
            ):
                raise _VerificationError(
                    f"Engine の cpu_affinity 診断イベントの {field} が不正です。",
                    result,
                )
        elif field == "cpu_num_threads":
            actual = _validated_cpu_num_threads(actual, result)
        if actual != expected:
            raise _VerificationError(
                f"Engine の cpu_affinity 診断イベントの {field} が想定値と一致しません。",
                result,
            )
    reason = event.get("reason")
    if not isinstance(reason, str) or len(reason) == 0:
        raise _VerificationError(
            "Engine の cpu_affinity 診断イベントの reason が不正です。", result
        )
    return _validated_cpu_num_threads(event["cpu_num_threads"], result)


def _validated_cpu_set(
    value: object, field: str, result: dict[str, object]
) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(
        not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0 for cpu in value
    ):
        raise _VerificationError(
            f"Engine の cpu_affinity 診断イベントの {field} が不正です。", result
        )
    if value != sorted(set(value)):
        raise _VerificationError(
            f"Engine の cpu_affinity 診断イベントの {field} が整列されていません。",
            result,
        )
    return value


def _validated_cpu_num_threads(value: object, result: dict[str, object]) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise _VerificationError(
            "Engine の cpu_affinity 診断イベントの cpu_num_threads が不正です。",
            result,
        )
    return value


def _http_request(url: str, method: str, body: bytes | None) -> bytes:
    request = Request(url, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=_REQUEST_TIMEOUT_SEC) as response:
        body = response.read()
    if not isinstance(body, bytes):
        raise RuntimeError("HTTP 応答の body が bytes ではありません。")
    return body


def _audio_query(port: int) -> dict[str, object]:
    query = _http_request(
        url=(
            f"http://127.0.0.1:{port}/audio_query?"
            + urlencode({"speaker": "1", "text": _TEXT})
        ),
        method="POST",
        body=None,
    )
    parsed = json.loads(query)
    if not isinstance(parsed, dict):
        raise RuntimeError("audio_query の応答が JSON オブジェクトではありません。")
    return parsed


def _synthesis(port: int, query: dict[str, object]) -> None:
    wave = _http_request(
        url=f"http://127.0.0.1:{port}/synthesis?speaker=1",
        method="POST",
        body=json.dumps(query, ensure_ascii=False).encode("utf-8"),
    )
    if len(wave) < 44 or wave[:4] != b"RIFF":
        raise RuntimeError("synthesis の応答が WAV ではありません。")


def _find_engine_process_id(root_pid: int, command: list[str]) -> int:
    expected_names = {
        Path(part).name
        for part in command
        if part.endswith((".py", ".exe")) or Path(part).name in {"run", "run.exe"}
    }
    if not expected_names:
        expected_names.add(Path(command[0]).name)
    queue: list[tuple[psutil.Process, int]] = [(psutil.Process(root_pid), 0)]
    candidates: list[tuple[int, int]] = []
    visited: set[int] = set()
    while queue:
        process, depth = queue.pop(0)
        if process.pid in visited:
            continue
        visited.add(process.pid)
        try:
            command_line = process.cmdline()
        except psutil.NoSuchProcess:
            continue
        if not expected_names or any(
            Path(part).name in expected_names for part in command_line
        ):
            candidates.append((process.pid, depth))
        try:
            children = process.children()
        except psutil.NoSuchProcess:
            children = []
        queue.extend((child, depth + 1) for child in children)
    if candidates:
        return max(candidates, key=lambda candidate: candidate[1])[0]
    raise RuntimeError("Engine プロセスを特定できませんでした。")


def _read_thread_masks(pid: int) -> dict[str, list[int]] | dict[str, object]:
    system = platform.system()
    if system == "Linux":
        return _linux_thread_masks(pid)
    if system == "Windows":
        return {"process": _windows_process_cpu_set(pid)}
    return {}


def _configured_cpu_set(
    pid: int, thread_masks: dict[str, list[int]] | dict[str, object]
) -> list[int] | None:
    if platform.system() == "Linux":
        main_mask = thread_masks.get(str(pid))
        if not isinstance(main_mask, list):
            raise RuntimeError(
                "Engine のメインスレッド affinity を取得できませんでした。"
            )
        return main_mask
    if platform.system() == "Windows":
        process_mask = thread_masks.get("process")
        if not isinstance(process_mask, list):
            raise RuntimeError("Engine の process affinity を取得できません。")
        return process_mask
    return None


def _linux_thread_masks(pid: int) -> dict[str, list[int]]:
    task_root = Path(f"/proc/{pid}/task")
    masks: dict[str, list[int]] = {}
    for task_dir in task_root.iterdir():
        status = (task_dir / "status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                masks[task_dir.name] = _parse_cpu_list(line.split(":", 1)[1].strip())
                break
        else:
            raise RuntimeError(
                f"TID {task_dir.name} の CPU affinity が見つかりません。"
            )
    return masks


def _parse_cpu_list(value: str) -> list[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise RuntimeError(f"CPU 集合の範囲が不正です: {value}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    if not cpus:
        raise RuntimeError(f"CPU 集合が空です: {value}")
    return sorted(cpus)


def _linux_thread_stats(pid: int) -> dict[str, tuple[int, int]]:
    task_root = Path(f"/proc/{pid}/task")
    thread_stats: dict[str, tuple[int, int]] = {}
    for task_dir in task_root.iterdir():
        stat = (task_dir / "stat").read_text(encoding="utf-8")
        closing_parenthesis = stat.rfind(")")
        if closing_parenthesis < 0:
            raise RuntimeError(f"TID {task_dir.name} の stat が不正です。")
        fields = stat[closing_parenthesis + 2 :].split()
        if len(fields) <= 36:
            raise RuntimeError(
                f"TID {task_dir.name} の stat フィールドが不足しています。"
            )
        try:
            ticks = int(fields[11]) + int(fields[12])
            processor = int(fields[36])
        except ValueError as error:
            raise RuntimeError(f"TID {task_dir.name} の stat が不正です。") from error
        if ticks < 0:
            raise RuntimeError(f"TID {task_dir.name} の CPU 時間 tick が負数です。")
        thread_stats[task_dir.name] = (ticks, processor)
    return thread_stats


def _thread_cpu_times(pid: int) -> dict[str, dict[str, float]]:
    process = psutil.Process(pid)
    return {
        str(thread.id): {
            "user_sec": float(thread.user_time),
            "system_sec": float(thread.system_time),
        }
        for thread in process.threads()
    }


def _thread_cpu_time_result(
    before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]
) -> dict[str, object]:
    common_thread_ids = sorted(before.keys() & after.keys())
    new_thread_ids = sorted(after.keys() - before.keys())
    terminated_thread_ids = sorted(before.keys() - after.keys())
    delta: dict[str, dict[str, float]] = {}
    for tid in common_thread_ids:
        before_time = before[tid]
        after_time = after[tid]
        delta[tid] = {
            "user_sec": after_time["user_sec"] - before_time["user_sec"],
            "system_sec": after_time["system_sec"] - before_time["system_sec"],
        }
    return {
        "status": "supported",
        "before": before,
        "after": after,
        "delta": delta,
        "common_thread_ids": common_thread_ids,
        "new_thread_ids": new_thread_ids,
        "terminated_thread_ids": terminated_thread_ids,
        "unavailable_thread_ids": terminated_thread_ids,
    }


def _unsupported_thread_cpu_time() -> dict[str, str]:
    return {
        "status": "unsupported",
        "reason": "macOS では別プロセスの thread CPU 時間を取得できません。",
    }


def _process_cpu_times(pid: int) -> dict[str, float]:
    process = psutil.Process(pid)
    times = process.cpu_times()
    return {
        "user_sec": float(times.user),
        "system_sec": float(times.system),
    }


def _process_cpu_time_result(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, dict[str, float]]:
    delta: dict[str, float] = {}
    for field in ("user_sec", "system_sec"):
        value = after[field] - before[field]
        if value < 0:
            raise RuntimeError(f"Engine process の {field} counter が逆行しました。")
        delta[field] = value
    return {"before": before, "after": after, "delta": delta}


def _logical_cpu_times() -> dict[str, dict[str, float]]:
    cpu_times: dict[str, dict[str, float]] = {}
    for cpu, times in enumerate(psutil.cpu_times(percpu=True)):
        total_sec = float(sum(times))
        total_sec -= float(getattr(times, "guest", 0.0))
        total_sec -= float(getattr(times, "guest_nice", 0.0))
        idle_sec = float(times.idle)
        iowait_sec = float(getattr(times, "iowait", 0.0))
        cpu_times[str(cpu)] = {
            "user_sec": float(times.user),
            "system_sec": float(times.system),
            "idle_sec": idle_sec,
            "total_sec": total_sec,
            "busy_sec": total_sec - idle_sec - iowait_sec,
        }
    if not cpu_times:
        raise RuntimeError("論理 CPU の時間を取得できませんでした。")
    return cpu_times


def _logical_cpu_time_delta(
    before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    if not before:
        raise RuntimeError("論理 CPU の計測結果が空です。")
    if before.keys() != after.keys():
        raise RuntimeError("論理 CPU の計測前後で CPU 集合が変化しました。")
    fields = ("user_sec", "system_sec", "idle_sec", "total_sec", "busy_sec")
    delta: dict[str, dict[str, float]] = {}
    for cpu in sorted(before.keys() & after.keys()):
        cpu_delta: dict[str, float] = {}
        for field in fields:
            value = after[cpu][field] - before[cpu][field]
            if value < 0:
                raise RuntimeError(
                    f"論理 CPU {cpu} の {field} counter が逆行しました。"
                )
            cpu_delta[field] = value
        total_sec = cpu_delta["total_sec"]
        if total_sec <= 0:
            raise RuntimeError(f"論理 CPU {cpu} の total_sec が正数ではありません。")
        cpu_delta["busy_ratio"] = cpu_delta["busy_sec"] / total_sec
        delta[cpu] = cpu_delta
    return delta


def _start_linux_sampler(pid: int) -> _LinuxCpuSampler | None:
    if platform.system() != "Linux":
        return None
    sampler = _LinuxCpuSampler(pid)
    sampler.start()
    return sampler


def _stop_linux_sampler(sampler: _LinuxCpuSampler | None) -> list[dict[str, object]]:
    if sampler is None:
        return []
    return sampler.stop()


def _verify_linux_affinity(
    result: dict[str, object], requested_cpu_set: list[int]
) -> None:
    requested_cpus = set(requested_cpu_set)
    mask_fields = (
        "thread_masks_after_core_initialization",
        "thread_masks_before_measurement",
        "thread_masks_after_measurement",
    )
    for mask_field in mask_fields:
        masks = result[mask_field]
        if not isinstance(masks, dict) or len(masks) == 0:
            raise _InconclusiveError(
                f"Linux の {mask_field} を取得できませんでした。", result
            )
        for mask in masks.values():
            if not isinstance(mask, list):
                raise _InconclusiveError(
                    f"Linux の {mask_field} の形式が不正です。", result
                )
            if set(mask) != requested_cpus:
                raise _VerificationError(
                    f"Linux の {mask_field} が要求集合と一致しません。", result
                )
    samples = result["linux_thread_execution_cpu_samples"]
    if not isinstance(samples, list) or len(samples) == 0:
        raise _InconclusiveError(
            "Linux の thread 実行 CPU サンプルがありません。", result
        )
    for sample in samples:
        if not isinstance(sample, dict):
            raise _InconclusiveError(
                "Linux の thread 実行 CPU サンプルの形式が不正です。", result
            )
        thread_cpu = sample.get("thread_cpu")
        if not isinstance(thread_cpu, dict):
            raise _InconclusiveError(
                "Linux の thread 実行 CPU サンプルの形式が不正です。", result
            )
        if len(thread_cpu) == 0:
            raise _InconclusiveError(
                "Linux の thread 実行 CPU サンプルが空です。", result
            )
        for cpu in thread_cpu.values():
            if (
                not isinstance(cpu, int)
                or isinstance(cpu, bool)
                or cpu not in requested_cpus
            ):
                raise _VerificationError(
                    "Linux の thread 実行 CPU サンプルに要求外 CPU が現れました。",
                    result,
                )


def _verify_windows_affinity(
    result: dict[str, object], requested_cpu_set: list[int]
) -> None:
    configured_cpu_set = result["configured_cpu_set"]
    if not isinstance(configured_cpu_set, list):
        raise _InconclusiveError(
            "子 Engine の process affinity を読み取れませんでした。", result
        )
    if configured_cpu_set != requested_cpu_set:
        raise _VerificationError(
            "子 Engine の process affinity が要求集合と一致しません。", result
        )
    mask_fields = (
        "thread_masks_after_core_initialization",
        "thread_masks_before_measurement",
        "thread_masks_after_measurement",
    )
    for mask_field in mask_fields:
        masks = result[mask_field]
        if not isinstance(masks, dict):
            raise _InconclusiveError(
                f"Windows の {mask_field} を取得できませんでした。", result
            )
        process_mask = masks.get("process")
        if not isinstance(process_mask, list):
            raise _InconclusiveError(
                f"Windows の {mask_field} の形式が不正です。", result
            )
        if process_mask != requested_cpu_set:
            raise _VerificationError(
                f"Windows の {mask_field} が要求集合と一致しません。", result
            )


def _verified_reason(mode: str) -> str:
    if mode == "default":
        return "VV_CPU_AFFINITY_MODE=disabled を渡して CPU affinity を比較用に無効化し、cpu_num_threads を指定せず合成と計測が完了しました。"
    if mode == "thread-count":
        return "VV_CPU_AFFINITY_MODE=disabled を渡して CPU affinity を比較用に無効化し、正数 cpu_num_threads を指定して合成と計測が完了しました。"
    if mode == "affinity":
        return "Engine 自身が設定した CPU affinity が要求集合と一致し、CPU 時間 tick が増加した TID の実行 CPU サンプルにも除外 CPU が現れませんでした。"
    raise ValueError(f"想定外の mode です: {mode}")


def _terminate_process(pid: int) -> None:
    try:
        process = psutil.Process(pid)
        process.terminate()
    except psutil.NoSuchProcess:
        return
    try:
        process.wait(timeout=10.0)
    except psutil.TimeoutExpired:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            return
        process.wait(timeout=10.0)


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
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    return kernel32


def _windows_error(kernel32: ctypes.CDLL) -> OSError:
    error_code = int(kernel32.GetLastError())
    return OSError(error_code, f"Windows API エラー: {error_code}")


def _windows_process_cpu_set(pid: int) -> list[int]:
    kernel32 = _windows_kernel32()
    processor_group_count = int(kernel32.GetActiveProcessorGroupCount())
    if processor_group_count > 1:
        raise _UnsupportedError(
            "複数の processor group を持つ Windows 環境は未対応です。", {}
        )
    if processor_group_count != 1:
        raise RuntimeError("Windows の processor group 数を取得できません。")
    current_pid = os.getpid()
    if pid == current_pid:
        handle = kernel32.GetCurrentProcess()
        should_close = False
    else:
        handle = kernel32.OpenProcess(0x0400, 0, pid)
        if not handle:
            raise _windows_error(kernel32)
        should_close = True
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    try:
        if not kernel32.GetProcessAffinityMask(
            handle, ctypes.byref(process_mask), ctypes.byref(system_mask)
        ):
            raise _windows_error(kernel32)
        return [
            bit
            for bit in range(ctypes.sizeof(ctypes.c_size_t) * 8)
            if process_mask.value & (1 << bit)
        ]
    finally:
        if should_close and not kernel32.CloseHandle(handle):
            raise _windows_error(kernel32)


if __name__ == "__main__":
    sys.exit(main())
