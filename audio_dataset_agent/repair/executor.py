"""修复执行器：把 RepairPlan 转换成真实工具调用。"""
import subprocess
import shutil
from pathlib import Path
from typing import Callable, Protocol

from audio_dataset_agent.core.models import CandidateSample, RepairExecution, RepairPlan
from audio_dataset_agent.repair.validation import validate_repair_plan


CommandRunner = Callable[[list[str]], None]


# 优先查找项目本地工具目录，找不到时再退回系统 PATH 中的 deep-filter。
def default_deepfilter_path() -> str:
    local_binary = Path("local_tools") / "deepfilternet" / "deep-filter.exe"
    if local_binary.exists():
        return str(local_binary)
    return "deep-filter"


def run_subprocess(command: list[str]) -> None:
    """执行 FFmpeg 修复命令，失败时抛出异常给 service 记录。"""
    subprocess.run(command, check=True, capture_output=True, text=True)


class RepairExecutor(Protocol):
    # RepairExecutor 负责执行 RepairPlan，但不决定是否替换原样本。
    def execute(
        self,
        plan: RepairPlan,
        sample: CandidateSample,
        repairs_dir: Path,
    ) -> RepairExecution:
        """执行修复计划并返回执行记录。"""


class FFmpegRepairExecutor:
    """使用 FFmpeg 滤镜执行音频修复操作。"""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        runner: CommandRunner = run_subprocess,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.runner = runner

    def execute(
        self,
        plan: RepairPlan,
        sample: CandidateSample,
        repairs_dir: Path,
    ) -> RepairExecution:
        repairs_dir.mkdir(parents=True, exist_ok=True)
        # 优先修复物化后的样本切片；如果没有切片文件，再退回源工作音频。
        input_path = sample.sample_path or sample.source_path
        output_path = repairs_dir / f"{sample.sample_id}_{plan.operation}.wav"
        command = self.build_command(plan, input_path, output_path)

        try:
            self.runner(command)
        except Exception as exc:
            return RepairExecution(
                sample_id=plan.sample_id,
                operation=plan.operation,
                input_path=input_path,
                output_path=output_path,
                status="failed",
                error_message=str(exc),
            )

        return RepairExecution(
            sample_id=plan.sample_id,
            operation=plan.operation,
            input_path=input_path,
            output_path=output_path,
            status="succeeded",
        )

    def build_command(
        self,
        plan: RepairPlan,
        input_path: Path,
        output_path: Path,
    ) -> list[str]:
        # 即使计划来自外部调用方，也要在执行前再次校验参数边界。
        validate_repair_plan(plan)
        # 不同 operation 对应不同 FFmpeg filter，参数来自 RepairPlan。
        if plan.operation == "loudness_normalization":
            target_lufs = plan.parameters.get("target_lufs", -23.0)
            filter_spec = f"loudnorm=I={target_lufs}"
        elif plan.operation == "silence_trimming":
            threshold = plan.parameters.get("silence_threshold_dbfs", -50.0)
            duration_ms = plan.parameters.get("min_silence_duration_ms", 200)
            duration_seconds = float(duration_ms) / 1000
            filter_spec = (
                "silenceremove="
                f"start_periods=1:start_duration={duration_seconds}:"
                f"start_threshold={threshold}dB:"
                "stop_periods=1:"
                f"stop_duration={duration_seconds}:"
                f"stop_threshold={threshold}dB"
            )
        elif plan.operation == "noise_reduction":
            noise_reduction_db = plan.parameters.get("noise_reduction_db", 12.0)
            filter_spec = f"afftdn=nr={noise_reduction_db}"
        else:
            raise ValueError(f"Unsupported repair operation: {plan.operation}")

        return [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(input_path),
            "-af",
            filter_spec,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-vn",
            str(output_path),
        ]


class DeepFilterNetRepairExecutor:
    """使用 DeepFilterNet CLI 执行神经网络降噪修复。"""

    def __init__(
        self,
        deepfilter_path: str | None = None,
        runner: CommandRunner = run_subprocess,
    ):
        self.deepfilter_path = deepfilter_path or default_deepfilter_path()
        self.runner = runner

    def execute(
        self,
        plan: RepairPlan,
        sample: CandidateSample,
        repairs_dir: Path,
    ) -> RepairExecution:
        repairs_dir.mkdir(parents=True, exist_ok=True)
        input_path = sample.sample_path or sample.source_path
        output_path = repairs_dir / f"{sample.sample_id}_{plan.operation}.wav"
        generated_dir = repairs_dir / f"{sample.sample_id}_{plan.operation}_deepfilter"
        command = self.build_command(plan, input_path, generated_dir)

        try:
            self.runner(command)
            generated_path = self._find_generated_output(generated_dir, input_path)
            shutil.copyfile(generated_path, output_path)
        except Exception as exc:
            return RepairExecution(
                sample_id=plan.sample_id,
                operation=plan.operation,
                input_path=input_path,
                output_path=output_path,
                status="failed",
                error_message=str(exc),
            )

        return RepairExecution(
            sample_id=plan.sample_id,
            operation=plan.operation,
            input_path=input_path,
            output_path=output_path,
            status="succeeded",
        )

    def build_command(
        self,
        plan: RepairPlan,
        input_path: Path,
        output_dir: Path,
    ) -> list[str]:
        validate_repair_plan(plan)
        if plan.operation != "neural_noise_reduction":
            raise ValueError(f"Unsupported DeepFilterNet operation: {plan.operation}")

        command = [self.deepfilter_path]
        if self._uses_python_cli():
            command.extend(["--output-dir", str(output_dir)])
        else:
            command.extend(["-o", str(output_dir)])
        if bool(plan.parameters.get("post_filter", False)):
            command.append("--pf")
        command.append(str(input_path))
        return command

    def _uses_python_cli(self) -> bool:
        executable_name = Path(self.deepfilter_path).name.lower()
        return executable_name in {"deepfilter", "deepfilter.exe", "deep-filter-py.exe"}

    def _find_generated_output(self, output_dir: Path, input_path: Path) -> Path:
        direct_output = output_dir / input_path.name
        if direct_output.exists():
            return direct_output

        wav_outputs = sorted(output_dir.glob("*.wav"))
        if wav_outputs:
            return wav_outputs[0]
        raise FileNotFoundError(
            f"DeepFilterNet did not create a WAV output in {output_dir}"
        )


class CompositeRepairExecutor:
    """根据操作类型将修复计划路由到 FFmpeg 或 DeepFilterNet 执行器。"""

    def __init__(
        self,
        ffmpeg_executor: RepairExecutor | None = None,
        deepfilternet_executor: RepairExecutor | None = None,
    ):
        self.ffmpeg_executor = ffmpeg_executor or FFmpegRepairExecutor()
        self.deepfilternet_executor = (
            deepfilternet_executor or DeepFilterNetRepairExecutor()
        )

    def execute(
        self,
        plan: RepairPlan,
        sample: CandidateSample,
        repairs_dir: Path,
    ) -> RepairExecution:
        if plan.operation == "neural_noise_reduction":
            return self.deepfilternet_executor.execute(plan, sample, repairs_dir)
        return self.ffmpeg_executor.execute(plan, sample, repairs_dir)
