"""MSST/pymss 去混响适配器。

适配器只负责模型校验和调用边界，不把 MSST 依赖强行写入基础环境。
没有推理运行时且用户选择 error 时直接失败；选择 skip 时明确记录跳过。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class MSSTModelSpec:
    name: str
    checkpoint_path: Path
    config_path: Path
    model_type: str = "bs_roformer"
    version: str = "MSST-WebUI"

    def is_ready(self) -> bool:
        return self.checkpoint_path.is_file() and self.config_path.is_file()

    def validation_error(self) -> str:
        if not self.checkpoint_path.is_file():
            return f"模型权重不存在：{self.checkpoint_path}"
        if not self.config_path.is_file():
            return f"模型配置不存在：{self.config_path}"
        return ""


@dataclass(frozen=True)
class MSSTPreprocessResult:
    status: str
    input_path: Path
    output_path: Path
    model_name: str
    model_version: str
    parameters: dict[str, object]
    error: str | None = None


class MSSTPreprocessor:
    """调用 pymss/MSST 的最小边界；实际运行器通过依赖注入提供。"""

    def __init__(
        self,
        model: MSSTModelSpec,
        runner: Callable[[MSSTModelSpec, Path, Path, dict[str, object]], None] | None = None,
        runtime_available: bool | None = None,
        separator_factory: Callable[[MSSTModelSpec, dict[str, object]], object] | None = None,
    ):
        self.model = model
        self.runner = runner
        self.separator_factory = separator_factory
        if runtime_available is None:
            if runner is not None:
                self.runtime_available = True
            else:
                try:
                    import pymss  # noqa: F401
                    self.runtime_available = True
                except ImportError:
                    self.runtime_available = False
        else:
            self.runtime_available = runtime_available
        if self.runner is None and self.runtime_available:
            self.runner = self._pymss_runner

    def prepare_files(
        self,
        files: Iterable[tuple[Path, Path]],
        missing_policy: str = "error",
        parameters: dict[str, object] | None = None,
    ) -> list[MSSTPreprocessResult]:
        """一次加载模型，处理当前 Run 的全部音频文件。"""
        parameters = dict(parameters or {"device": "auto", "output_format": "wav", "use_tta": False})
        pairs = list(files)
        if not pairs:
            return []
        validation_error = self.model.validation_error()
        if validation_error or not self.runtime_available:
            if missing_policy == "skip":
                message = validation_error or "未安装 MSST/pymss 推理运行时，无法执行去混响"
                return [MSSTPreprocessResult("skipped", source, source, self.model.name, self.model.version, parameters, message) for source, _ in pairs]
            raise RuntimeError(validation_error or "未安装 MSST/pymss 推理运行时，无法执行去混响")
        if self.separator_factory is None:
            return [self.prepare_input(source, destination, missing_policy, parameters) for source, destination in pairs]
        separator = self.separator_factory(self.model, parameters)
        results = []
        try:
            for source, destination in pairs:
                destination.parent.mkdir(parents=True, exist_ok=True)
                separator.separate_file(source, destination)
                results.append(MSSTPreprocessResult("succeeded", source, destination, self.model.name, self.model.version, parameters))
        finally:
            close = getattr(separator, "close", None)
            if callable(close):
                close()
        return results

    @staticmethod
    def pymss_separator_factory(model: MSSTModelSpec, parameters: dict[str, object]):
        try:
            from pymss import MSSeparator
        except ImportError as exc:
            raise RuntimeError("未安装 MSST/pymss 推理运行时") from exc

        separator = MSSeparator(
            model_type=model.model_type,
            config_path=str(model.config_path),
            model_path=str(model.checkpoint_path),
            device=str(parameters.get("device", "auto")),
            output_format="wav",
            store_dirs=str(Path(parameters.get("cache_dir", "."))),
            use_tta=bool(parameters.get("use_tta", False)),
            inference_params={"batch_size": 1, "overlap_size": int(parameters.get("overlap_size", 17638))},
        )

        class SeparatorAdapter:
            def separate_file(self, source: Path, destination: Path) -> None:
                import numpy as np
                import soundfile as sf
                audio, sample_rate = sf.read(str(source), always_2d=False)
                channels_first = audio.T if getattr(audio, "ndim", 1) == 2 else audio
                stems = separator.separate(channels_first, pbar=False, stems=["noreverb"])
                if "noreverb" not in stems:
                    raise RuntimeError("MSST 未返回 noreverb 去混响音轨")
                cleaned = np.asarray(stems["noreverb"])
                if cleaned.ndim == 2 and cleaned.shape[1] > 4:
                    cleaned = cleaned.T
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(destination), cleaned, sample_rate, subtype="PCM_16")

            def close(self) -> None:
                if hasattr(separator, "del_cache"):
                    separator.del_cache()

        return SeparatorAdapter()

    def prepare_input(
        self,
        input_path: Path,
        output_path: Path,
        missing_policy: str = "error",
        parameters: dict[str, object] | None = None,
    ) -> MSSTPreprocessResult:
        parameters = dict(parameters or {"device": "auto", "output_format": "wav", "use_tta": False})
        if missing_policy not in {"error", "skip"}:
            raise ValueError("MSST 缺失策略只能是 error 或 skip")
        validation_error = self.model.validation_error()
        if validation_error:
            if missing_policy == "skip":
                return MSSTPreprocessResult("skipped", input_path, input_path, self.model.name, self.model.version, parameters, validation_error)
            raise RuntimeError(validation_error)
        if not self.runtime_available or self.runner is None:
            message = "未安装 MSST/pymss 推理运行时，无法执行去混响"
            if missing_policy == "skip":
                return MSSTPreprocessResult("skipped", input_path, input_path, self.model.name, self.model.version, parameters, message)
            raise RuntimeError(message)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.runner(self.model, input_path, output_path, parameters)
        return MSSTPreprocessResult("succeeded", input_path, output_path, self.model.name, self.model.version, parameters)

    @staticmethod
    def _pymss_runner(model: MSSTModelSpec, input_path: Path, output_path: Path, parameters: dict[str, object]) -> None:
        """调用 pymss API；依赖不存在时由上层以明确错误处理。"""
        try:
            from pymss import MSSeparator
        except ImportError as exc:
            raise RuntimeError("未安装 pymss/MSST 推理运行时") from exc
        separator = MSSeparator(
            model_type=model.model_type,
            config_path=str(model.config_path),
            model_path=str(model.checkpoint_path),
            device=str(parameters.get("device", "auto")),
            output_format="wav",
            store_dirs=str(output_path.parent),
            use_tta=bool(parameters.get("use_tta", False)),
            inference_params={
                "batch_size": 1,
                "overlap_size": int(parameters.get("overlap_size", 17638)),
            },
        )
        try:
            import numpy as np
            import soundfile as sf

            audio, sample_rate = sf.read(str(input_path), always_2d=False)
            channels_first = audio.T if getattr(audio, "ndim", 1) == 2 else audio
            stems = separator.separate(channels_first, pbar=False, stems=["noreverb"])
            if "noreverb" not in stems:
                raise RuntimeError("MSST 未返回 noreverb 去混响音轨")
            cleaned = np.asarray(stems["noreverb"])
            if cleaned.ndim == 2 and cleaned.shape[1] <= 4:
                pass
            elif cleaned.ndim == 2:
                cleaned = cleaned.T
            sf.write(str(output_path), cleaned, sample_rate, subtype="PCM_16")
        finally:
            if hasattr(separator, "del_cache"):
                separator.del_cache()
