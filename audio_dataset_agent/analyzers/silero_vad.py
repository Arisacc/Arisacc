# Silero VAD 适配器：调用现成模型输出语音活动时间段。
from collections.abc import Callable
from pathlib import Path
from typing import Any

from audio_dataset_agent.analyzers.vad import SpeechSegment


class SileroVADAnalyzer:
    """将 silero-vad 时间戳转换为 SpeechSegment 对象的适配器。"""

    def __init__(
        self,
        sampling_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        model_loader: Callable[[], Any] | None = None,
        audio_reader: Callable[[str, int], Any] | None = None,
        timestamp_getter: Callable[..., list[dict[str, float]]] | None = None,
    ):
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self._model_loader = model_loader
        self._audio_reader = audio_reader
        self._timestamp_getter = timestamp_getter
        self._model: Any | None = None

    def detect(self, audio_path: Path) -> list[SpeechSegment]:
        model = self._load_model()
        wav = self._read_audio(str(audio_path))
        timestamps = self._get_timestamps(
            wav,
            model,
            sampling_rate=self.sampling_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=True,
        )
        return self._to_segments(timestamps)

    def _load_model(self) -> Any:
        if self._model is None:
            if self._model_loader is not None:
                self._model = self._model_loader()
            else:
                self._model = self._default_model_loader()
        return self._model

    def _read_audio(self, audio_path: str) -> Any:
        if self._audio_reader is not None:
            return self._audio_reader(audio_path, self.sampling_rate)
        return self._read_audio_with_soundfile(audio_path)

    def _get_timestamps(self, wav: Any, model: Any, **kwargs: Any) -> list[dict[str, float]]:
        if self._timestamp_getter is not None:
            return self._timestamp_getter(wav, model, **kwargs)
        _load_silero_vad, _read_audio, get_speech_timestamps = self._import_silero()
        return get_speech_timestamps(wav, model, **kwargs)

    def _default_model_loader(self) -> Any:
        load_silero_vad, _read_audio, _get_speech_timestamps = self._import_silero()
        # Windows 下 PyTorch JIT 对包含中文字符的安装路径兼容性较差，
        # 使用同包提供的 ONNX 权重可以避免模型文件存在但无法打开的问题。
        return load_silero_vad(onnx=True)

    def _import_silero(self) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
        except ImportError as exc:
            raise RuntimeError(
                "silero-vad is not installed. Install it or use --vad fixed/wave-energy."
            ) from exc

        return load_silero_vad, read_audio, get_speech_timestamps

    def _read_audio_with_soundfile(self, audio_path: str) -> Any:
        try:
            import soundfile as sf
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "soundfile and torch are required for Silero VAD audio reading."
            ) from exc

        audio, sample_rate = sf.read(audio_path, always_2d=True)
        if sample_rate != self.sampling_rate:
            raise ValueError(
                f"Expected {self.sampling_rate} Hz audio for Silero VAD, "
                f"got {sample_rate} Hz. Run FFmpeg normalization first."
            )

        mono_audio = audio.mean(axis=1)
        return torch.from_numpy(mono_audio).float()

    def _to_segments(self, timestamps: list[dict[str, float]]) -> list[SpeechSegment]:
        segments = []
        for timestamp in timestamps:
            start = float(timestamp["start"])
            end = float(timestamp["end"])
            if end <= start:
                continue
            segments.append(
                SpeechSegment(
                    start_time=round(start, 6),
                    end_time=round(end, 6),
                )
            )
        return segments
