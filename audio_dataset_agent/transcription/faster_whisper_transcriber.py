# faster-whisper 适配器：本地 ASR 候选文本生成。
from collections.abc import Callable
from math import exp
from typing import Any

from audio_dataset_agent.core.models import CandidateSample
from audio_dataset_agent.transcription.transcriber import TranscriptResult


class FasterWhisperTranscriber:
    """将 faster-whisper 输出转换为 TranscriptResult 的适配器。"""

    def __init__(
        self,
        model_size_or_path: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        language: str | None = None,
        vad_filter: bool = False,
        model_factory: Callable[..., Any] | None = None,
    ):
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.vad_filter = vad_filter
        self._model_factory = model_factory
        self._model: Any | None = None

    def transcribe(self, sample: CandidateSample) -> TranscriptResult:
        audio_path = sample.sample_path or sample.source_path
        segments, _info = self._load_model().transcribe(
            str(audio_path),
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=self.vad_filter,
        )
        segment_list = list(segments)
        text = " ".join(
            segment.text.strip()
            for segment in segment_list
            if getattr(segment, "text", "").strip()
        )
        return TranscriptResult(
            text=text,
            confidence=self._estimate_confidence(segment_list),
            source="faster_whisper",
        )

    def _load_model(self) -> Any:
        if self._model is None:
            factory = self._model_factory or self._default_model_factory
            self._model = factory(
                model_size_or_path=self.model_size_or_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def _default_model_factory(self, **kwargs: Any) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install it or use --asr fixed."
            ) from exc

        return WhisperModel(**kwargs)

    def _estimate_confidence(self, segments: list[Any]) -> float:
        if not segments:
            return 0.0

        confidence_values = []
        for segment in segments:
            avg_logprob = getattr(segment, "avg_logprob", None)
            no_speech_prob = getattr(segment, "no_speech_prob", None)
            if avg_logprob is None:
                continue

            token_confidence = max(0.0, min(1.0, exp(avg_logprob)))
            speech_confidence = 1.0
            if no_speech_prob is not None:
                speech_confidence = max(0.0, min(1.0, 1.0 - no_speech_prob))
            confidence_values.append(token_confidence * speech_confidence)

        if not confidence_values:
            return 0.5
        return sum(confidence_values) / len(confidence_values)
