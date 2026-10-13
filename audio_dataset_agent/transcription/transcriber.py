# 转写接口定义：统一 fixed/faster-whisper 等 ASR 返回结构。
from dataclasses import dataclass
from typing import Protocol

from audio_dataset_agent.core.models import CandidateSample


@dataclass(frozen=True)
class TranscriptResult:
    # TranscriptResult 统一不同 ASR/文本来源的输出格式。
    text: str
    confidence: float
    source: str = "asr"

#协议接口
class Transcriber(Protocol):
    # Transcriber 是转写接口；faster-whisper、fun-asr、Qwen 多模态都可以实现它。
    def transcribe(self, sample: CandidateSample) -> TranscriptResult:
        """返回候选样本的转写文本和置信度。"""

#开发和测试用固定实现
class FixedTranscriber:
    """用于测试和早期流程开发的确定性转写器。"""

    def __init__(self, result: TranscriptResult):
        self._result = result

    def transcribe(self, sample: CandidateSample) -> TranscriptResult:
        return self._result
