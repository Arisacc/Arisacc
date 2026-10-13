# VAD 抽象与本地实现：包含 fixed、whole-file、wave-energy 三种模式。
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol
import wave


@dataclass(frozen=True)
class SpeechSegment:
    # One detected speech interval in seconds.
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 6)


class VADAnalyzer(Protocol):
    # Voice activity detection interface used by segmentation services.
    def detect(self, audio_path: Path) -> list[SpeechSegment]:
        """Return detected speech segments from a working audio file."""


class FixedVADAnalyzer:
    """Deterministic VAD used by tests and early pipeline smoke runs."""

    def __init__(self, segments: list[SpeechSegment]):
        self._segments = segments

    def detect(self, audio_path: Path) -> list[SpeechSegment]:
        return list(self._segments)


class WholeFileVADAnalyzer:
    """Treat the normalized audio file as one candidate speech segment."""

    def detect(self, audio_path: Path) -> list[SpeechSegment]:
        with wave.open(str(audio_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
        if sample_rate <= 0:
            return []
        duration = round(frame_count / sample_rate, 6)
        if duration <= 0:
            return []
        return [SpeechSegment(start_time=0.0, end_time=duration)]


class WaveEnergyVADAnalyzer:
    """Simple PCM WAV VAD based on frame-level RMS energy."""

    def __init__(
        self,
        frame_ms: int = 30,
        energy_threshold_dbfs: float = -40.0,
        min_duration_seconds: float = 0.8,
        merge_gap_seconds: float = 0.35,
    ):
        self.frame_ms = frame_ms
        self.energy_threshold_dbfs = energy_threshold_dbfs
        self.min_duration_seconds = min_duration_seconds
        self.merge_gap_seconds = merge_gap_seconds

    def detect(self, audio_path: Path) -> list[SpeechSegment]:
        with wave.open(str(audio_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            frames = wav_file.readframes(frame_count)

        if sample_width not in {1, 2, 4}:
            raise ValueError(f"Unsupported PCM sample width: {sample_width}")

        samples = self._read_samples(frames, sample_width)
        samples_per_frame = max(1, int(sample_rate * self.frame_ms / 1000))
        active_ranges = []
        active_start: float | None = None

        for start in range(0, len(samples), samples_per_frame):
            frame_samples = samples[start : start + samples_per_frame]
            if not frame_samples:
                continue

            frame_start = start / sample_rate
            is_active = (
                self._rms_dbfs(frame_samples, sample_width)
                >= self.energy_threshold_dbfs
            )

            if is_active and active_start is None:
                active_start = frame_start
            elif not is_active and active_start is not None:
                active_ranges.append((active_start, frame_start))
                active_start = None

        if active_start is not None:
            active_ranges.append((active_start, len(samples) / sample_rate))

        return self._post_process_ranges(active_ranges)

    def _post_process_ranges(self, ranges: list[tuple[float, float]]) -> list[SpeechSegment]:
        if not ranges:
            return []

        ordered = sorted((start, end) for start, end in ranges if end > start)
        merged: list[tuple[float, float]] = []
        for start, end in ordered:
            if not merged:
                merged.append((start, end))
                continue
            previous_start, previous_end = merged[-1]
            if start - previous_end <= self.merge_gap_seconds:
                merged[-1] = (previous_start, max(previous_end, end))
            else:
                merged.append((start, end))

        return [
            SpeechSegment(start_time=round(start, 6), end_time=round(end, 6))
            for start, end in merged
            if end - start >= self.min_duration_seconds
        ]

    def _read_samples(self, frames: bytes, sample_width: int) -> list[int]:
        if sample_width == 1:
            return [byte - 128 for byte in frames]

        samples = []
        for index in range(0, len(frames), sample_width):
            chunk = frames[index : index + sample_width]
            samples.append(int.from_bytes(chunk, byteorder="little", signed=True))
        return samples

    def _rms_dbfs(self, samples: list[int], sample_width: int) -> float:
        if not samples:
            return -math.inf

        max_amplitude = float(2 ** (8 * sample_width - 1))
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        if rms <= 0:
            return -math.inf
        return 20 * math.log10(rms / max_amplitude)
