# 音频指标分析：计算时长、响度、静音比例、削波等质量证据。
import math
import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


@dataclass(frozen=True)
class AudioMetrics:
    """供 Inspector 和 Agent 消费的结构化音频质量证据。"""

    rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    noise_floor_dbfs: float | None = None
    dynamic_range_db: float | None = None
    dc_offset: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        payload: dict[str, float | None] = {
            "rms_dbfs": self.rms_dbfs,
            "peak_dbfs": self.peak_dbfs,
            "clipping_ratio": self.clipping_ratio,
            "silence_ratio": self.silence_ratio,
        }
        if self.noise_floor_dbfs is not None:
            payload["noise_floor_dbfs"] = self.noise_floor_dbfs
        if self.dynamic_range_db is not None:
            payload["dynamic_range_db"] = self.dynamic_range_db
        if self.dc_offset is not None:
            payload["dc_offset"] = self.dc_offset
        return payload


class AudioMetricsAnalyzer(Protocol):
    def analyze(self, audio_path: Path) -> AudioMetrics:
        """返回单个样本文件的音频质量指标。"""


class WaveAudioMetricsAnalyzer:
    """使用标准库分析单声道或多声道 PCM WAV 文件。"""

    def __init__(self, silence_threshold_dbfs: float = -50.0):
        self.silence_threshold_dbfs = silence_threshold_dbfs

    def analyze(self, audio_path: Path) -> AudioMetrics:
        with wave.open(str(audio_path), "rb") as wav_file:
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            frames = wav_file.readframes(frame_count)

        if sample_width not in {1, 2, 4}:
            raise ValueError(f"Unsupported PCM sample width: {sample_width}")

        if frame_count == 0 or not frames:
            return AudioMetrics(
                rms_dbfs=-math.inf,
                peak_dbfs=-math.inf,
                clipping_ratio=0.0,
                silence_ratio=1.0,
            )

        max_amplitude = float(2 ** (8 * sample_width - 1))
        samples = self._read_samples(frames, sample_width)
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        peak = max(abs(sample) for sample in samples)
        clip_threshold = max_amplitude - 1
        silence_threshold = max_amplitude * (
            10 ** (self.silence_threshold_dbfs / 20)
        )

        clipping_count = sum(1 for sample in samples if abs(sample) >= clip_threshold)
        silence_count = sum(1 for sample in samples if abs(sample) <= silence_threshold)

        return AudioMetrics(
            rms_dbfs=self._to_dbfs(rms, max_amplitude),
            peak_dbfs=self._to_dbfs(peak, max_amplitude),
            clipping_ratio=clipping_count / len(samples),
            silence_ratio=silence_count / len(samples),
            noise_floor_dbfs=self._noise_floor_dbfs(samples, max_amplitude),
            dynamic_range_db=self._dynamic_range_db(samples, max_amplitude),
            dc_offset=self._dc_offset(samples, max_amplitude),
        )

    def _read_samples(self, frames: bytes, sample_width: int) -> list[int]:
        if sample_width == 1:
            return [byte - 128 for byte in frames]

        samples = []
        for index in range(0, len(frames), sample_width):
            chunk = frames[index : index + sample_width]
            samples.append(int.from_bytes(chunk, byteorder="little", signed=True))
        return samples

    def _to_dbfs(self, amplitude: float, max_amplitude: float) -> float:
        if amplitude <= 0:
            return -math.inf
        return 20 * math.log10(amplitude / max_amplitude)

    def _dynamic_range_db(self, samples: list[int], max_amplitude: float) -> float:
        peak = max(abs(sample) for sample in samples)
        non_silent = [abs(sample) for sample in samples if sample != 0]
        if peak <= 0 or not non_silent:
            return 0.0
        floor = min(non_silent)
        return self._to_dbfs(peak, max_amplitude) - self._to_dbfs(floor, max_amplitude)

    def _dc_offset(self, samples: list[int], max_amplitude: float) -> float:
        if not samples:
            return 0.0
        return (sum(samples) / len(samples)) / max_amplitude

    def _noise_floor_dbfs(self, samples: list[int], max_amplitude: float) -> float:
        non_silent = sorted(abs(sample) for sample in samples if sample != 0)
        if not non_silent:
            return -math.inf
        index = max(0, int(len(non_silent) * 0.1) - 1)
        return self._to_dbfs(non_silent[index], max_amplitude)


FFmpegMetricsRunner = Callable[[list[str]], str]


def run_ffmpeg_metrics(command: list[str]) -> str:
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    return f"{process.stdout}\n{process.stderr}"


class FFmpegAudioMetricsAnalyzer:
    """使用 FFmpeg astats 和 silencedetect 分析音频。"""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        silence_threshold_dbfs: float = -50.0,
        runner: FFmpegMetricsRunner = run_ffmpeg_metrics,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.silence_threshold_dbfs = silence_threshold_dbfs
        self.runner = runner

    def analyze(self, audio_path: Path) -> AudioMetrics:
        astats_output = self.runner(self._astats_command(audio_path))
        silencedetect_output = self.runner(self._silencedetect_command(audio_path))
        duration = self._duration_seconds(astats_output, silencedetect_output)
        silence_seconds = self._silence_seconds(silencedetect_output)
        silence_ratio = 0.0
        if duration > 0:
            silence_ratio = min(1.0, silence_seconds / duration)

        return AudioMetrics(
            rms_dbfs=self._parse_float(astats_output, "RMS level dB", -math.inf),
            peak_dbfs=self._parse_float(astats_output, "Peak level dB", -math.inf),
            clipping_ratio=0.0,
            silence_ratio=silence_ratio,
            noise_floor_dbfs=self._parse_optional_float(astats_output, "Noise floor dB"),
            dynamic_range_db=self._parse_optional_float(astats_output, "Dynamic range"),
            dc_offset=self._parse_optional_float(astats_output, "DC offset"),
        )

    def _astats_command(self, audio_path: Path) -> list[str]:
        return [
            self.ffmpeg_path,
            "-hide_banner",
            "-i",
            str(audio_path),
            "-af",
            "astats=metadata=0:reset=0",
            "-f",
            "null",
            "-",
        ]

    def _silencedetect_command(self, audio_path: Path) -> list[str]:
        return [
            self.ffmpeg_path,
            "-hide_banner",
            "-i",
            str(audio_path),
            "-af",
            f"silencedetect=noise={self.silence_threshold_dbfs}dB:d=0.1",
            "-f",
            "null",
            "-",
        ]

    def _parse_float(self, text: str, label: str, default: float) -> float:
        value = self._parse_optional_float(text, label)
        if value is None:
            return default
        return value

    def _parse_optional_float(self, text: str, label: str) -> float | None:
        match = re.search(
            rf"{re.escape(label)}\s*:\s*(-?(?:\d+(?:\.\d+)?|\.\d+))",
            text,
        )
        if match is None:
            return None
        return float(match.group(1))

    def _duration_seconds(self, *outputs: str) -> float:
        for text in outputs:
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
            if match is not None:
                return (
                    int(match.group(1)) * 3600
                    + int(match.group(2)) * 60
                    + float(match.group(3))
                )

        silence_ends = []
        for text in outputs:
            silence_ends.extend(
                float(value)
                for value in re.findall(
                    r"silence_end:\s*(-?(?:\d+(?:\.\d+)?|\.\d+))",
                    text,
                )
            )
        if silence_ends:
            return max(silence_ends)
        return 0.0

    def _silence_seconds(self, text: str) -> float:
        return sum(
            float(value)
            for value in re.findall(
                r"silence_duration:\s*(-?(?:\d+(?:\.\d+)?|\.\d+))",
                text,
            )
        )
