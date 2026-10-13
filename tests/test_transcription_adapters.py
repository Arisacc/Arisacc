import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestFasterWhisperTranscriber(unittest.TestCase):
    def test_transcribes_sample_path_with_injected_model(self):
        from audio_dataset_agent.core.models import CandidateSample
        from audio_dataset_agent.transcription.faster_whisper_transcriber import (
            FasterWhisperTranscriber,
        )

        calls = []

        class FakeSegment:
            def __init__(self, text, avg_logprob, no_speech_prob):
                self.text = text
                self.avg_logprob = avg_logprob
                self.no_speech_prob = no_speech_prob

        class FakeModel:
            def transcribe(self, audio_path, **kwargs):
                calls.append((audio_path, kwargs))
                return (
                    iter(
                        [
                            FakeSegment(" hello", math.log(0.9), 0.1),
                            FakeSegment(" world ", math.log(0.8), 0.2),
                        ]
                    ),
                    object(),
                )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source.wav"
            sample_path = root / "sample.wav"
            source_path.write_bytes(b"source")
            sample_path.write_bytes(b"sample")
            sample = CandidateSample(
                sample_id="sample_0001",
                batch_id="batch_0001",
                asset_id="asset_0001",
                source_path=source_path,
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
                sample_path=sample_path,
            )

            result = FasterWhisperTranscriber(
                model_factory=lambda **kwargs: FakeModel(),
                beam_size=3,
                language="zh",
                vad_filter=True,
            ).transcribe(sample)

        self.assertEqual(calls[0][0], str(sample_path))
        self.assertEqual(calls[0][1]["beam_size"], 3)
        self.assertEqual(calls[0][1]["language"], "zh")
        self.assertTrue(calls[0][1]["vad_filter"])
        self.assertEqual(result.text, "hello world")
        self.assertAlmostEqual(result.confidence, 0.725, places=4)
        self.assertEqual(result.source, "faster_whisper")

    def test_falls_back_to_source_path_when_sample_path_is_missing(self):
        from audio_dataset_agent.core.models import CandidateSample
        from audio_dataset_agent.transcription.faster_whisper_transcriber import (
            FasterWhisperTranscriber,
        )

        calls = []

        class FakeModel:
            def transcribe(self, audio_path, **kwargs):
                calls.append(audio_path)
                return (iter([]), object())

        sample = CandidateSample(
            sample_id="sample_0001",
            batch_id="batch_0001",
            asset_id="asset_0001",
            source_path=Path("source.wav"),
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
        )

        FasterWhisperTranscriber(
            model_factory=lambda **kwargs: FakeModel(),
        ).transcribe(sample)

        self.assertEqual(calls[0], "source.wav")


if __name__ == "__main__":
    unittest.main()
