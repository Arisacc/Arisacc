import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class TestSileroVADAnalyzer(unittest.TestCase):
    def test_default_model_loader_uses_onnx_backend_for_windows_path_compatibility(self):
        from audio_dataset_agent.analyzers.silero_vad import SileroVADAnalyzer

        with patch("silero_vad.load_silero_vad") as load_model:
            load_model.return_value = "onnx-model"

            result = SileroVADAnalyzer()._default_model_loader()

        self.assertEqual(result, "onnx-model")
        load_model.assert_called_once_with(onnx=True)

    def test_detect_converts_silero_timestamps_to_speech_segments(self):
        from audio_dataset_agent.analyzers.silero_vad import SileroVADAnalyzer

        calls = []

        def fake_model_loader():
            calls.append(("load_model",))
            return "fake-model"

        def fake_audio_reader(audio_path, sampling_rate):
            calls.append(("read_audio", audio_path, sampling_rate))
            return "fake-waveform"

        def fake_timestamp_getter(wav, model, **kwargs):
            calls.append(("get_timestamps", wav, model, kwargs))
            return [
                {"start": 0.12, "end": 0.78},
                {"start": 1.5, "end": 2.25},
            ]

        segments = SileroVADAnalyzer(
            sampling_rate=16000,
            threshold=0.6,
            min_speech_duration_ms=200,
            min_silence_duration_ms=120,
            speech_pad_ms=40,
            model_loader=fake_model_loader,
            audio_reader=fake_audio_reader,
            timestamp_getter=fake_timestamp_getter,
        ).detect(Path("voice.wav"))

        self.assertEqual(segments[0].start_time, 0.12)
        self.assertEqual(segments[0].end_time, 0.78)
        self.assertEqual(segments[1].start_time, 1.5)
        self.assertEqual(segments[1].end_time, 2.25)
        self.assertEqual(calls[0], ("load_model",))
        self.assertEqual(calls[1], ("read_audio", "voice.wav", 16000))
        self.assertEqual(calls[2][0], "get_timestamps")
        self.assertEqual(calls[2][1], "fake-waveform")
        self.assertEqual(calls[2][2], "fake-model")
        self.assertEqual(calls[2][3]["threshold"], 0.6)
        self.assertEqual(calls[2][3]["return_seconds"], True)
        self.assertEqual(calls[2][3]["min_speech_duration_ms"], 200)
        self.assertEqual(calls[2][3]["min_silence_duration_ms"], 120)
        self.assertEqual(calls[2][3]["speech_pad_ms"], 40)

    def test_detect_ignores_non_positive_segments(self):
        from audio_dataset_agent.analyzers.silero_vad import SileroVADAnalyzer

        def fake_timestamp_getter(wav, model, **kwargs):
            return [
                {"start": 1.0, "end": 1.0},
                {"start": 2.0, "end": 1.5},
                {"start": 3.0, "end": 3.5},
            ]

        segments = SileroVADAnalyzer(
            model_loader=lambda: object(),
            audio_reader=lambda audio_path, sampling_rate: object(),
            timestamp_getter=fake_timestamp_getter,
        ).detect(Path("voice.wav"))

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start_time, 3.0)
        self.assertEqual(segments[0].end_time, 3.5)

    def test_default_audio_reader_uses_soundfile_tensor_without_torchaudio(self):
        from audio_dataset_agent.analyzers.silero_vad import SileroVADAnalyzer

        stereo_audio = np.array(
            [
                [0.25, 0.75],
                [0.5, 1.0],
            ],
            dtype=np.float32,
        )

        def fake_soundfile_read(audio_path, always_2d):
            self.assertEqual(audio_path, "voice.wav")
            self.assertTrue(always_2d)
            return stereo_audio, 16000

        with patch("soundfile.read", fake_soundfile_read):
            wav = SileroVADAnalyzer(sampling_rate=16000)._read_audio("voice.wav")

        self.assertEqual(tuple(wav.shape), (2,))
        self.assertAlmostEqual(float(wav[0]), 0.5)
        self.assertAlmostEqual(float(wav[1]), 0.75)

    def test_default_audio_reader_rejects_unexpected_sample_rate(self):
        from audio_dataset_agent.analyzers.silero_vad import SileroVADAnalyzer

        def fake_soundfile_read(audio_path, always_2d):
            return np.zeros((2, 1), dtype=np.float32), 8000

        with patch("soundfile.read", fake_soundfile_read):
            with self.assertRaisesRegex(ValueError, "Expected 16000 Hz"):
                SileroVADAnalyzer(sampling_rate=16000)._read_audio("voice.wav")




class TestWaveEnergyVADPostProcessing(unittest.TestCase):
    def test_merge_gap_combines_nearby_segments_before_min_duration_filter(self):
        from audio_dataset_agent.analyzers.vad import WaveEnergyVADAnalyzer

        analyzer = WaveEnergyVADAnalyzer(min_duration_seconds=1.0, merge_gap_seconds=0.35)
        merged = analyzer._post_process_ranges([(0.0, 0.45), (0.7, 1.25), (2.0, 2.2)])

        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].start_time, 0.0)
        self.assertAlmostEqual(merged[0].end_time, 1.25)

    def test_whole_file_vad_returns_complete_audio_duration(self):
        import wave
        from tempfile import TemporaryDirectory
        from audio_dataset_agent.analyzers.vad import WholeFileVADAnalyzer

        with TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "sample.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 24000)

            segments = WholeFileVADAnalyzer().detect(wav_path)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start_time, 0.0)
        self.assertAlmostEqual(segments[0].end_time, 1.5)

if __name__ == "__main__":
    unittest.main()
