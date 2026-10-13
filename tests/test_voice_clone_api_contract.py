import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestVoiceCloneApiContract(unittest.TestCase):
    def test_voice_clone_is_the_only_supported_web_profile(self):
        from fastapi.testclient import TestClient
        from audio_dataset_agent.web.fastapi_server import create_app

        with TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "audio"
            input_dir.mkdir()
            client = TestClient(create_app(Path(tmp) / "runs", None))
            response = client.post(
                "/audio-dataset/jobs",
                json={"input_dir": str(input_dir), "target_policy": "tts"},
            )
            self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
