import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestFastAPIHarnessServer(unittest.TestCase):
    def test_fastapi_validates_request_and_exposes_health_endpoint(self):
        try:
            from fastapi.testclient import TestClient
            from audio_dataset_agent.web.fastapi_server import create_app
        except ImportError:
            self.skipTest("FastAPI 可选依赖未安装")

        with TemporaryDirectory() as tmp:
            app = create_app(Path(tmp) / "runs", None)
            client = TestClient(app)
            response = client.get("/hello")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["data"]["runtime"], "fastapi-harness")

            invalid = client.post("/audio-dataset/jobs", json={"input_dir": "", "target_policy": "unknown"})
            self.assertEqual(invalid.status_code, 422)

            unknown_events = client.get("/audio-dataset/jobs/missing/events")
            self.assertEqual(unknown_events.status_code, 404)


if __name__ == "__main__":
    unittest.main()
