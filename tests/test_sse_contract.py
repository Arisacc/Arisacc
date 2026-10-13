import unittest


class TestSSEContract(unittest.TestCase):
    def test_event_payload_is_json_serializable(self):
        import json

        event = {
            "event_id": 1,
            "event_type": "stage_succeeded",
            "stage": "inspect",
            "status": "running",
        }
        payload = json.dumps(event, ensure_ascii=False)
        self.assertIn('"event_type": "stage_succeeded"', payload)


if __name__ == "__main__":
    unittest.main()
