import unittest


class TestLLMJsonParser(unittest.TestCase):
    def test_extracts_json_object_from_surrounding_text(self):
        from audio_dataset_agent.core.llm_json import parse_json_object

        payload = parse_json_object(
            """
            我会根据证据选择修复。
            {
              "decision": "repair",
              "confidence": 0.81
            }
            以上是结构化结果。
            """
        )

        self.assertEqual(payload["decision"], "repair")
        self.assertEqual(payload["confidence"], 0.81)

    def test_raises_value_error_when_no_json_object_exists(self):
        from audio_dataset_agent.core.llm_json import parse_json_object

        with self.assertRaises(ValueError):
            parse_json_object("not json")

