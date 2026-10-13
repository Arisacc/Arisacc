import io
import json
import unittest
from contextlib import redirect_stdout


class TestToolRegistry(unittest.TestCase):
    def test_default_tool_registry_has_unique_tool_ids(self):
        from audio_dataset_agent.tools.registry import default_tool_registry

        registry = default_tool_registry()
        tool_ids = [capability.tool_id for capability in registry.capabilities]

        self.assertGreaterEqual(len(tool_ids), 8)
        self.assertEqual(len(tool_ids), len(set(tool_ids)))

    def test_tool_registry_can_filter_by_family(self):
        from audio_dataset_agent.tools.registry import default_tool_registry

        registry = default_tool_registry()
        vad_tools = registry.by_family("vad")

        self.assertTrue(vad_tools)
        self.assertTrue(all(tool.family == "vad" for tool in vad_tools))

    def test_tool_registry_serializes_to_json_ready_dict(self):
        from audio_dataset_agent.tools.registry import default_tool_registry
        from audio_dataset_agent.tools.registry import tool_registry_to_dict

        payload = tool_registry_to_dict(default_tool_registry())
        encoded = json.dumps(payload, ensure_ascii=False)

        self.assertIn('"tool_id": "vad.silero"', encoded)
        self.assertIn("capabilities", payload)

    def test_default_registry_includes_lightweight_denoise_repair_tool(self):
        from audio_dataset_agent.tools.registry import default_tool_registry

        registry = default_tool_registry()
        capability = registry.get("repair.ffmpeg_denoise")

        self.assertIsNotNone(capability)
        self.assertEqual(capability.family, "repair")
        self.assertIn("audio_noise_high", capability.best_for)
        self.assertEqual(capability.configurable_parameters["noise_reduction_db"], 12.0)

    def test_default_registry_includes_ffmpeg_metrics_and_deepfilter_tools(self):
        from audio_dataset_agent.tools.registry import default_tool_registry

        registry = default_tool_registry()

        astats = registry.get("metrics.ffmpeg_astats")
        silencedetect = registry.get("metrics.ffmpeg_silencedetect")
        deepfilter = registry.get("repair.deepfilternet_denoise")

        self.assertIsNotNone(astats)
        self.assertEqual(astats.family, "audio_metrics")
        self.assertIn("audio_noise_high", astats.best_for)
        self.assertIn("noise_floor_dbfs", astats.output_artifacts)

        self.assertIsNotNone(silencedetect)
        self.assertEqual(silencedetect.family, "audio_metrics")
        self.assertIn("audio_high_silence_ratio", silencedetect.best_for)

        self.assertIsNotNone(deepfilter)
        self.assertEqual(deepfilter.family, "repair")
        self.assertIn("neural_denoise", deepfilter.best_for)
        self.assertIn("DeepFilterNet", deepfilter.dependencies)


class TestToolRegistryCli(unittest.TestCase):
    def test_cli_lists_tool_registry_without_input_dirs(self):
        from audio_dataset_agent.cli import main

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["--list-tools"])
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertIn("capabilities", payload)
        self.assertTrue(
            any(tool["tool_id"] == "asr.faster_whisper" for tool in payload["capabilities"])
        )


if __name__ == "__main__":
    unittest.main()
