import unittest
from pathlib import Path

from local_tools.download_models import MODEL_OPTIONS, build_parser, resolve_option


class TestModelDownloadMenu(unittest.TestCase):
    def test_registry_contains_qwen_faster_whisper_and_msst(self):
        self.assertEqual(set(MODEL_OPTIONS), {"qwen", "faster-whisper", "msst"})
        self.assertEqual(resolve_option("qwen").local_dir, Path("local_tools/models/qwen/Qwen3-4B-Instruct-2507"))
        self.assertEqual(resolve_option("faster-whisper").local_dir, Path("local_tools/models/faster-whisper/small"))

    def test_cli_parser_supports_non_interactive_selection(self):
        args = build_parser().parse_args(["--model", "faster-whisper", "--yes"])
        self.assertEqual(args.model, "faster-whisper")
        self.assertTrue(args.yes)


if __name__ == "__main__":
    unittest.main()
