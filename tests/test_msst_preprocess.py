import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audio_dataset_agent.preprocessing.msst import MSSTModelSpec, MSSTPreprocessor


class TestMSSTPreprocess(unittest.TestCase):
    def test_missing_runtime_blocks_by_default(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = MSSTModelSpec(
                name="deverb_bs_roformer_8_256dim_8depth",
                checkpoint_path=root / "model.ckpt",
                config_path=root / "model.yaml",
                model_type="bs_roformer",
            )
            spec.checkpoint_path.write_bytes(b"model")
            spec.config_path.write_text("config", encoding="utf-8")
            processor = MSSTPreprocessor(spec, runtime_available=False)
            with self.assertRaises(RuntimeError):
                processor.prepare_input(root / "input.wav", root / "output.wav", missing_policy="error")

    def test_missing_runtime_can_explicitly_skip(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = MSSTModelSpec("model", root / "model.ckpt", root / "model.yaml", "bs_roformer")
            processor = MSSTPreprocessor(spec, runtime_available=False)
            result = processor.prepare_input(root / "input.wav", root / "output.wav", missing_policy="skip")
            self.assertEqual(result.status, "skipped")
            self.assertEqual(result.output_path, root / "input.wav")

    def test_model_spec_reports_missing_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = MSSTModelSpec("model", root / "missing.ckpt", root / "missing.yaml", "bs_roformer")
            self.assertFalse(spec.is_ready())
            self.assertIn("模型权重不存在", spec.validation_error())

    def test_ready_model_requires_both_checkpoint_and_yaml(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.ckpt"
            config = root / "model.yaml"
            checkpoint.write_bytes(b"model")
            spec = MSSTModelSpec("model", checkpoint, config, "bs_roformer")
            self.assertFalse(spec.is_ready())
            config.write_text("config", encoding="utf-8")
            self.assertTrue(spec.is_ready())

    def test_batch_reuses_one_separator_and_processes_each_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.ckpt"
            config = root / "model.yaml"
            checkpoint.write_bytes(b"model")
            config.write_text("config", encoding="utf-8")
            inputs = [(root / "a.wav", root / "out" / "a.wav"), (root / "b.wav", root / "out" / "b.wav")]
            for source, _ in inputs:
                source.write_bytes(b"audio")
            create_count = []
            processed = []

            class FakeSeparator:
                def separate_file(self, source, destination):
                    processed.append(source.name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(b"clean")

                def close(self):
                    pass

            def factory(model, parameters):
                create_count.append(model.name)
                return FakeSeparator()

            processor = MSSTPreprocessor(
                MSSTModelSpec("model", checkpoint, config),
                separator_factory=factory,
            )
            results = processor.prepare_files(inputs)
            self.assertEqual(len(create_count), 1)
            self.assertEqual(processed, ["a.wav", "b.wav"])
            self.assertEqual([item.status for item in results], ["succeeded", "succeeded"])


if __name__ == "__main__":
    unittest.main()
