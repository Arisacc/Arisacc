import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestRepairChain(unittest.TestCase):
    def test_chain_validator_rejects_too_many_steps_and_invalid_order(self):
        from audio_dataset_agent.core.models import RepairChain, RepairPlan
        from audio_dataset_agent.repair.chain import validate_repair_chain

        too_many = RepairChain(
            chain_id="chain_long",
            sample_id="sample_1",
            plans=[
                RepairPlan("sample_1", "noise_reduction", {"noise_reduction_db": 8}),
                RepairPlan("sample_1", "noise_reduction", {"noise_reduction_db": 8}),
                RepairPlan("sample_1", "noise_reduction", {"noise_reduction_db": 8}),
                RepairPlan("sample_1", "noise_reduction", {"noise_reduction_db": 8}),
            ],
        )
        invalid_order = RepairChain(
            chain_id="chain_order",
            sample_id="sample_1",
            plans=[
                RepairPlan("sample_1", "loudness_normalization", {"target_lufs": -23}, chain_id="chain_order", step_index=1),
                RepairPlan("sample_1", "silence_trimming", {"silence_threshold_dbfs": -50, "min_silence_duration_ms": 200}, chain_id="chain_order", step_index=2),
            ],
        )

        with self.assertRaisesRegex(ValueError, "最多"):
            validate_repair_chain(too_many)
        with self.assertRaisesRegex(ValueError, "顺序"):
            validate_repair_chain(invalid_order)

    def test_auto_chain_builds_ordered_plans_from_repair_decision(self):
        from audio_dataset_agent.core.models import CuratorDecision, DecisionType
        from audio_dataset_agent.repair.chain import build_repair_chain

        decision = CuratorDecision(
            sample_id="sample_chain",
            decision=DecisionType.REPAIR,
            reason="noise and quiet audio",
            confidence=0.8,
            recommended_action="denoise_light",
        )

        chain = build_repair_chain(decision, target_policy="tts", mode="auto")

        self.assertEqual(chain.sample_id, "sample_chain")
        self.assertEqual(
            [plan.operation for plan in chain.plans],
            ["noise_reduction", "loudness_normalization"],
        )
        self.assertEqual([plan.step_index for plan in chain.plans], [1, 2])
        self.assertTrue(all(plan.chain_id == chain.chain_id for plan in chain.plans))

    def test_single_mode_keeps_existing_one_step_behavior(self):
        from audio_dataset_agent.core.models import CuratorDecision, DecisionType
        from audio_dataset_agent.repair.chain import build_repair_chain

        decision = CuratorDecision(
            sample_id="sample_single",
            decision=DecisionType.REPAIR,
            reason="silence",
            confidence=0.8,
            recommended_action="trim_silence",
        )

        chain = build_repair_chain(decision, target_policy="voice_clone", mode="single")

        self.assertEqual(len(chain.plans), 1)
        self.assertEqual(chain.plans[0].operation, "silence_trimming")

    def test_final_selector_rolls_back_when_any_chain_step_is_not_adopted(self):
        from audio_dataset_agent.core.models import (
            Batch,
            CandidateSample,
            CuratorDecision,
            DecisionType,
            RepairEvaluation,
            RepairExecution,
        )
        from audio_dataset_agent.selection.service import select_final_samples

        original = Path("original.wav")
        repaired = Path("repaired.wav")
        batch = Batch(
            batch_id="batch_rollback",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy="tts",
            candidate_samples=[CandidateSample(
                sample_id="sample_rollback", batch_id="batch_rollback", asset_id="asset",
                source_path=original, sample_path=original, start_time=0.0, end_time=2.0, duration=2.0,
            )],
            curator_decisions=[CuratorDecision(
                "sample_rollback", DecisionType.REPAIR, "repair", 0.8, "denoise_light",
            )],
            repair_executions=[
                RepairExecution("sample_rollback", "noise_reduction", original, repaired, "succeeded", chain_id="chain", step_index=1),
                RepairExecution("sample_rollback", "loudness_normalization", repaired, Path("final.wav"), "succeeded", chain_id="chain", step_index=2),
            ],
            repair_evaluations=[
                RepairEvaluation("sample_rollback", "noise_reduction", {}, {}, True, "adopt_repaired", chain_id="chain", step_index=1),
                RepairEvaluation("sample_rollback", "loudness_normalization", {}, {}, False, "keep_original", chain_id="chain", step_index=2),
            ],
        )

        selection = select_final_samples(batch).final_selections[0]

        self.assertEqual(selection.status, "needs_review")
        self.assertEqual(selection.selected_audio_path, original)

    def test_plan_repairs_auto_mode_persists_chain_and_flattened_plans(self):
        from audio_dataset_agent.core.models import Batch, CuratorDecision, DecisionType
        from audio_dataset_agent.repair.service import plan_repairs

        batch = Batch(
            batch_id="batch_plan_chain",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy="tts",
            run_config={"repair_chain": "auto"},
            curator_decisions=[CuratorDecision(
                "sample_plan", DecisionType.REPAIR, "repair", 0.8, "denoise_light",
            )],
        )

        planned = plan_repairs(batch)

        self.assertEqual(planned.repair_chain_count, 1)
        self.assertEqual(len(planned.repair_plans), 2)
        self.assertEqual(planned.repair_plans[1].step_index, 2)

    def test_manifest_preserves_all_repair_chain_steps(self):
        from audio_dataset_agent.core.models import Batch, CandidateSample, RepairChain, RepairPlan
        from audio_dataset_agent.exporters.manifest import build_manifest_records

        plans = [
            RepairPlan("sample_manifest", "noise_reduction", {"noise_reduction_db": 8}, chain_id="chain_manifest", step_index=1),
            RepairPlan("sample_manifest", "loudness_normalization", {"target_lufs": -20}, chain_id="chain_manifest", step_index=2),
        ]
        batch = Batch(
            batch_id="batch_manifest",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy="asr",
            candidate_samples=[CandidateSample(
                sample_id="sample_manifest", batch_id="batch_manifest", asset_id="asset",
                source_path=Path("sample.wav"), start_time=0.0, end_time=2.0, duration=2.0,
            )],
            repair_plans=plans,
            repair_chains=[RepairChain("chain_manifest", "sample_manifest", plans)],
        )

        record = build_manifest_records(batch)[0]

        self.assertEqual(record["repair_chain_id"], "chain_manifest")
        self.assertEqual(
            [step["operation"] for step in record["repair_chain"]],
            ["noise_reduction", "loudness_normalization"],
        )

    def test_execute_repairs_runs_chain_sequentially(self):
        from audio_dataset_agent.core.models import (
            Batch,
            CandidateSample,
            CuratorDecision,
            DecisionType,
            RepairChain,
            RepairPlan,
            RepairExecution,
        )
        from audio_dataset_agent.repair.service import execute_repairs

        class RecordingExecutor:
            def __init__(self):
                self.calls = []

            def execute(self, plan, sample, repairs_dir):
                output = repairs_dir / f"step_{plan.step_index}.wav"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(f"step-{plan.step_index}".encode())
                self.calls.append((plan.step_index, sample.sample_path))
                return RepairExecution(
                    sample_id=plan.sample_id,
                    operation=plan.operation,
                    input_path=sample.sample_path or sample.source_path,
                    output_path=output,
                    status="succeeded",
                    chain_id=plan.chain_id,
                    step_index=plan.step_index,
                )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            source.write_bytes(b"source")
            sample = CandidateSample(
                sample_id="sample_chain",
                batch_id="batch_chain",
                asset_id="asset_chain",
                source_path=source,
                sample_path=source,
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
            )
            plans = [
                RepairPlan("sample_chain", "noise_reduction", {"noise_reduction_db": 8}, chain_id="chain_1", step_index=1),
                RepairPlan("sample_chain", "loudness_normalization", {"target_lufs": -20}, chain_id="chain_1", step_index=2),
            ]
            batch = Batch(
                batch_id="batch_chain",
                input_dir=root,
                output_dir=root / "work",
                target_policy="asr",
                candidate_samples=[sample],
                curator_decisions=[CuratorDecision("sample_chain", DecisionType.REPAIR, "repair", 0.8, "denoise_light")],
                repair_plans=plans,
                repair_chains=[RepairChain("chain_1", "sample_chain", plans)],
            )
            executor = RecordingExecutor()

            result = execute_repairs(batch, executor=executor)

        self.assertEqual([step for step, _ in executor.calls], [1, 2])
        self.assertEqual(executor.calls[0][1], source)
        self.assertTrue(str(executor.calls[1][1]).endswith("step_1.wav"))
        self.assertEqual([item.step_index for item in result.repair_executions], [1, 2])


if __name__ == "__main__":
    unittest.main()
