import unittest
from types import SimpleNamespace
from audio_dataset_agent.orchestrator.agent_graph import AgentGraphRecorder


class TestStageProgress(unittest.TestCase):
    def test_graph_node_emits_started_and_completed_events(self):
        events = []
        graph = AgentGraphRecorder(progress_callback=events.append)
        batch = SimpleNamespace(batch_id="batch-1", phase=SimpleNamespace(value="created"), raw_asset_count=0, candidate_sample_count=0, analysis_result_count=0, processing_strategy_count=0, curator_decision_count=0, repair_plan_count=0, repair_execution_count=0, repair_evaluation_count=0, final_selection_count=0, export_paths={})
        graph.run_node("import", "Data Intake", batch, lambda current: current, notes="test")
        self.assertEqual([item["event"] for item in events], ["stage_started", "stage_completed"])
        self.assertEqual(events[0]["stage"], "import")
        self.assertEqual(events[1]["percent"], 100)


if __name__ == "__main__":
    unittest.main()
