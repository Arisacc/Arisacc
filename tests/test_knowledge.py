import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestLocalKnowledgeRetriever(unittest.TestCase):
    def test_retriever_builds_index_and_returns_filtered_source_hits(self):
        from audio_dataset_agent.knowledge.retriever import LocalKnowledgeRetriever

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            documents = root / "documents"
            (documents / "audio_tools").mkdir(parents=True)
            (documents / "audio_tools" / "deepfilternet.md").write_text(
                "---\ntool_ids: repair.deepfilternet_denoise\nprofiles: voice_clone\n---\n"
                "# DeepFilterNet\n## 降噪\n适合高噪声语音增强，但可能改变说话人音色。",
                encoding="utf-8",
            )
            (documents / "audio_tools" / "faster_whisper.md").write_text(
                "# faster-whisper\n用于 ASR 候选文本生成和置信度估计。",
                encoding="utf-8",
            )

            retriever = LocalKnowledgeRetriever(documents, root / "index.sqlite")
            hits = retriever.search(
                "voice_clone 噪声 降噪 音色",
                profile="voice_clone",
                top_k=2,
            )

        self.assertTrue(hits)
        self.assertEqual(hits[0].source, "audio_tools/deepfilternet.md")
        self.assertEqual(hits[0].tool_ids, ["repair.deepfilternet_denoise"])
        self.assertGreater(hits[0].score, 0.0)

    def test_context_builder_limits_length_and_preserves_references(self):
        from audio_dataset_agent.knowledge.context import build_knowledge_context
        from audio_dataset_agent.knowledge.models import KnowledgeHit

        hits = [
            KnowledgeHit(
                chunk_id="one",
                source="audio_tools/ffmpeg.md",
                title="FFmpeg",
                section="降噪",
                content="A" * 200,
                score=0.9,
                tool_ids=["repair.ffmpeg_denoise"],
                profiles=["tts"],
            )
        ]

        context = build_knowledge_context(hits, max_chars=80)

        self.assertLessEqual(len(context.text), 80)
        self.assertEqual(context.references[0]["source"], "audio_tools/ffmpeg.md")
        self.assertIn("chunk_id", context.text)

    def test_index_rebuild_removes_stale_document_chunks(self):
        from audio_dataset_agent.knowledge.retriever import LocalKnowledgeRetriever

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            documents = root / "documents"
            documents.mkdir()
            old_file = documents / "old.md"
            old_file.write_text("旧工具说明", encoding="utf-8")
            retriever = LocalKnowledgeRetriever(documents, root / "index.sqlite")
            retriever.rebuild()
            old_file.unlink()
            (documents / "new.md").write_text("新工具说明", encoding="utf-8")
            retriever.rebuild()
            hits = retriever.search("旧工具说明 新工具说明", top_k=10)

        self.assertTrue(hits)
        self.assertTrue(all(hit.source != "old.md" for hit in hits))

    def test_retriever_writes_audit_files_without_full_document_content(self):
        from audio_dataset_agent.knowledge.retriever import LocalKnowledgeRetriever

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            documents = root / "documents"
            documents.mkdir()
            (documents / "tool.md").write_text("# 工具\n用于测试检索。", encoding="utf-8")
            retriever = LocalKnowledgeRetriever(documents, root / "index.sqlite")
            retriever.search("测试检索", agent_name="strategy_agent", sample_id="sample_1")
            paths = retriever.write_audit_files(root / "export")
            retriever.write_audit_files(root / "export", append=True)
            query_text = Path(paths["rag_queries"]).read_text(encoding="utf-8")
            retrieval_text = Path(paths["rag_retrievals"]).read_text(encoding="utf-8")

        self.assertIn("strategy_agent", query_text)
        self.assertIn("sample_1", query_text)
        self.assertIn("chunk_id", retrieval_text)
        self.assertNotIn("用于测试检索", retrieval_text)

    def test_rebuild_option_can_refresh_an_existing_index(self):
        from audio_dataset_agent.knowledge.retriever import LocalKnowledgeRetriever

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            documents = root / "documents"
            documents.mkdir()
            path = documents / "tool.md"
            path.write_text("# 旧说明\n旧内容", encoding="utf-8")
            retriever = LocalKnowledgeRetriever(documents, root / "index.sqlite")
            path.write_text("# 新说明\n新内容", encoding="utf-8")
            count = retriever.rebuild()
            hits = retriever.search("新内容", top_k=1)

        self.assertEqual(count, 1)
        self.assertEqual(hits[0].section, "新说明")


class TestAgentKnowledgeIntegration(unittest.TestCase):
    def test_report_agent_exposes_local_knowledge_references(self):
        from audio_dataset_agent.core.models import Batch
        from audio_dataset_agent.knowledge.models import KnowledgeHit
        from audio_dataset_agent.reporting.agent import DatasetReportAgent

        class FixedRetriever:
            def search(self, query, **kwargs):
                return [KnowledgeHit(
                    "chunk", "audio_tools/silero_vad.md", "Silero VAD", "用途",
                    "用于语音活动检测。", 0.9, ["vad.silero"], ["tts"],
                )]

        batch = Batch(
            batch_id="batch_report_rag",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy="tts",
        )
        report = DatasetReportAgent(knowledge_retriever=FixedRetriever()).build(batch)

        self.assertEqual(report.knowledge_references[0]["source"], "audio_tools/silero_vad.md")

    def test_strategy_prompt_includes_local_knowledge_when_enabled(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.knowledge.models import KnowledgeHit
        from audio_dataset_agent.strategy.agent import LLMProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        class PromptClient:
            def complete(self, prompt):
                self.prompt = prompt
                return {
                    "action": "repair_audio",
                    "recommended_operation": "denoise_light",
                    "selected_tool_ids": ["repair.ffmpeg_denoise"],
                    "reason": "噪声偏高",
                    "confidence": 0.8,
                }

        class FixedRetriever:
            def search(self, query, **kwargs):
                return [KnowledgeHit(
                    "chunk", "audio_tools/ffmpeg.md", "FFmpeg", "降噪",
                    "轻量降噪可能产生伪影。", 0.8, ["repair.ffmpeg_denoise"], ["tts"],
                )]

        client = PromptClient()
        LLMProcessingStrategyAgent(
            client,
            default_tool_registry(),
            target_policy="tts",
            knowledge_retriever=FixedRetriever(),
        ).decide(AnalysisResult(
            "sample", 2.0, 10, 0.9, ["audio_noise_high"], {}
        ))

        self.assertIn("本地工具知识", client.prompt)
        self.assertIn("audio_tools/ffmpeg.md", client.prompt)

    def test_portfolio_context_contains_local_knowledge_without_credentials(self):
        from audio_dataset_agent.core.models import Batch
        from audio_dataset_agent.knowledge.models import KnowledgeHit
        from audio_dataset_agent.portfolio.context import build_portfolio_context

        class FixedRetriever:
            def search(self, query, **kwargs):
                return [KnowledgeHit(
                    "chunk", "audio_tools/ffmpeg.md", "FFmpeg", "本地工具",
                    "通过 FFmpeg 完成本地音频格式统一和导出。", 0.9, [], [],
                )]

        batch = Batch(
            batch_id="batch_portfolio_rag",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy="voice_clone",
        )
        payload = json.loads(build_portfolio_context(batch, FixedRetriever()))

        self.assertTrue(payload["local_knowledge"]["references"])
        self.assertNotIn("API_KEY", build_portfolio_context(batch, FixedRetriever()))


if __name__ == "__main__":
    unittest.main()
