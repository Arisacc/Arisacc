# Agent Graph 记录层：描述每个流程节点是否使用 Agent 或工具。
import json
import time
from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from audio_dataset_agent.core.models import Batch


@dataclass(frozen=True)
class AgentGraphNodeTrace:
    node_name: str
    agent_name: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    duration_ms: float
    uses_agent: bool = False
    uses_tool: bool = False
    notes: str = ""


@dataclass
class AgentGraphRecorder:
    graph_name: str = "音频数据集 Agent 工作流"
    nodes: list[AgentGraphNodeTrace] = field(default_factory=list)
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AgentGraphRecorder":
        graph = cls(graph_name=(payload or {}).get("graph_name", "音频数据集 Agent 工作流"))
        for item in (payload or {}).get("nodes", []):
            if not isinstance(item, dict):
                continue
            graph.nodes.append(AgentGraphNodeTrace(
                node_name=item["node_name"], agent_name=item["agent_name"],
                input_summary=dict(item.get("input_summary") or {}),
                output_summary=dict(item.get("output_summary") or {}),
                duration_ms=float(item.get("duration_ms", 0)),
                uses_agent=bool(item.get("uses_agent", False)), uses_tool=bool(item.get("uses_tool", False)),
                notes=item.get("notes", ""),
            ))
        return graph

    def run_node(
        self,
        node_name: str,
        agent_name: str,
        batch: Batch,
        runner: Callable[[Batch], Batch],
        uses_agent: bool = False,
        uses_tool: bool = False,
        notes: str = "",
    ) -> Batch:
        started = time.perf_counter()
        if self.progress_callback:
            self.progress_callback({"event": "stage_started", "stage": node_name, "agent": agent_name, "message": f"开始{agent_name}", "percent": 0})
        output = runner(batch)
        duration_ms = (time.perf_counter() - started) * 1000
        self.nodes.append(
            AgentGraphNodeTrace(
                node_name=node_name,
                agent_name=agent_name,
                input_summary=batch_summary(batch),
                output_summary=batch_summary(output),
                duration_ms=round(duration_ms, 3),
                uses_agent=uses_agent,
                uses_tool=uses_tool,
                notes=notes,
            )
        )
        if self.progress_callback:
            self.progress_callback({"event": "stage_completed", "stage": node_name, "agent": agent_name, "message": f"完成{agent_name}", "percent": 100, "duration_ms": round(duration_ms, 3)})
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "nodes": [node_trace_to_dict(node) for node in self.nodes],
        }


def node_trace_to_dict(node: AgentGraphNodeTrace) -> dict[str, Any]:
    return {
        "node_name": node.node_name,
        "agent_name": node.agent_name,
        "input_summary": dict(node.input_summary),
        "output_summary": dict(node.output_summary),
        "duration_ms": node.duration_ms,
        "uses_agent": node.uses_agent,
        "uses_tool": node.uses_tool,
        "notes": node.notes,
    }


def attach_agent_graph_trace(
    batch: Batch,
    recorder: AgentGraphRecorder,
) -> Batch:
    report_path = batch.export_paths.get("report") or batch.export_paths.get(
        "dataset_quality_json"
    )
    if not report_path:
        return batch

    trace_path = Path(report_path).parent / "agent_graph_trace.json"
    trace_path.write_text(
        json.dumps(recorder.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_paths = dict(batch.export_paths)
    export_paths["agent_graph_trace"] = str(trace_path)
    return replace(batch, export_paths=export_paths)


def batch_summary(batch: Batch) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "phase": batch.phase.value,
        "raw_assets": batch.raw_asset_count,
        "candidate_samples": batch.candidate_sample_count,
        "analysis_results": batch.analysis_result_count,
        "processing_strategies": batch.processing_strategy_count,
        "curator_decisions": batch.curator_decision_count,
        "repair_plans": batch.repair_plan_count,
        "repair_executions": batch.repair_execution_count,
        "repair_evaluations": batch.repair_evaluation_count,
        "final_selections": batch.final_selection_count,
        "exported": bool(batch.export_paths),
    }
