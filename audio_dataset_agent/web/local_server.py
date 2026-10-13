from __future__ import annotations
"""本地 Web API 入口。

前端只和这个轻量 HTTP 服务通信：POST 创建处理任务，GET 轮询 job 状态，
成功后再读取 summary、音频文件和报告文件。文件返回前会校验路径必须属于当前 job，
避免前端任意读取本机文件。
"""

import argparse
import csv
import json
import math
import mimetypes
import re
import shutil
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from audio_dataset_agent.web.jobs import AudioDatasetJobManager, RunRequest
from audio_dataset_agent.web.summary import build_summary_from_export_dir, write_summary_file


def normalize_request_path(raw_path: str) -> str:
    path = urlparse(raw_path).path or "/"
    while path.startswith("//"):
        path = path[1:]
    return path


def resolve_frontend_path(frontend_root: Path | str, raw_path: str) -> Path:
    """解析前端静态文件；未知前端路由回退到 index.html，并阻止路径逃逸。"""
    root = Path(frontend_root).resolve()
    requested = unquote(urlparse(raw_path).path or "/").lstrip("/")
    candidate = (root / requested).resolve()
    if candidate.is_file() and _is_relative_to(candidate, root):
        return candidate
    index_path = (root / "index.html").resolve()
    if not index_path.is_file():
        raise ValueError("前端构建产物不存在，请先执行 frontend\\npm run build")
    return index_path


def build_success_payload(data, message: str = "ok") -> dict:
    return {"status": "success", "message": message, "data": data}


def build_error_payload(message: str) -> dict:
    return {"status": "error", "message": message}


def sanitize_json_value(value):
    """HTTP JSON 不允许 NaN/Infinity，这里递归清理音频指标里的非法浮点值。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    return value


def dumps_strict_json(payload: dict) -> str:
    return json.dumps(sanitize_json_value(payload), ensure_ascii=False, allow_nan=False)


REPORT_FILE_NAMES = {
    "batch_report": "batch_report.md",
    "dataset_quality_report": "dataset_quality_report.md",
    "portfolio_report_html": "portfolio_report.html",
    "portfolio_report_markdown": "portfolio_report.md",
}


def resolve_sample_audio_path(job, sample_id: str) -> Path:
    """音频播放接口只允许读取当前 job summary 中出现过的 final_audio_path。"""
    if not job.summary:
        raise ValueError("摘要尚未生成")
    for sample in job.summary.get("samples", []):
        if sample.get("sample_id") != sample_id:
            continue
        audio_path = sample.get("final_audio_path")
        if not audio_path:
            raise ValueError("该样本没有可播放的音频")
        return _resolve_job_artifact_path(job, audio_path)
    raise ValueError("未找到样本")


def resolve_report_path(job, report_name: str) -> Path:
    if report_name not in REPORT_FILE_NAMES:
        raise ValueError("不支持的报告类型")
    if not job.summary:
        raise ValueError("摘要尚未生成")
    report_paths = job.summary.get("report_paths", {})
    report_path = report_paths.get(report_name)
    if not report_path:
        if not job.batch_export_dir:
            raise ValueError("批次导出目录尚未生成")
        report_path = str(Path(job.batch_export_dir) / REPORT_FILE_NAMES[report_name])
    return _resolve_job_artifact_path(job, report_path)


def apply_manual_review_override(job, payload: dict) -> dict:
    """保存人工复核覆盖意见，并导出 manual_review_manifest.jsonl。

    这一步不回写原始 manifest，也不改变已经跑完的 Agent 决策；它生成一份带人工覆盖标记的
    修订 manifest，适合后续人工确认后再进入训练数据准备。
    """
    if not job.batch_export_dir:
        raise ValueError("批次导出目录尚未生成")
    batch_dir = Path(job.batch_export_dir)
    if not batch_dir.exists():
        raise ValueError("未找到批次导出目录")

    sample_id = str(payload.get("sample_id") or "").strip()
    decision = str(payload.get("decision") or "").strip()
    transcript = str(payload.get("transcript") or "")
    note = str(payload.get("note") or "").strip()
    if not sample_id:
        raise ValueError("sample_id 不能为空")
    if decision not in {"keep", "reject"}:
        raise ValueError("人工复核决策只能是 keep 或 reject")

    manifest_path = batch_dir / "manifest.jsonl"
    records = _read_jsonl(manifest_path)
    if not any(record.get("sample_id") == sample_id for record in records):
        raise ValueError("manifest 中未找到该样本")

    override = {
        "sample_id": sample_id,
        "decision": decision,
        "transcript": transcript,
        "note": note,
        "source": "manual_review",
    }
    overrides_path = batch_dir / "manual_review_overrides.json"
    overrides = _read_json(overrides_path, default={})
    if not isinstance(overrides, dict):
        overrides = {}
    overrides[sample_id] = override

    archive_group = decision
    archive_root = batch_dir / "manual_groups" / archive_group
    archive_root.mkdir(parents=True, exist_ok=True)
    archived_path: Path | None = None
    archive_name: str | None = None
    corrected_records = []
    for record in records:
        if record.get("sample_id") == sample_id:
            record = dict(record)
            source_value = record.get("source_audio_path") or record.get("audio_path") or record.get("final_audio_path")
            source_path = Path(source_value) if source_value else None
            if source_path and source_path.exists() and source_path.is_file():
                safe_sample_id = re.sub(r"[^A-Za-z0-9._-]", "_", sample_id).strip("._") or "sample"
                archive_name = f"{safe_sample_id}{source_path.suffix or '.wav'}"
                # 同一样本重新改判时删除旧人工分组副本，保证它只属于当前分组。
                previous_group_root = batch_dir / "manual_groups"
                for old_group in ("keep", "reject", "repair", "review"):
                    old_path = previous_group_root / old_group / archive_name
                    if old_path.exists() and old_group != archive_group:
                        old_path.unlink()
                archived_path = archive_root / archive_name
                shutil.copy2(source_path, archived_path)
            record["decision"] = decision
            record["decision_reason"] = f"manual override: {note}" if note else "manual override"
            if transcript:
                record["text"] = transcript
                record["transcript_source"] = "manual_review"
                record["reference_text"] = transcript
                record["reference_text_source"] = "manual_review"
                record["metadata_status"] = "matched"
            record["final_status"] = _manual_final_status(decision)
            if archived_path is not None:
                record["final_audio_path"] = str(archived_path)
                record["audio_path"] = str(archived_path)
                record["manual_archive_group"] = archive_group
                record["manual_archive_path"] = str(archived_path)
            record["manual_review"] = override
        corrected_records.append(record)

    override["archive_group"] = archive_group
    override["archive_path"] = str(archived_path) if archived_path else None
    override["msst"] = (job.summary or {}).get("run_config", {}).get("msst", {})
    overrides[sample_id] = override
    overrides_path.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")

    corrected_path = batch_dir / "manual_review_manifest.jsonl"
    corrected_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in corrected_records) + "\n",
        encoding="utf-8",
    )

    # 复核覆盖不重新跑音频处理，但更新当前进程中的 summary，
    # 让决策页立即显示新的状态和归档音频路径。
    write_summary_file(batch_dir)
    _refresh_review_queue(batch_dir, corrected_records)
    if job.summary is not None:
        job.summary = build_summary_from_export_dir(batch_dir)
        job.summary["review_overrides"] = overrides
    return {
        "sample_id": sample_id,
        "decision": decision,
        "overrides_path": str(overrides_path),
        "manual_manifest_path": str(corrected_path),
        "archive_group": archive_group,
        "archive_path": str(archived_path) if archived_path else None,
    }


def _manual_final_status(decision: str) -> str:
    return {
        "keep": "manual_keep",
        "reject": "manual_reject",
        "repair": "manual_repair",
        "review": "manual_review",
    }[decision]


def _refresh_review_queue(batch_dir: Path, records: list[dict]) -> None:
    """人工去留改变后同步复核队列，避免已保留样本仍显示为待复核。"""
    queue = []
    for record in records:
        if record.get("final_status") not in {"needs_review", "manual_review"}:
            continue
        queue.append({
            "sample_id": record.get("sample_id"),
            "recommended_action": record.get("recommended_action") or record.get("decision"),
            "issue_tags": record.get("issue_tags") or [],
            "confidence": record.get("confidence"),
            "reason": record.get("decision_reason") or "",
        })
    (batch_dir / "review_queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (batch_dir / "review_queue.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["sample_id", "recommended_action", "issue_tags", "confidence", "reason"])
        writer.writeheader()
        for item in queue:
            writer.writerow({**item, "issue_tags": ";".join(str(tag) for tag in item["issue_tags"])})


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_job_artifact_path(job, raw_path: str) -> Path:
    """所有报告、音频文件都经过路径检查，确保文件位于当前 job 的工作区或导出区。"""
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError("未找到产物文件")

    allowed_roots = []
    for raw_root in [job.work_dir, job.export_dir, job.batch_export_dir]:
        if raw_root:
            root = Path(raw_root)
            if not root.is_absolute():
                root = Path.cwd() / root
            allowed_roots.append(root.resolve())

    if not any(_is_relative_to(resolved, allowed_root) for allowed_root in allowed_roots):
        raise ValueError("产物路径不属于当前任务")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class AudioDatasetRequestHandler(BaseHTTPRequestHandler):
    manager: AudioDatasetJobManager
    frontend_root: Path | None = None

    def do_OPTIONS(self):
        self._send_json(build_success_payload({}, "ok"))

    def do_GET(self):
        path = normalize_request_path(self.path)
        try:
            if path == "/api/report":
                query = parse_qs(urlparse(self.path).query)
                requested_job_id = (query.get("job_id") or [""])[0]
                if requested_job_id:
                    job = self.manager.get_job(requested_job_id)
                else:
                    jobs = self.manager.list_jobs()
                    if not jobs:
                        self._send_json(build_error_payload("暂无处理任务"), status=404)
                        return
                    job = jobs[0]
                if not job.summary:
                    self._send_json(build_error_payload("摘要尚未生成"), status=404)
                    return
                samples = [
                    {
                        "id": sample.get("sample_id"),
                        "snr": sample.get("snr"),
                        "qualityScore": sample.get("quality_score"),
                        "duration": sample.get("duration"),
                        "finalStatus": sample.get("final_status"),
                    }
                    for sample in job.summary.get("samples", [])
                ]
                self._send_json(build_success_payload({
                    "qualityScore": job.summary.get("qualityScore"),
                    "samples": samples,
                }, "report loaded"))
                return
            if path == "/hello":
                self._send_json(build_success_payload({"service": "audio_dataset_agent"}, "ok"))
                return
            if path == "/audio-dataset/jobs":
                jobs = [job.to_dict() for job in self.manager.list_jobs()]
                self._send_json(build_success_payload(jobs, "jobs loaded"))
                return
            if path.startswith("/audio-dataset/jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 3:
                    job = self.manager.get_job(parts[2])
                    self._send_json(build_success_payload(job.to_dict(), "job loaded"))
                    return
                if len(parts) == 4 and parts[3] == "summary":
                    job = self.manager.get_job(parts[2])
                    if job.summary is None:
                        self._send_json(build_error_payload("摘要尚未生成"), status=404)
                        return
                    self._send_json(build_success_payload(job.summary, "summary loaded"))
                    return
                if len(parts) == 4 and parts[3] == "audio":
                    job = self.manager.get_job(parts[2])
                    query = parse_qs(urlparse(self.path).query)
                    sample_id = (query.get("sample_id") or [""])[0]
                    audio_path = resolve_sample_audio_path(job, sample_id)
                    self._send_file(audio_path, "audio/wav")
                    return
                if len(parts) == 5 and parts[3] == "reports":
                    job = self.manager.get_job(parts[2])
                    report_path = resolve_report_path(job, parts[4])
                    self._send_file(report_path)
                    return
            if path == "/api" or path.startswith("/api/") or path.startswith("/audio-dataset/"):
                self._send_json(build_error_payload(f"未知接口路径：{path}"), status=404)
                return
            if self.frontend_root is not None:
                self._send_file(resolve_frontend_path(self.frontend_root, self.path))
                return
            self._send_json(build_error_payload(f"未知接口路径：{path}"), status=404)
        except KeyError:
            self._send_json(build_error_payload("未找到任务"), status=404)
        except Exception as exc:
            self._send_json(build_error_payload(str(exc)), status=500)

    def do_POST(self):
        path = normalize_request_path(self.path)
        try:
            if path == "/api/upload-directory":
                files = self._read_multipart_files()
                staged = self.manager.stage_uploaded_directory(files)
                self._send_json(build_success_payload(staged, "目录已准备，可以开始处理"), status=201)
                return
            if path == "/audio-dataset/jobs":
                payload = self._read_json_body()
                request = RunRequest(
                    input_dir=str(payload.get("input_dir", "")),
                    target_policy=payload.get("target_policy", "voice_clone"),
                    workflow=payload.get("workflow", "basic"),
                    asr=payload.get("asr", "none"),
                    vad=payload.get("vad", "wave-energy"),
                    strategy_agent=payload.get("strategy_agent", "baseline"),
                    curator_agent=payload.get("curator_agent", "baseline"),
                    report_agent=payload.get("report_agent", "deterministic"),
                    qwen_model_path=payload.get("qwen_model_path", "local_tools/models/qwen/Qwen3-4B-Instruct-2507"),
                    qwen_max_new_tokens=int(payload.get("qwen_max_new_tokens", 192)),
                    llm_only_issue_samples=bool(payload.get("llm_only_issue_samples", True)),
                    curator_confidence_threshold=float(payload.get("curator_confidence_threshold", 0.8)),
                    speaker_consistency_check=bool(payload.get("speaker_consistency_check", False)),
                    demo_transcript=payload.get("demo_transcript", "demo transcript"),
                    demo_confidence=float(payload.get("demo_confidence", 0.95)),
                    vad_min_duration_seconds=float(payload.get("vad_min_duration_seconds", 0.6)),
                    vad_merge_gap_seconds=float(payload.get("vad_merge_gap_seconds", 0.45)),
                    metadata_path=str(payload.get("metadata_path", "") or ""),
                    metadata_audio_column=str(payload.get("metadata_audio_column", "audio_path")),
                    metadata_text_column=str(payload.get("metadata_text_column", "text")),
                    metadata_language_column=str(payload.get("metadata_language_column", "language")),
                    metadata_speaker_column=str(payload.get("metadata_speaker_column", "speaker_id")),
                    use_sidecar_text=bool(payload.get("use_sidecar_text", True)),
                    repair_chain=str(payload.get("repair_chain", "single")),
                    rag_mode=str(payload.get("rag_mode", "none")),
                    rag_documents=str(payload.get("rag_documents", "knowledge_base/documents")),
                    rag_index=str(payload.get("rag_index", "local_tools/cache/rag/index.sqlite")),
                    rag_rebuild=bool(payload.get("rag_rebuild", False)),
                )
                job = self.manager.start_job(request)
                self._send_json(build_success_payload(job.to_dict(), "任务已启动"), status=201)
                return

            if path.startswith("/audio-dataset/jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] == "cancel":
                    job = self.manager.cancel_job(parts[2])
                    self._send_json(build_success_payload(job.to_dict(), "已请求取消任务"))
                    return
                if len(parts) == 4 and parts[3] == "review-overrides":
                    job = self.manager.get_job(parts[2])
                    result = apply_manual_review_override(job, self._read_json_body())
                    self.manager.store.upsert_run(job.to_dict())
                    self.manager.store.append_event(
                        parts[2], "manual_review_applied", "review", "succeeded", result
                    )
                    self._send_json(build_success_payload(result, "人工复核意见已保存"))
                    return

            self._send_json(build_error_payload(f"未知接口路径：{path}"), status=404)
        except KeyError:
            self._send_json(build_error_payload("未找到任务"), status=404)
        except ValueError as exc:
            self._send_json(build_error_payload(str(exc)), status=400)
        except Exception as exc:
            self._send_json(build_error_payload(str(exc)), status=500)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def _read_multipart_files(self) -> list[tuple[str, bytes]]:
        """解析浏览器 FormData 中的目录文件。"""
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise ValueError("目录上传必须使用 multipart/form-data")
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("上传内容为空")
        body = self.rfile.read(length)
        header = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8")
        message = BytesParser(policy=policy.default).parsebytes(header + body)
        files: list[tuple[str, bytes]] = []
        for part in message.walk():
            if part.is_multipart():
                continue
            filename = part.get_filename()
            if not filename:
                continue
            payload = part.get_payload(decode=True)
            if payload is not None:
                files.append((filename, payload))
        return files

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = dumps_strict_json(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        data = path.read_bytes()
        guessed_type = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", guessed_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        print("[audio-web] " + format % args)


def create_server(
    host: str,
    port: int,
    runs_root: Path | str,
    frontend_root: Path | str | None = None,
) -> ThreadingHTTPServer:
    manager = AudioDatasetJobManager(runs_root)

    class Handler(AudioDatasetRequestHandler):
        pass

    Handler.manager = manager
    Handler.frontend_root = Path(frontend_root).resolve() if frontend_root else None
    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动音频数据集本地 Web API。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs-root", default="local_tools/cache/audio_web_runs")
    parser.add_argument("--frontend-dir", default="frontend/dist")
    args = parser.parse_args(argv)

    frontend_dir = Path(args.frontend_dir)
    if not frontend_dir.is_absolute():
        frontend_dir = Path.cwd() / frontend_dir
    server = create_server(args.host, args.port, args.runs_root, frontend_root=frontend_dir)
    print(f"音频数据集 Agent API 已启动：http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Audio Dataset Agent API...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
