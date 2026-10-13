"""FastAPI Harness API。

接口路径与旧本地 HTTP 服务保持一致，前端无需切换数据协议。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from audio_dataset_agent.web.jobs import AudioDatasetJobManager, RunRequest
from audio_dataset_agent.web.local_server import (
    REPORT_FILE_NAMES,
    apply_manual_review_override,
    resolve_report_path,
    resolve_sample_audio_path,
)
from audio_dataset_agent.web.package_export import create_export_package


class RunRequestModel(BaseModel):
    """前端任务请求的边界模型，额外字段禁止进入 Harness。"""

    model_config = ConfigDict(extra="ignore")

    input_dir: str = Field(min_length=1)
    target_policy: str = "voice_clone"
    workflow: str = "basic"
    asr: str = "none"
    vad: str = "wave-energy"
    strategy_agent: str = "baseline"
    curator_agent: str = "baseline"
    report_agent: str = "deterministic"
    qwen_model_path: str = "local_tools/models/qwen/Qwen3-4B-Instruct-2507"
    qwen_max_new_tokens: int = Field(default=192, gt=0, le=4096)
    llm_only_issue_samples: bool = True
    curator_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    speaker_consistency_check: bool = False
    demo_transcript: str = "demo transcript"
    demo_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    vad_min_duration_seconds: float = Field(default=0.6, ge=0.0, le=60.0)
    vad_merge_gap_seconds: float = Field(default=0.45, ge=0.0, le=60.0)
    metadata_path: str = ""
    metadata_audio_column: str = "audio_path"
    metadata_text_column: str = "text"
    metadata_language_column: str = "language"
    metadata_speaker_column: str = "speaker_id"
    use_sidecar_text: bool = True
    repair_chain: str = "single"
    rag_mode: str = "none"
    rag_documents: str = "knowledge_base/documents"
    rag_index: str = "local_tools/cache/rag/index.sqlite"
    rag_rebuild: bool = False
    msst_enabled: bool = False
    msst_missing_policy: str = "error"
    msst_model_name: str = "deverb_bs_roformer_8_256dim_8depth"
    msst_model_path: str = "local_tools/models/msst/deverb_bs_roformer_8_256dim_8depth/deverb_bs_roformer_8_256dim_8depth.ckpt"
    msst_config_path: str = "local_tools/models/msst/deverb_bs_roformer_8_256dim_8depth/deverb_bs_roformer_8_256dim_8depth.ckpt.yaml"

    @model_validator(mode="after")
    def validate_profile_options(self):
        if self.target_policy != "voice_clone":
            raise ValueError("当前只支持 voice_clone 数据处理")
        if self.workflow not in {"basic", "repair"}:
            raise ValueError("workflow 不受支持")
        if self.asr not in {"none", "fixed", "faster-whisper"}:
            raise ValueError("asr 不受支持")
        if self.asr != "none":
            raise ValueError("当前 voice_clone 正式流程不启用 ASR")
        if self.vad not in {"fixed", "wave-energy", "whole-file"}:
            raise ValueError("vad 不受支持")
        if self.msst_missing_policy not in {"error", "skip"}:
            raise ValueError("MSST 缺失策略只能是 error 或 skip")
        if self.msst_model_name != "deverb_bs_roformer_8_256dim_8depth":
            raise ValueError("当前只支持 deverb_bs_roformer_8_256dim_8depth 模型")
        return self

    def to_domain(self) -> RunRequest:
        return RunRequest(**self.model_dump())


class ReviewOverrideModel(BaseModel):
    sample_id: str = Field(min_length=1)
    decision: str
    transcript: str = ""
    note: str = ""


def success(data: Any, message: str = "ok") -> dict[str, Any]:
    return {"status": "success", "message": message, "data": data}


def error(message: str) -> JSONResponse:
    return JSONResponse({"status": "error", "message": message}, status_code=400)


def create_app(
    runs_root: Path | str = "local_tools/cache/audio_web_runs",
    frontend_dir: Path | str | None = "frontend/dist",
) -> FastAPI:
    app = FastAPI(title="音频数据集 Agent Harness API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    manager = AudioDatasetJobManager(runs_root, queue_mode=True)
    app.state.manager = manager
    frontend_root = Path(frontend_dir).resolve() if frontend_dir else None

    @app.get("/hello")
    def hello():
        return success({"service": "audio_dataset_agent", "runtime": "fastapi-harness"})

    @app.get("/audio-dataset/jobs")
    def list_jobs():
        return success([job.to_dict() for job in manager.list_jobs()], "jobs loaded")

    @app.post("/audio-dataset/jobs", status_code=201)
    def start_job(payload: RunRequestModel):
        try:
            job = manager.start_job(payload.to_domain())
            return success(job.to_dict(), "任务已启动")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/audio-dataset/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return success(manager.get_job(job_id).to_dict(), "job loaded")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="未找到任务") from exc

    @app.get("/audio-dataset/jobs/{job_id}/summary")
    def get_summary(job_id: str):
        try:
            job = manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="未找到任务") from exc
        if job.summary is None:
            raise HTTPException(status_code=404, detail="摘要尚未生成")
        return success(job.summary, "summary loaded")

    @app.get("/audio-dataset/jobs/{job_id}/events")
    def get_events(job_id: str):
        try:
            manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="未找到任务") from exc
        return success(manager.store.list_events(job_id), "events loaded")

    @app.get("/api/v1/runs/{job_id}/task")
    def get_task(job_id: str):
        try:
            manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="未找到任务") from exc
        task = manager.store.get_task_for_job(job_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务尚未入队")
        return success(task, "task loaded")

    @app.get("/api/v1/runs/{job_id}/events/stream")
    async def stream_events(job_id: str, last_event_id: int = 0):
        try:
            manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="未找到任务") from exc

        async def event_generator():
            cursor = last_event_id
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline:
                events = manager.store.list_events_after(job_id, cursor)
                if events:
                    for event in events:
                        cursor = event["event_id"]
                        yield f"id: {cursor}\nevent: {event['event_type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    continue
                job = manager.get_job(job_id)
                if job.status in {"succeeded", "failed", "cancelled"}:
                    break
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/audio-dataset/jobs/{job_id}/artifacts")
    def get_artifacts(job_id: str):
        try:
            manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="未找到任务") from exc
        return success(manager.store.list_artifacts(job_id), "artifacts loaded")

    @app.get("/audio-dataset/jobs/{job_id}/export-package")
    def export_package(job_id: str):
        try:
            job = manager.get_job(job_id)
            if not job.batch_export_dir:
                raise ValueError("批次尚未导出")
            package = create_export_package(job.batch_export_dir)
            return FileResponse(package, filename="voice_clone_dataset_package.zip", media_type="application/zip")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="未找到任务") from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/audio-dataset/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            return success(manager.cancel_job(job_id).to_dict(), "已请求取消任务")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="未找到任务") from exc

    @app.post("/audio-dataset/jobs/{job_id}/review-overrides")
    def review_override(job_id: str, payload: ReviewOverrideModel):
        try:
            job = manager.get_job(job_id)
            result = apply_manual_review_override(job, payload.model_dump())
            # 复核结果和归档路径需要进入 SQLite，服务重启后仍能回读。
            manager.store.upsert_run(job.to_dict())
            manager.store.append_event(
                job_id,
                event_type="manual_review_applied",
                stage="review",
                status="succeeded",
                payload=result,
            )
            return success(result, "人工复核意见已保存")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="未找到任务") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/upload-directory", status_code=201)
    async def upload_directory(files: list[UploadFile] = File(...)):
        uploaded: list[tuple[str, bytes]] = []
        for upload in files:
            if not upload.filename:
                continue
            uploaded.append((upload.filename, await upload.read()))
        try:
            return success(manager.stage_uploaded_directory(uploaded), "目录已准备，可以开始处理")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/report")
    def report(job_id: str | None = None):
        jobs = manager.list_jobs()
        job = manager.get_job(job_id) if job_id else (jobs[0] if jobs else None)
        if job is None or job.summary is None:
            raise HTTPException(status_code=404, detail="摘要尚未生成")
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
        return success({"qualityScore": job.summary.get("qualityScore"), "samples": samples}, "report loaded")

    @app.get("/audio-dataset/jobs/{job_id}/reports/{report_name}")
    def report_file(job_id: str, report_name: str):
        try:
            path = resolve_report_path(manager.get_job(job_id), report_name)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type=mimetypes.guess_type(str(path))[0])

    @app.get("/audio-dataset/jobs/{job_id}/audio")
    def audio_file(job_id: str, sample_id: str):
        try:
            path = resolve_sample_audio_path(manager.get_job(job_id), sample_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type=mimetypes.guess_type(str(path))[0] or "audio/wav")

    if frontend_root and (frontend_root / "index.html").exists():
        @app.get("/{path:path}")
        def frontend(path: str):
            requested = (frontend_root / path).resolve()
            if requested.is_file() and requested.is_relative_to(frontend_root):
                return FileResponse(requested)
            return FileResponse(frontend_root / "index.html", media_type="text/html")

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动音频数据集 Agent FastAPI Harness 服务。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs-root", default="local_tools/cache/audio_web_runs")
    parser.add_argument("--frontend-dir", default="frontend/dist")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        create_app(args.runs_root, args.frontend_dir),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
