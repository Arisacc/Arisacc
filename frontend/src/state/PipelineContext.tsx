import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'

export type TargetPolicy = 'voice_clone' | 'tts' | 'asr'
export type WorkflowName = 'basic' | 'repair'
export type RepairChainMode = 'single' | 'auto'
export type AsrMode = 'none' | 'fixed' | 'faster-whisper'
export type VadMode = 'fixed' | 'wave-energy' | 'whole-file'
export type StrategyAgentMode = 'baseline' | 'qwen-local'
export type CuratorAgentMode = 'baseline' | 'qwen-local'
export type ReportAgentMode = 'deterministic' | 'qwen-local'
export type RagMode = 'none' | 'local'
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export type AudioRunRequest = {
  input_dir: string
  target_policy: TargetPolicy
  workflow: WorkflowName
  asr: AsrMode
  vad: VadMode
  strategy_agent: StrategyAgentMode
  curator_agent: CuratorAgentMode
  report_agent: ReportAgentMode
  qwen_model_path: string
  qwen_max_new_tokens: number
  llm_only_issue_samples: boolean
  curator_confidence_threshold: number
  speaker_consistency_check: boolean
  demo_transcript?: string
  demo_confidence?: number
  vad_min_duration_seconds: number
  vad_merge_gap_seconds: number
  metadata_path: string
  metadata_audio_column: string
  metadata_text_column: string
  metadata_language_column: string
  metadata_speaker_column: string
  use_sidecar_text: boolean
  repair_chain: RepairChainMode
  rag_mode: RagMode
  rag_documents: string
  rag_index: string
  rag_rebuild: boolean
}

export type JobProgress = {
  current_stage: string
  stage_message: string
  processed_samples: number
  total_samples: number
  percent: number
}

export type AudioJob = {
  job_id: string
  status: JobStatus
  request: AudioRunRequest
  work_dir: string | null
  export_dir: string | null
  batch_export_dir: string | null
  summary: AudioSummary | null
  error_message: string | null
  cancel_requested: boolean
  progress: JobProgress
}

export type PortfolioResearch = {
  status?: 'disabled' | 'running' | 'succeeded' | 'failed' | string
  message?: string
  problem_explanation?: string
  audio_processing_background?: Array<{ topic: string; summary: string; why_it_matters_for_this_project?: string }>
  tool_notes?: Array<{ tool_name: string; purpose: string; how_this_project_uses_it?: string; portfolio_value?: string }>
  external_references?: Array<{ title: string; url: string; used_for?: string }>
  local_references?: Array<{ source: string; section?: string; chunk_id?: string }>
  limitations?: string[]
}

export type AudioSample = {
  sample_id: string
  duration: number
  transcript?: string
  transcript_source: string
  reference_text?: string
  reference_text_source?: string
  language?: string
  speaker_id?: string
  metadata_status?: string
  text_alignment?: { cer?: number; wer?: number; similarity?: number }
  speaker_consistency?: { warning?: boolean; rms_range_db?: number; silence_range?: number; warning_reasons?: string[] }
  asr_confidence: number | null
  snr?: number | null
  quality_score?: number | null
  issue_tags: string[]
  strategy: string
  strategy_reason?: string
  curator_decision: string
  curator_reason?: string
  strategy_consistency?: string
  final_status: string
  final_audio_path?: string | null
  repair_chain_id?: string | null
  repair_chain?: Array<{ step_index: number; operation: string; parameters?: Record<string, unknown>; status?: string }>
}

export type AudioSummary = {
  qualityScore?: number | null
  batch: { batch_id: string; target_policy: string; total_samples: number; usable_samples: number; review_samples: number; rejected_samples: number; export_dir?: string }
  funnel: Array<{ label: string; value: number }>
  agents: Array<{ name: string; role: string; input: string; output: string; decision_power: string }>
  tools: Array<{ name: string; type: string; purpose: string; version?: string; status?: string }>
  strategy_collaboration?: Record<string, number>
  repair_preset?: { target_lufs: number; silence_threshold_dbfs: number; min_silence_duration_ms: number; noise_reduction_db: number }
  text_quality_summary?: Record<string, number>
  rag?: { enabled?: boolean; retrieval_count?: number; references?: Array<{ source: string; section?: string; score?: number }> }
  samples: AudioSample[]
  graph_trace: Array<{ node_name: string; agent_name: string; uses_agent: boolean; uses_tool: boolean; input_summary?: Record<string, unknown>; output_summary?: Record<string, unknown> }>
  portfolio_research: PortfolioResearch
  report_paths?: Record<string, string>
}

export const API_BASE = (import.meta.env.VITE_AUDIO_API_BASE || 'http://127.0.0.1:8000').replace(/\/$/, '')

export const defaultRequest: AudioRunRequest = {
  input_dir: '',
  target_policy: 'voice_clone',
  workflow: 'repair',
  asr: 'none',
  vad: 'whole-file',
  strategy_agent: 'baseline',
  curator_agent: 'baseline',
  report_agent: 'deterministic',
  qwen_model_path: 'local_tools/models/qwen/Qwen3-4B-Instruct-2507',
  qwen_max_new_tokens: 192,
  llm_only_issue_samples: true,
  curator_confidence_threshold: 0.8,
  speaker_consistency_check: false,
  demo_transcript: 'demo transcript',
  demo_confidence: 0.95,
  vad_min_duration_seconds: 0.6,
  vad_merge_gap_seconds: 0.45,
  metadata_path: '',
  metadata_audio_column: 'audio_path',
  metadata_text_column: 'text',
  metadata_language_column: 'language',
  metadata_speaker_column: 'speaker_id',
  use_sidecar_text: true,
  repair_chain: 'single',
  rag_mode: 'none',
  rag_documents: 'knowledge_base/documents',
  rag_index: 'local_tools/cache/rag/index.sqlite',
  rag_rebuild: false,
}

export const policyLabels: Record<TargetPolicy, string> = { voice_clone: '声音克隆', tts: '语音合成', asr: '语音识别' }
export const decisionLabels: Record<string, string> = { keep: '保留', reject: '拒绝', repair: '修复', review: '人工复核' }
export const strategyLabels: Record<string, string> = {
  keep: '直接保留', keep_without_processing: '直接保留', prepare_for_export: '准备导出',
  reject: '拒绝样本', reject_sample: '拒绝样本', reject_low_quality: '质量过低，已拒绝', exclude_sample: '排除样本', exclude_clipped_sample: '排除削波样本',
  repair: '进入修复', trim_silence: '裁剪静音', normalize_loudness: '响度标准化', denoise_light: '轻量降噪', neural_denoise: 'AI 降噪修复',
  manual_text_review: '人工复核', manual_decision_review: '人工复核', no_action: '无需处理',
}

export const pipelineStages = [
  { id: 'import', label: '导入', description: '扫描音频目录并匹配元数据。' },
  { id: 'normalize', label: '标准化', description: '使用 FFmpeg 生成统一格式的分析副本。' },
  { id: 'vad', label: 'VAD 切分', description: '检测语音区间并生成候选样本。' },
  { id: 'asr', label: '可选 ASR', description: '按用途生成候选文本和置信度。' },
  { id: 'inspect', label: '质量分析', description: '整理音频、文本和批次一致性证据。' },
  { id: 'decide', label: 'Agent 决策', description: 'Strategy 提议，Curator 审核并判断去留。' },
  { id: 'repair', label: '修复复评', description: '执行受控工具链并重新分析结果。' },
  { id: 'export', label: '导出追溯', description: '生成 manifest、报告和 Agent Graph 记录。' },
] as const

export async function requestData<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(API_BASE + path, init)
  const payload = await response.json()
  if (!response.ok || payload.status !== 'success') throw new Error(payload.message || payload.detail || '请求失败')
  return payload.data as T
}

export function displayStrategy(value: string): string {
  return strategyLabels[value] || value.replace(/_/g, ' ')
}

export function formatNumber(value: number | undefined | null): string {
  return typeof value === 'number' && !Number.isNaN(value) ? value.toLocaleString('zh-CN') : '0'
}

export function jobReportUrl(job: AudioJob | null, reportName: string): string {
  return job ? `${API_BASE}/audio-dataset/jobs/${job.job_id}/reports/${reportName}` : ''
}

export function sampleAudioUrl(job: AudioJob | null, sampleId: string): string {
  return job ? `${API_BASE}/audio-dataset/jobs/${job.job_id}/audio?sample_id=${encodeURIComponent(sampleId)}` : ''
}

export function jobStatusLabel(status: JobStatus | undefined): string {
  return status === 'running' ? '处理中' : status === 'queued' ? '排队中' : status === 'succeeded' ? '已完成' : status === 'failed' ? '失败' : status === 'cancelled' ? '已取消' : '演示数据'
}

export function stageProgressIndex(progress: JobProgress | undefined, status: JobStatus | undefined): number {
  if (status === 'succeeded') return pipelineStages.length - 1
  if (!progress) return -1
  return Math.min(pipelineStages.length - 1, Math.max(0, Math.floor((progress.percent / 100) * pipelineStages.length)))
}

type PipelineContextValue = {
  form: AudioRunRequest
  setForm: React.Dispatch<React.SetStateAction<AudioRunRequest>>
  updateTargetPolicy: (targetPolicy: TargetPolicy) => void
  updateNumberField: (key: 'vad_min_duration_seconds' | 'vad_merge_gap_seconds' | 'qwen_max_new_tokens' | 'curator_confidence_threshold', value: string) => void
  job: AudioJob | null
  summary: AudioSummary | null
  message: string
  backendStatus: 'checking' | 'connected' | 'offline'
  submitting: boolean
  startJob: () => Promise<void>
  cancelJob: () => Promise<void>
  saveReviewOverride: (sample: AudioSample, decision: string, transcript: string, note: string) => Promise<void>
  resetSummary: () => void
}

const PipelineContext = createContext<PipelineContextValue | null>(null)

export function PipelineProvider({ children }: { children: React.ReactNode }) {
  const [form, setForm] = useState(defaultRequest)
  const [job, setJob] = useState<AudioJob | null>(null)
  const [summary, setSummary] = useState<AudioSummary | null>(null)
  const [message, setMessage] = useState('请输入本地音频目录并启动处理任务。')
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'offline'>('checking')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    fetch(`${API_BASE}/hello`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error('后端响应异常')
        setBackendStatus('connected')
      })
      .catch(() => {
        setBackendStatus('offline')
        setMessage('音频处理后端未连接，请先启动 start_backend.bat。')
      })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return
    const timer = window.setInterval(async () => {
      try {
        const nextJob = await requestData<AudioJob>(`/audio-dataset/jobs/${job.job_id}`)
        setJob(nextJob)
        if (nextJob.summary) setSummary(nextJob.summary)
        if (nextJob.status === 'succeeded') setMessage('处理完成，结果已更新。')
        if (nextJob.status === 'failed') setMessage(nextJob.error_message || '处理失败。')
        if (nextJob.status === 'cancelled') setMessage('任务已取消。')
      } catch (error) {
        setMessage(error instanceof Error ? error.message : '查询任务失败。')
      }
    }, 1500)
    return () => window.clearInterval(timer)
  }, [job])

  useEffect(() => {
    if (!job?.job_id || job.status !== 'succeeded') return
    fetch(`${API_BASE}/api/report?job_id=${encodeURIComponent(job.job_id)}`)
      .then((response) => response.ok ? response.json() : null)
      .then((payload) => {
        const report = payload?.data || payload
        if (!report) return
        setSummary((current) => {
          if (!current) return current
          const snrById = new Map<string, number | null>((report.samples || []).map((sample: { id: string; snr?: unknown }) => [sample.id, typeof sample.snr === 'number' ? sample.snr : null] as [string, number | null]))
          const qualityById = new Map<string, number | null>((report.samples || []).map((sample: { id: string; qualityScore?: unknown }) => [sample.id, typeof sample.qualityScore === 'number' ? sample.qualityScore : null] as [string, number | null]))
          return {
            ...current,
            qualityScore: report.qualityScore ?? null,
            samples: current.samples.map((sample) => snrById.has(sample.sample_id) ? { ...sample, snr: snrById.get(sample.sample_id) ?? null, quality_score: qualityById.get(sample.sample_id) ?? null } : sample),
          }
        })
      })
      .catch(() => undefined)
  }, [job?.job_id, job?.status])

  const updateTargetPolicy = (targetPolicy: TargetPolicy) => setForm((current) => ({ ...current, target_policy: targetPolicy, asr: targetPolicy === 'voice_clone' ? current.asr : current.asr === 'none' ? 'fixed' : current.asr }))
  const updateNumberField = (key: 'vad_min_duration_seconds' | 'vad_merge_gap_seconds' | 'qwen_max_new_tokens' | 'curator_confidence_threshold', value: string) => {
    const parsed = Number(value)
    setForm((current) => ({ ...current, [key]: Number.isFinite(parsed) ? parsed : current[key] }))
  }

  const startJob = async () => {
    setSubmitting(true)
    setSummary(null)
    setMessage('正在启动本地处理任务...')
    try {
      const nextJob = await requestData<AudioJob>('/audio-dataset/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) })
      setJob(nextJob)
      setMessage('任务已启动，正在等待后端处理。')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '启动失败。')
    } finally {
      setSubmitting(false)
    }
  }

  const cancelJob = async () => {
    if (!job || !['queued', 'running'].includes(job.status)) return
    try {
      setJob(await requestData<AudioJob>(`/audio-dataset/jobs/${job.job_id}/cancel`, { method: 'POST' }))
      setMessage('已发送取消请求。')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '取消失败。')
    }
  }

  const saveReviewOverride = async (sample: AudioSample, decision: string, transcript: string, note: string) => {
    if (!job) return
    await requestData(`/audio-dataset/jobs/${job.job_id}/review-overrides`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sample_id: sample.sample_id, decision, transcript, note }) })
    setMessage('人工复核意见已保存。')
  }

  const resetSummary = () => setSummary(null)

  const value = useMemo(() => ({ form, setForm, updateTargetPolicy, updateNumberField, job, summary, message, backendStatus, submitting, startJob, cancelJob, saveReviewOverride, resetSummary }), [form, job, summary, message, backendStatus, submitting])
  return <PipelineContext.Provider value={value}>{children}</PipelineContext.Provider>
}

export function usePipeline() {
  const context = useContext(PipelineContext)
  if (!context) throw new Error('usePipeline 必须在 PipelineProvider 内使用')
  return context
}
