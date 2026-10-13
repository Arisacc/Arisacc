import { useRef, useState } from 'react'
import { AudioLines, Bot, Database, FileText, FolderOpen, Gauge, HardDrive, Save, Settings2, UploadCloud, Wrench } from 'lucide-react'
import { API_BASE, policyLabels, TargetPolicy, usePipeline } from '../state/PipelineContext'

const profileNotes: Record<TargetPolicy, string> = {
  voice_clone: '重点检查音色稳定、噪声、静音和削波，文本不是硬性要求。',
  tts: '重点检查参考文本完整性和音频文本匹配关系。',
  asr: '重点检查语音可识别性、转写置信度和候选文本质量。',
}

export function SettingsSection({ onRun }: { onRun?: () => Promise<void> }) {
  const { form, setForm, updateTargetPolicy, updateNumberField, startJob, submitting, resetSummary, summary } = usePipeline()
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [directoryName, setDirectoryName] = useState('')
  const [directoryFileCount, setDirectoryFileCount] = useState(0)
  const [directoryMessage, setDirectoryMessage] = useState('选择后会将文件放入本地临时工作区，不会修改原始目录。')
  const [uploading, setUploading] = useState(false)

  const applyDirectoryFiles = async (files: FileList | File[]) => {
    const values = Array.from(files)
    if (!values.length) return
    const firstPath = values[0] instanceof File ? values[0].webkitRelativePath : ''
    const name = firstPath.split('/')[0] || values[0].name
    setDirectoryName(name)
    setDirectoryFileCount(values.length)
    setForm((current) => ({ ...current, input_dir: '' }))
    setUploading(true)
    setDirectoryMessage('正在把所选文件准备到本地处理工作区...')
    try {
      const body = new FormData()
      values.forEach((file) => body.append('files', file, file.webkitRelativePath || file.name))
      const response = await fetch(`${API_BASE}/api/upload-directory`, { method: 'POST', body })
      const payload = await response.json()
      if (!response.ok || payload.status !== 'success') throw new Error(payload.message || '目录准备失败')
      const staged = payload.data || payload
      setForm((current) => ({ ...current, input_dir: staged.input_dir }))
      setDirectoryName(staged.directory_name || name)
      setDirectoryFileCount(Number(staged.file_count || values.length))
      setDirectoryMessage('目录已准备，开始处理时将使用这批文件。')
      resetSummary()
    } catch (error) {
      setDirectoryMessage(error instanceof Error ? error.message : '目录准备失败，请重试。')
    } finally {
      setUploading(false)
    }
  }

  const chooseDirectory = () => {
    fileInputRef.current?.click()
  }

  const run = onRun || startJob
  return <>
    <section className="glassPanel settingsSection" data-reveal>
      <div className="panelHeader"><div><p className="sectionKicker">PROFILE</p><h2>数据用途</h2></div><Database size={16} /></div>
      <div className="profileCards">{(['voice_clone', 'tts', 'asr'] as TargetPolicy[]).map((policy) => <button type="button" className={`profileCard ${form.target_policy === policy ? 'selected' : ''}`} aria-pressed={form.target_policy === policy} onClick={() => updateTargetPolicy(policy)} key={policy}><span className="profileIcon"><AudioLines size={17} /></span><strong>{policyLabels[policy]}</strong><small>{profileNotes[policy]}</small></button>)}</div>
    </section>

    <div className="settingsGrid">
      <section className="glassPanel settingsSection" data-reveal>
        <div className="panelHeader"><div><p className="sectionKicker">AUDIO QUALITY</p><h2>分析参数</h2></div><Gauge size={16} /></div>
        <div className="rangeField"><div><label htmlFor="min-duration">最短片段时长</label><output>{form.vad_min_duration_seconds.toFixed(1)} 秒</output></div><input id="min-duration" type="range" min="0" max="5" step="0.1" value={form.vad_min_duration_seconds} onChange={(event) => updateNumberField('vad_min_duration_seconds', event.target.value)} disabled={form.vad !== 'wave-energy'} /><small>仅在能量检测 VAD 下生效。</small></div>
        <div className="rangeField"><div><label htmlFor="merge-gap">VAD 合并间隔</label><output>{form.vad_merge_gap_seconds.toFixed(2)} 秒</output></div><input id="merge-gap" type="range" min="0" max="2" step="0.05" value={form.vad_merge_gap_seconds} onChange={(event) => updateNumberField('vad_merge_gap_seconds', event.target.value)} disabled={form.vad !== 'wave-energy'} /><small>用于合并相邻语音片段。</small></div>
        <div className="rangeField"><div><label htmlFor="curator-threshold">Curator 置信度阈值</label><output>{form.curator_confidence_threshold.toFixed(2)}</output></div><input id="curator-threshold" type="range" min="0.5" max="0.95" step="0.05" value={form.curator_confidence_threshold} onChange={(event) => updateNumberField('curator_confidence_threshold', event.target.value)} /><small>置信度越高，进入人工复核的样本越少。</small></div>
        <label className="checkField"><input type="checkbox" checked={form.speaker_consistency_check} onChange={(event) => setForm({ ...form, speaker_consistency_check: event.target.checked })} /><span>启用批次声学一致性检查</span></label>
        <small className="fieldHint">默认关闭。开启后会比较 voice_clone 批次内的响度、静音和噪声差异，异常批次可能增加人工复核。</small>
        <div className="readonlyThresholds"><span>当前质量分界</span><b>SNR 下限建议从 15dB 起步；最终阈值仍由 DatasetProfile 和实际分布校准。</b></div>
      </section>

      <section className="glassPanel settingsSection" data-reveal>
        <div className="panelHeader"><div><p className="sectionKicker">ENGINES</p><h2>工具与模型</h2></div><Wrench size={16} /></div>
        <label className="formField"><span>VAD 引擎</span><select value={form.vad} onChange={(event) => setForm({ ...form, vad: event.target.value as typeof form.vad })}><option value="whole-file">整段音频</option><option value="wave-energy">能量检测</option><option value="fixed">固定测试切分</option></select></label>
        <label className="formField"><span>ASR 引擎</span><select value={form.asr} onChange={(event) => setForm({ ...form, asr: event.target.value as typeof form.asr })}><option value="none">不启用</option><option value="fixed">固定测试文本</option><option value="faster-whisper">faster-whisper</option></select></label>
        <label className="formField"><span>ASR 语言</span><input value="zh" readOnly aria-label="ASR 语言" /></label>
        <label className="formField"><span>流程</span><select value={form.workflow} onChange={(event) => setForm({ ...form, workflow: event.target.value as typeof form.workflow })}><option value="repair">含修复流程</option><option value="basic">基础流程</option></select></label>
        <p className="selectNote">{form.workflow === 'basic' ? '扫描 → 标准化 → VAD → 分析 → 决策 → 导出' : '在基础流程上增加 修复/复评 环节，自动 repair 低质量样本'}</p>
      </section>

      <section className="glassPanel settingsSection" data-reveal>
        <div className="panelHeader"><div><p className="sectionKicker">AGENTS</p><h2>Agent 执行器</h2></div><Bot size={16} /></div>
        <label className="formField"><span>Strategy Agent</span><select value={form.strategy_agent} onChange={(event) => setForm({ ...form, strategy_agent: event.target.value as typeof form.strategy_agent })}><option value="baseline">规则基线</option><option value="qwen-local">本地 Qwen</option></select></label>
        <label className="formField"><span>Curator Agent</span><select value={form.curator_agent} onChange={(event) => setForm({ ...form, curator_agent: event.target.value as typeof form.curator_agent })}><option value="baseline">规则基线</option><option value="qwen-local">本地 Qwen</option></select></label>
        <label className="formField"><span>Report Agent</span><select value={form.report_agent} onChange={(event) => setForm({ ...form, report_agent: event.target.value as typeof form.report_agent })}><option value="deterministic">确定性报告</option><option value="qwen-local">本地 Qwen</option></select></label>
        <label className="formField"><span>Qwen 模型路径</span><input value={form.qwen_model_path} onChange={(event) => setForm({ ...form, qwen_model_path: event.target.value })} /></label>
        <label className="formField"><span>输出长度</span><input type="number" min="32" max="1024" step="32" value={form.qwen_max_new_tokens} onChange={(event) => updateNumberField('qwen_max_new_tokens', event.target.value)} /></label>
      </section>

      <section className="glassPanel settingsSection" data-reveal>
        <div className="panelHeader"><div><p className="sectionKicker">SOURCE</p><h2>数据来源</h2></div><HardDrive size={16} /></div>
        <input ref={fileInputRef} className="visuallyHiddenInput" type="file" multiple onChange={(event) => { if (event.target.files) applyDirectoryFiles(event.target.files) }} {...({ webkitdirectory: '', directory: '' } as Record<string, string>)} />
        <div className={`directoryDropzone ${uploading ? 'uploading' : ''}`} role="button" tabIndex={0} onClick={chooseDirectory} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') chooseDirectory() }} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); void applyDirectoryFiles(event.dataTransfer.files) }}>
          <span className="directoryIcon"><FolderOpen size={19} /></span><strong>{directoryName || '选择本地音频目录'}</strong><small>{uploading ? '正在准备文件...' : directoryFileCount ? `${directoryFileCount} 个文件已准备` : '点击选择，或将目录拖到这里'}</small><UploadCloud size={15} className="dropHint" />
        </div>
        <p className="directoryMessage">{directoryMessage}</p>
        <div className="sourceSummaryWidget" aria-label="数据来源摘要">
          <div><span>已选择文件</span><strong>{directoryFileCount || '--'}</strong></div>
          <div><span>识别音频</span><strong>{summary ? summary.funnel[0]?.value ?? '--' : '运行后统计'}</strong></div>
          <div><span>当前工作区</span><code>{form.input_dir ? form.input_dir.split(/[\\/]/).pop() : '未准备'}</code></div>
        </div>
        <label className="formField"><span>元数据文件</span><input value={form.metadata_path} placeholder="可留空自动查找" onChange={(event) => setForm({ ...form, metadata_path: event.target.value })} /></label>
        <label className="checkField"><input type="checkbox" checked={form.use_sidecar_text} onChange={(event) => setForm({ ...form, use_sidecar_text: event.target.checked })} /><span>读取同名 TXT 参考文本</span></label>
        <label className="checkField"><input type="checkbox" checked={form.llm_only_issue_samples} onChange={(event) => setForm({ ...form, llm_only_issue_samples: event.target.checked })} /><span>仅问题样本调用 Qwen</span></label>
      </section>
    </div>
    <div className="settingsActions"><span><FileText size={14} /> 配置会随下一次运行请求发送</span><button className="primaryAction" type="button" onClick={run} disabled={submitting || uploading}><Save size={15} /> {uploading ? '准备文件中' : submitting ? '启动中' : '开始处理'}</button></div>
  </>
}

export function SettingsPage() {
  return <div className="pageContent settingsPage"><header className="pageHeading"><div><p className="sectionKicker">SETTINGS / 04</p><h2>处理配置</h2><p>设置数据用途、音频分析方式和本地 Agent 执行器。</p></div><span className="routeBadge"><Settings2 size={14} /> 当前配置</span></header><SettingsSection /></div>
}
