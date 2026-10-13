import { Activity, AlertTriangle, CheckCircle2, Cpu, Database, FileAudio, GitBranch, Info, Wrench } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { PipelineStepper } from '../components/PipelineStepper'
import { SettingsSection } from './SettingsPage'
import { AudioSummary, formatNumber, jobStatusLabel, pipelineStages, stageProgressIndex, usePipeline } from '../state/PipelineContext'

function revealClass() { return 'glassPanel' }

function InfoHint({ text }: { text: string }) {
  return <span className="infoHint" title={text} aria-label={text}><Info size={13} /></span>
}

function qualityBand(score: number | null | undefined): 'A' | 'B' | 'C' | 'unknown' {
  if (typeof score !== 'number') return 'unknown'
  if (score >= 0.9) return 'A'
  if (score >= 0.75) return 'B'
  return 'C'
}

function QualityDistribution({ summary }: { summary: AudioSummary }) {
  const counts = summary.samples.reduce<Record<string, number>>((result, sample) => {
    const band = qualityBand(sample.quality_score)
    result[band] = (result[band] || 0) + 1
    return result
  }, {})
  const rows = [
    { key: 'A', label: '优秀', className: 'qualityA' },
    { key: 'B', label: '良好', className: 'qualityB' },
    { key: 'C', label: '待复核', className: 'qualityC' },
    { key: 'unknown', label: '未计算', className: 'qualityUnknown' },
  ]
  const max = Math.max(...rows.map((row) => counts[row.key] || 0), 1)
  const scoredCount = (counts.A || 0) + (counts.B || 0) + (counts.C || 0)
  return <section className="glassPanel insightPanel" data-reveal>
    <div className="panelHeader"><div><p className="sectionKicker">QUALITY DISTRIBUTION</p><h2>质量分布</h2></div><span className="panelMeta">{scoredCount} / {summary.samples.length} 条有质量分</span></div>
    <div className="qualityBars">{rows.map((row) => { const count = counts[row.key] || 0; return <div className="qualityBarRow" key={row.key}><span className={`qualityBadge ${row.className}`}>{row.key === 'unknown' ? '--' : row.key}</span><span className="qualityBarLabel">{row.label}</span><div className="qualityBarTrack"><span className={`qualityBarFill ${row.className}`} style={{ width: `${(count / max) * 100}%` }} /></div><strong>{count}</strong></div> })}</div>
    <p className="widgetNote"><Info size={13} /> 质量分为基于音频指标的规则估算值，不代表模型准确率。</p>
  </section>
}

const issueLabels: Record<string, string> = {
  speaker_consistency_warning: '批次声学一致性',
  audio_high_silence_ratio: '静音比例过高',
  audio_noise_high: '噪声偏高',
  audio_clipping: '音频削波',
  audio_too_quiet: '音量过低',
  too_short: '时长过短',
  missing_transcript: '缺少文本',
  text_mismatch: '文本不匹配',
}

function ReviewReasons({ summary }: { summary: AudioSummary }) {
  const navigate = useNavigate()
  const counts = summary.samples.reduce<Record<string, number>>((result, sample) => {
    if (sample.curator_decision !== 'review' && sample.final_status !== 'needs_review') return result
    const tags = sample.issue_tags.length ? sample.issue_tags : ['unknown']
    tags.forEach((tag) => { result[tag] = (result[tag] || 0) + 1 })
    return result
  }, {})
  const rows = Object.entries(counts).sort(([, left], [, right]) => right - left).slice(0, 5)
  const max = Math.max(...rows.map(([, count]) => count), 1)
  return <section className="glassPanel insightPanel" data-reveal>
    <div className="panelHeader"><div><p className="sectionKicker">REVIEW QUEUE</p><h2>复核原因排行</h2></div><AlertTriangle size={16} /></div>
    {rows.length ? <div className="reasonList">{rows.map(([tag, count]) => <div className="reasonRow" key={tag}><span>{issueLabels[tag] || tag.replace(/_/g, ' ')}</span><div className="reasonTrack"><span style={{ width: `${(count / max) * 100}%` }} /></div><strong>{count}</strong></div>)}</div> : <div className="widgetEmpty">当前批次没有待复核原因。</div>}
    <button className="textAction" type="button" onClick={() => navigate('/pipeline/decisions')}>查看复核样本 <span aria-hidden="true">→</span></button>
  </section>
}

function BatchTimeline() {
  const { job } = usePipeline()
  const currentIndex = stageProgressIndex(job?.progress, job?.status)
  const stages = [
    { label: '输入与标准化', index: 1 },
    { label: 'VAD / ASR', index: 3 },
    { label: '质量分析', index: 4 },
    { label: 'Agent 决策', index: 5 },
    { label: '导出报告', index: 7 },
  ]
  const percent = job?.progress?.percent ?? 0
  return <section className="glassPanel timelinePanel" data-reveal>
    <div className="panelHeader"><div><p className="sectionKicker">RUN STATUS</p><h2>批次运行时间线</h2></div><span className="panelMeta">{jobStatusLabel(job?.status)}</span></div>
    <div className="runProgress"><span style={{ width: `${percent}%` }} /></div>
    <div className="runTimeline" role="list" aria-label="批次运行时间线">{stages.map((stage) => { const complete = job?.status === 'succeeded' || currentIndex > stage.index; const active = job?.status === 'running' && currentIndex === stage.index; return <div className={`runTimelineStep ${complete ? 'complete' : ''} ${active ? 'active' : ''}`} role="listitem" key={stage.label}><span className="timelineDot">{complete ? <CheckCircle2 size={14} /> : <span />}</span><strong>{stage.label}</strong><small>{complete ? '已完成' : active ? (job?.progress?.stage_message || '处理中') : '待执行'}</small></div> })}</div>
  </section>
}

export function PipelineOverviewPage() {
  const { summary, job, message, startJob } = usePipeline()

  const total = summary?.batch.total_samples || 0
  const rejectRate = total ? `${((summary?.batch.rejected_samples || 0) / total * 100).toFixed(1)}%` : '未计算'
  const inputCount = summary?.funnel[0]?.value ?? total
  const qualityScore = summary?.qualityScore
  const qualityRating = qualityScore == null ? '' : qualityScore >= 0.9 ? 'A' : qualityScore >= 0.75 ? 'B' : 'C'
  const averageQuality = qualityScore == null ? '--' : `${qualityScore.toFixed(2)} · ${qualityRating}`
  const metrics = [
    { label: '原始音频数', value: summary ? formatNumber(inputCount) : '--', note: '导入目录中的可识别音频', icon: FileAudio },
    { label: '处理后样本数', value: summary ? formatNumber(summary.batch.usable_samples) : '--', note: '当前批次可用于导出的样本', icon: CheckCircle2 },
    { label: '拒绝率', value: summary ? rejectRate : '--', note: summary ? `${formatNumber(summary.batch.rejected_samples)} 条未通过当前规范` : '处理完成后生成', icon: Database },
    { label: '平均质量分', value: averageQuality, note: qualityScore == null ? '报告尚未提供统一质量分' : '来自本次质量报告', help: '质量分由 SNR、静音比例、削波比例、响度和时长综合估算。', icon: Activity },
  ]
  const runAndJump = async () => {
    await startJob()
    window.setTimeout(() => document.getElementById('pipeline-overview')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80)
  }

  return (
    <div className="pageContent">
      <header className="pageHeading" data-reveal>
        <div className="headingMain"><p className="sectionKicker">AUDIO PIPELINE / 01</p><div className="headingTitleRow"><h2>音频数据集质量自动化流水线</h2></div><p>先配置处理目标，再运行本地音频批次并查看质量结果。</p></div>
      </header>
      <section id="pipeline-config" className="configModule" data-reveal>
        <div className="moduleHeading"><div><p className="sectionKicker">CONFIGURATION</p><h2>处理配置</h2><p>选择数据用途、质量分析方式和执行器。配置只在点击运行后提交。</p></div></div>
        <SettingsSection onRun={runAndJump} />
      </section>
      {summary && <section id="pipeline-overview" className="overviewResults" aria-labelledby="overview-results-title">
      <div className="moduleHeading" data-reveal><div><p className="sectionKicker">QUALITY RESULT</p><h2 id="overview-results-title">处理概览</h2><p>查看本批次的处理进度、样本分布和 Agent 执行结果。</p></div></div>
      <section className="metrics overviewMetrics" data-reveal>
        {metrics.map(({ label, value, note, help, icon: Icon }) => <article className="metric glassPanel" key={label}><div className="metricTop"><span>{label}{help && <InfoHint text={help} />}</span><Icon size={16} /></div><strong>{value}</strong><small>{note}</small></article>)}
      </section>
      <div className="insightGrid"><QualityDistribution summary={summary} /><ReviewReasons summary={summary} /></div>
      <BatchTimeline />
      <PipelineStepper />
      <div className="overviewGrid">
        <section className={`${revealClass()} funnelPanel`} data-reveal>
          <div className="panelHeader"><div><p className="sectionKicker">SAMPLE FLOW</p><h2>样本处理漏斗</h2></div><span className="panelMeta">{formatNumber(total)} 条记录</span></div>
          <div className="funnelOverview">{summary.funnel.map((item, index) => <div className="funnelStep" key={item.label}><span className="funnelIndex">{String(index + 1).padStart(2, '0')}</span><strong>{formatNumber(item.value)}</strong><small>{item.label}</small>{index < summary.funnel.length - 1 && <span className="funnelArrow" aria-hidden="true">→</span>}</div>)}</div>
        </section>
        <section className={`${revealClass()} agentPanel`} data-reveal>
          <div className="panelHeader"><div><p className="sectionKicker">AGENT GRAPH</p><h2>决策节点</h2></div><GitBranch size={16} /></div>
          <div className="agentGraph">
            {(summary.agents.length ? summary.agents : [{ name: 'Inspector', role: '质量证据', input: '-', output: '等待数据', decision_power: '分析' }, { name: 'Strategy Agent', role: '处理建议', input: '-', output: '等待数据', decision_power: '建议' }, { name: 'Curator Agent', role: '样本去留', input: '-', output: '等待数据', decision_power: '决策' }]).slice(0, 3).map((agent, index) => <div className="agentNode" key={agent.name}><div className="agentNodeHeader"><span className="agentNodeIndex">0{index + 1}</span><span className="statusDot success" /></div><strong>{agent.name}</strong><small>{agent.role}</small><p title={agent.output}>{agent.output || '已完成节点处理'}</p></div>)}
          </div>
        </section>
      </div>
      <section className="glassPanel toolsPanel" data-reveal>
        <div className="panelHeader"><div><p className="sectionKicker">TOOLCHAIN</p><h2>处理引擎</h2></div><span className="panelMeta">{summary.tools.length} 个已登记工具</span></div>
        <div className="toolGrid">{summary.tools.map((tool) => <div className="toolItem" key={tool.name}><span className="toolIcon"><Wrench size={15} /></span><div><strong>{tool.name}</strong><small>{tool.version || tool.type || '本地工具'} · {tool.status || '已登记'}</small></div><Cpu size={14} className="toolState" /></div>)}</div>
      </section>
      <div className="overviewStatus" role="status"><Activity size={14} /> {message} {job?.progress && <span>· {job.progress.stage_message}</span>}</div>
      </section>}
      {!summary && <section id="pipeline-overview" className="overviewResults emptyOverview" aria-live="polite"><div className="pageState"><Activity size={18} /><span>尚未运行处理任务，选择目录后点击“开始处理”。</span></div></section>}
    </div>
  )
}
