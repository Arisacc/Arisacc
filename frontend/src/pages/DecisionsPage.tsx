import { AudioLines, ChevronLeft, ChevronRight, FileAudio, Info, Pause, Play, Search, SlidersHorizontal, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { StatusBadge } from '../components/StatusBadge'
import { useAudioPlayer } from '../hooks/useAudioPlayer'
import { AudioSample, decisionLabels, displayStrategy, formatNumber, sampleAudioUrl, usePipeline } from '../state/PipelineContext'

function MiniWave({ sampleId, playing, onToggle, disabled }: { sampleId: string; playing: boolean; onToggle: () => void; disabled: boolean }) {
  const bars = Array.from({ length: 12 }, (_, index) => ((sampleId.charCodeAt(index % Math.max(1, sampleId.length)) + index * 13) % 70) + 20)
  return <button className={`miniWaveButton ${playing ? 'playing' : ''}`} type="button" disabled={disabled} aria-label={playing ? '暂停音频' : '播放音频'} onClick={(event) => { event.stopPropagation(); onToggle() }}><span className="miniWave" aria-hidden="true">{bars.map((height, index) => <i key={index} style={{ height: `${height}%` }} />)}</span>{playing ? <Pause size={13} /> : <Play size={13} />}</button>
}

function SampleDrawer({ sample, onClose }: { sample: AudioSample; onClose: () => void }) {
  const { job, saveReviewOverride } = usePipeline()
  const [decision, setDecision] = useState(sample.curator_decision as 'keep' | 'reject' | 'repair' | 'review')
  const [transcript, setTranscript] = useState(sample.transcript || '')
  const [note, setNote] = useState('')
  const [saving, setSaving] = useState(false)
  const audioUrl = sample.final_audio_path ? sampleAudioUrl(job, sample.sample_id) : ''
  const save = async () => {
    setSaving(true)
    try { await saveReviewOverride(sample, decision, transcript, note); onClose() } finally { setSaving(false) }
  }
  return <aside className="drawer" aria-label={`样本 ${sample.sample_id} 详情`}>
    <button className="drawerClose" type="button" onClick={onClose} aria-label="关闭样本详情"><X size={17} /></button>
    <p className="sectionKicker">SAMPLE REVIEW</p>
    <h2>{sample.sample_id}</h2>
    {audioUrl ? <div className="audioPreview"><span><AudioLines size={14} /> 最终音频</span><audio controls src={audioUrl} preload="metadata" /></div> : <div className="emptyInline"><FileAudio size={16} />当前样本暂无可试听的最终音频</div>}
    <div className="reviewBox"><h3>人工复核</h3><label><span>覆盖决策</span><select value={decision} onChange={(event) => setDecision(event.target.value as typeof decision)}>{Object.entries(decisionLabels).map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label><label><span>修订文本</span><textarea value={transcript} onChange={(event) => setTranscript(event.target.value)} rows={3} /></label><label><span>复核备注</span><textarea value={note} onChange={(event) => setNote(event.target.value)} rows={2} /></label><button type="button" onClick={save} disabled={!job || job.status !== 'succeeded' || saving}>{saving ? '保存中' : '保存复核意见'}</button></div>
    <dl className="sampleExplain"><dt>审核决策</dt><dd><StatusBadge value={sample.curator_decision} /></dd><dt>最终状态</dt><dd><StatusBadge value={sample.final_status} /></dd><dt>处理策略</dt><dd>{displayStrategy(sample.strategy)}</dd><dt>决策理由</dt><dd>{sample.curator_reason || sample.strategy_reason || '未提供'}</dd><dt>文本来源</dt><dd>{sample.transcript_source}</dd><dt>ASR 置信度</dt><dd>{sample.asr_confidence ?? '未计算'}</dd><dt>SNR</dt><dd>{sample.snr == null ? '未计算' : `${sample.snr.toFixed(1)} dB`}</dd><dt>质量分</dt><dd>{sample.quality_score == null ? '未计算' : sample.quality_score.toFixed(4)}</dd><dt>参考文本</dt><dd>{sample.reference_text || '未提供'}</dd><dt>ASR 候选文本</dt><dd>{sample.transcript || '未生成'}</dd><dt>修复链</dt><dd>{sample.repair_chain?.length ? sample.repair_chain.map((step) => `${step.step_index}. ${step.operation}`).join(' → ') : '未生成'}</dd></dl>
  </aside>
}

export function DecisionsPage() {
  const { summary, job } = usePipeline()
  const [status, setStatus] = useState('all')
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<'sample_id' | 'duration' | 'confidence'>('sample_id')
  const [descending, setDescending] = useState(false)
  const [page, setPage] = useState(1)
  const [selected, setSelected] = useState<AudioSample | null>(null)
  const audioPlayer = useAudioPlayer()
  const sampleData = summary?.samples || []
  const counts = sampleData.reduce<Record<string, number>>((result, sample) => { result[sample.curator_decision] = (result[sample.curator_decision] || 0) + 1; return result }, {})
  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    const values = sampleData.filter((sample) => {
      const searchable = [sample.sample_id, sample.transcript, sample.reference_text, sample.strategy, sample.strategy_reason, sample.curator_reason, ...sample.issue_tags].filter(Boolean).join(' ').toLowerCase()
      return (status === 'all' || sample.curator_decision === status) && (!normalized || searchable.includes(normalized))
    }).sort((a, b) => {
      const value = sort === 'duration' ? a.duration - b.duration : sort === 'confidence' ? (a.asr_confidence ?? -1) - (b.asr_confidence ?? -1) : a.sample_id.localeCompare(b.sample_id)
      return descending ? -value : value
    })
    return values
  }, [sampleData, status, query, sort, descending])
  if (!summary) return <section className="pageState">正在加载样本决策...</section>
  const pageSize = 50
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize))
  const visible = filtered.slice((page - 1) * pageSize, page * pageSize)

  return <div className="pageContent decisionsPage">
    <header className="pageHeading" data-reveal><div><p className="sectionKicker">DECISIONS / 02</p><h2>样本决策</h2><p>按处理状态和问题证据筛选样本，打开详情进行人工复核。</p></div><span className="routeBadge"><FileAudio size={14} /> {formatNumber(filtered.length)} 条结果</span></header>
    <section className="glassPanel decisionToolbar" data-reveal><div className="decisionTabs" role="tablist" aria-label="决策状态筛选">{['all', 'keep', 'reject', 'repair', 'review'].map((value) => <button type="button" role="tab" aria-selected={status === value} className={status === value ? `active ${value}` : value} onClick={() => { setStatus(value); setPage(1) }} key={value}>{value === 'all' ? '全部' : decisionLabels[value]} <b>{value === 'all' ? summary.samples.length : counts[value] || 0}</b></button>)}</div><div className="filterRow"><label className="searchField"><Search size={15} /><input value={query} onChange={(event) => { setQuery(event.target.value); setPage(1) }} placeholder="搜索文件名、文本、原因或标签" /></label><label className="selectField"><SlidersHorizontal size={14} /><select value={`${sort}:${descending ? 'desc' : 'asc'}`} onChange={(event) => { const [field, direction] = event.target.value.split(':'); setSort(field as typeof sort); setDescending(direction === 'desc') }}><option value="sample_id:asc">样本 ID 升序</option><option value="sample_id:desc">样本 ID 降序</option><option value="duration:desc">时长降序</option><option value="confidence:asc">置信度升序</option></select></label></div></section>
    <section className="glassPanel tablePanel" data-reveal><div className="tableWrap"><table><thead><tr><th>样本</th><th>时长</th><th><span className="tableHeaderHint">SNR <span title="SNR 为 RMS 与噪声底之差的估算值，单位为 dB。"><Info size={12} /></span></span></th><th>策略建议</th><th>决策理由</th><th>最终状态</th><th><span className="srOnly">操作</span></th></tr></thead><tbody>{visible.map((sample) => { const audioUrl = sample.final_audio_path ? sampleAudioUrl(job, sample.sample_id) : ''; const playing = audioPlayer.activeId === sample.sample_id && audioPlayer.isPlaying; return <tr className={playing ? 'playingRow' : ''} key={sample.sample_id} tabIndex={0} onClick={() => setSelected(sample)} onKeyDown={(event) => { if (event.key === 'Enter') setSelected(sample); if (event.key === ' ') { event.preventDefault(); audioPlayer.toggle(sample.sample_id, audioUrl) } }}><td><div className="sampleCell"><MiniWave sampleId={sample.sample_id} playing={playing} disabled={!audioUrl} onToggle={() => audioPlayer.toggle(sample.sample_id, audioUrl)} /><span><strong>{sample.sample_id}</strong><small>{sample.transcript_source}</small></span></div></td><td>{sample.duration.toFixed(2)}s</td><td className="monoValue">{sample.snr == null ? '未计算' : `${sample.snr.toFixed(1)} dB`}</td><td>{displayStrategy(sample.strategy)}</td><td className="reasonCell">{sample.curator_reason || sample.strategy_reason || '未提供'}</td><td><StatusBadge value={sample.final_status} /></td><td><button className="rowAction" type="button" onClick={(event) => { event.stopPropagation(); setSelected(sample) }}>查看</button></td></tr>})}</tbody></table>{visible.length === 0 && <div className="emptyState"><Search size={22} /><strong>没有匹配样本</strong><span>调整筛选条件或搜索关键词后重试。</span></div>}</div><div className="pagination"><span>第 {page} / {pageCount} 页</span><button type="button" className="compactButton" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}><ChevronLeft size={14} /> 上一页</button><button type="button" className="compactButton" disabled={page >= pageCount} onClick={() => setPage((value) => value + 1)}>下一页 <ChevronRight size={14} /></button></div></section>
    {selected && <SampleDrawer sample={selected} onClose={() => setSelected(null)} />}
  </div>
}
