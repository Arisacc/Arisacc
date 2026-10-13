import { BarChart3, CheckCircle2, Download, FileJson, GitBranch, Link2, PackageCheck } from 'lucide-react'
import { jobReportUrl, formatNumber, usePipeline } from '../state/PipelineContext'

function JsonTree({ value, level = 0 }: { value: unknown; level?: number }) {
  if (value === null || typeof value !== 'object') return <span className="jsonValue">{JSON.stringify(value)}</span>
  if (Array.isArray(value)) return <details className="jsonNode" open={level < 1}><summary>数组 · {value.length} 项</summary><div className="jsonChildren">{value.slice(0, 20).map((item, index) => <div className="jsonEntry" key={index}><span className="jsonKey">[{index}]</span><JsonTree value={item} level={level + 1} /></div>)}</div></details>
  return <details className="jsonNode" open={level < 1}><summary>对象 · {Object.keys(value).length} 个字段</summary><div className="jsonChildren">{Object.entries(value).slice(0, 24).map(([key, item]) => <div className="jsonEntry" key={key}><span className="jsonKey">{key}</span><JsonTree value={item} level={level + 1} /></div>)}</div></details>
}

export function ReportPage() {
  const { summary, job } = usePipeline()
  if (!summary) return <section className="pageState">正在加载处理报告...</section>
  const maxValue = Math.max(1, ...summary.funnel.map((item) => item.value))
  const reportLinks = job?.status === 'succeeded' ? [['作品集 HTML', 'portfolio_report_html'], ['作品集 Markdown', 'portfolio_report_markdown'], ['质量报告', 'dataset_quality_report'], ['批次报告', 'batch_report']] : []
  const manifest = { batch: summary.batch, report_paths: summary.report_paths || {}, tools: summary.tools.map((tool) => ({ name: tool.name, version: tool.version || '未提供', status: tool.status || '已登记' })) }
  return <div className="pageContent reportPage">
    <header className="pageHeading" data-reveal><div><p className="sectionKicker">REPORT / 03</p><h2>处理报告</h2><p>查看本批次导出文件、统计变化和 Agent 决策追溯。</p></div><span className="routeBadge"><PackageCheck size={14} /> {job?.status === 'succeeded' ? '已生成' : '等待导出'}</span></header>
    <div className="reportGrid">
      <section className="glassPanel manifestPanel" data-reveal><div className="panelHeader"><div><p className="sectionKicker">MANIFEST</p><h2>导出结构</h2></div><FileJson size={16} /></div><div className="jsonTree"><JsonTree value={manifest} /></div></section>
      <section className="glassPanel reportStats" data-reveal><div className="panelHeader"><div><p className="sectionKicker">BATCH METRICS</p><h2>处理统计</h2></div><BarChart3 size={16} /></div><div className="barChart" role="img" aria-label="样本处理漏斗统计">{summary.funnel.map((item) => <div className="barItem" key={item.label}><div className="barTrack"><span style={{ height: `${Math.max(6, (item.value / maxValue) * 100)}%` }} /></div><strong>{formatNumber(item.value)}</strong><small>{item.label}</small></div>)}</div></section>
    </div>
    <section className="glassPanel reportOutputs" data-reveal><div className="panelHeader"><div><p className="sectionKicker">OUTPUTS</p><h2>报告文件</h2></div><span className="panelMeta">保留原始事实，不覆盖样本决策</span></div><div className="reportButtons">{reportLinks.length ? reportLinks.map(([label, name]) => <a href={jobReportUrl(job, name)} target="_blank" rel="noreferrer" key={name}><Download size={14} /> {label}</a>) : <span className="emptyInline">任务完成后可下载本批次报告</span>}</div></section>
    <section className="glassPanel tracePanel" data-reveal><div className="panelHeader"><div><p className="sectionKicker">TRACE</p><h2>决策追溯链</h2></div><GitBranch size={16} /></div><div className="traceChain">{summary.graph_trace.map((node, index) => <div className="traceNode" key={`${node.node_name}-${node.agent_name}`}><span className="traceIndex">{String(index + 1).padStart(2, '0')}</span><div><strong>{node.agent_name}</strong><small>{node.node_name} · {node.uses_agent ? 'Agent' : '工具'}{node.uses_tool ? ' + Tool' : ''}</small></div>{index < summary.graph_trace.length - 1 && <Link2 size={14} className="traceArrow" />}</div>)}</div></section>
    <section className="reportNote" role="note"><CheckCircle2 size={15} /> 数据规范、工具版本和处理链已随批次产物一并记录。</section>
  </div>
}
