import { useEffect, useState } from 'react'
import { Activity, AudioLines, CheckCircle2, FileText, Menu, Moon, Sun, X } from 'lucide-react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { jobStatusLabel, policyLabels, usePipeline } from '../state/PipelineContext'

function routeKey(pathname: string): string {
  if (pathname.startsWith('/pipeline/decisions')) return 'decisions'
  if (pathname.startsWith('/pipeline/report')) return 'report'
  if (pathname.startsWith('/settings')) return 'settings'
  return 'pipeline'
}

export function AppShell() {
  const location = useLocation()
  const navigate = useNavigate()
  const { form, job, summary, backendStatus } = usePipeline()
  const [theme, setTheme] = useState<'light' | 'dark'>(() => window.localStorage.getItem('audio-agent-theme-v2') === 'light' ? 'light' : 'dark')
  const [menuOpen, setMenuOpen] = useState(false)
  const [runNotice, setRunNotice] = useState<{ tone: 'success' | 'error'; text: string } | null>(null)
  const currentRoute = routeKey(location.pathname)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.body.dataset.route = currentRoute
    window.localStorage.setItem('audio-agent-theme-v2', theme)
  }, [theme, currentRoute])

  useEffect(() => setMenuOpen(false), [location.pathname])

  useEffect(() => {
    if (job?.status === 'succeeded') setRunNotice({ tone: 'success', text: `处理完成，共 ${summary?.batch.total_samples ?? summary?.samples.length ?? 0} 个样本` })
    if (job?.status === 'failed') setRunNotice({ tone: 'error', text: job.error_message || '处理失败，请检查后端日志和输入目录。' })
    if (!job || !['succeeded', 'failed'].includes(job.status)) return
    const timer = window.setTimeout(() => setRunNotice(null), 3000)
    return () => window.clearTimeout(timer)
  }, [job?.status, job?.error_message, summary])

  useEffect(() => {
    const elements = Array.from(document.querySelectorAll<HTMLElement>('[data-reveal]'))
    elements.forEach((element) => element.classList.add('reveal-pending'))
    if (!('IntersectionObserver' in window)) {
      elements.forEach((element) => element.classList.add('reveal-visible'))
      return
    }
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return
        entry.target.classList.add('reveal-visible')
        observer.unobserve(entry.target)
      })
    }, { threshold: 0.08 })
    elements.forEach((element) => observer.observe(element))
    return () => observer.disconnect()
  }, [location.pathname, summary])

  return (
    <div className="appFrame">
      <header className="appBar">
        <div className="brandBlock">
          <div className="brandMark"><img src="/audio-agent-icon.png" alt="" /></div>
          <div>
            <p className="eyebrow">AUDIO DATASET AGENT</p>
            <h1>声音数据处理工作台</h1>
          </div>
        </div>
        <button className="mobileMenuButton" type="button" aria-label={menuOpen ? '关闭导航' : '打开导航'} aria-expanded={menuOpen} onClick={() => setMenuOpen((open) => !open)}>
          {menuOpen ? <X size={18} /> : <Menu size={18} />}
        </button>
        <div className={`appNavigation ${menuOpen ? 'open' : ''}`}>
          <span className="navProduct"><Activity size={14} /> 音频流水线</span>
          <span className="navDivider" aria-hidden="true">›</span>
          <nav className="navSegmented" aria-label="主导航" role="tablist">
            <NavLink to="/pipeline" end role="tab"><Activity size={17} /> 概览</NavLink>
            <NavLink to="/pipeline/decisions" role="tab"><AudioLines size={17} /> 决策</NavLink>
            <NavLink to="/pipeline/report" role="tab"><FileText size={17} /> 报告</NavLink>
          </nav>
          <div className="navActions">
            <span className="navStatus"><span className={`statusDot ${backendStatus === 'offline' ? 'danger' : job?.status === 'running' ? 'running' : 'success'}`} />{backendStatus === 'offline' ? '后端未连接' : jobStatusLabel(job?.status)}</span>
            <button className="iconButton" type="button" title={theme === 'dark' ? '切换浅色主题' : '切换深色主题'} aria-label={theme === 'dark' ? '切换浅色主题' : '切换深色主题'} onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
              {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </div>
      </header>
      <div className="appContextLine">
        <span>{summary ? `${policyLabels[summary.batch.target_policy as keyof typeof policyLabels] || summary.batch.target_policy} · ${summary.batch.batch_id}` : `${policyLabels[form.target_policy]} · 等待输入`}</span>
        <span className="contextMessage">本地优先 · 数据不离开当前环境</span>
      </div>
      <main className="routeMain">
        <div className="routeView" key={location.pathname}>
          <Outlet />
        </div>
      </main>
      {runNotice && <div className={`completionToast ${runNotice.tone}`} role={runNotice.tone === 'error' ? 'alert' : 'status'}><CheckCircle2 size={17} /><span>{runNotice.text}</span>{runNotice.tone === 'success' && <button className="toastAction" type="button" onClick={() => { navigate('/pipeline'); window.setTimeout(() => document.getElementById('pipeline-overview')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80) }}>查看结果</button>}<button className="toastClose" type="button" onClick={() => setRunNotice(null)} aria-label="关闭提示"><X size={14} /></button></div>}
    </div>
  )
}
