import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { DecisionsPage } from './pages/DecisionsPage'
import { PipelineOverviewPage } from './pages/PipelineOverviewPage'
import { ReportPage } from './pages/ReportPage'
import { PipelineProvider } from './state/PipelineContext'
import './style.css'

function App() {
  return <PipelineProvider><BrowserRouter><Routes><Route element={<AppShell />}><Route path="/" element={<Navigate to="/pipeline" replace />} /><Route path="/pipeline" element={<PipelineOverviewPage />} /><Route path="/pipeline/decisions" element={<DecisionsPage />} /><Route path="/pipeline/report" element={<ReportPage />} /><Route path="/settings" element={<Navigate to="/pipeline#pipeline-config" replace />} /><Route path="*" element={<Navigate to="/pipeline" replace />} /></Route></Routes></BrowserRouter></PipelineProvider>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
