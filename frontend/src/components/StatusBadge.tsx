import { AlertCircle, CheckCircle2, RefreshCw, XCircle } from 'lucide-react'
import { decisionLabels } from '../state/PipelineContext'

const icons = { keep: CheckCircle2, reject: XCircle, repair: RefreshCw, review: AlertCircle }

export function StatusBadge({ value }: { value: string }) {
  const Icon = icons[value as keyof typeof icons] || AlertCircle
  return <span className={`statusPill ${value}`}><Icon size={13} /> {decisionLabels[value] || value}</span>
}
