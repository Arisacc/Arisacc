import { useEffect, useRef } from 'react'
const API_BASE = (import.meta.env.VITE_AUDIO_API_BASE || 'http://127.0.0.1:8000').replace(/\/$/, '')

const eventNames = ['run_created', 'task_enqueued', 'task_claimed', 'worker_started', 'status_changed', 'checkpoint_saved', 'task_succeeded', 'task_failed', 'worker_failed']

export function useRunEvents(jobId: string | undefined, status: string | undefined, onEvent: () => void) {
  const lastEventId = useRef(0)
  const callback = useRef(onEvent)
  callback.current = onEvent

  useEffect(() => {
    if (!jobId || !['queued', 'running'].includes(status || '')) return
    let source: EventSource | null = null
    let stopped = false
    const connect = () => {
      if (stopped) return
      const query = lastEventId.current ? `?last_event_id=${lastEventId.current}` : ''
      source = new EventSource(`${API_BASE}/api/v1/runs/${encodeURIComponent(jobId)}/events/stream${query}`)
      const handle = (event: MessageEvent) => {
        const id = Number(event.lastEventId)
        if (Number.isFinite(id) && id > 0) lastEventId.current = id
        callback.current()
      }
      eventNames.forEach((name) => source?.addEventListener(name, handle))
      source.onerror = () => {
        source?.close()
        if (!stopped) window.setTimeout(connect, 1200)
      }
    }
    connect()
    return () => {
      stopped = true
      source?.close()
    }
  }, [jobId, status])
}
