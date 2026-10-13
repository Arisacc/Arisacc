import { useCallback, useEffect, useRef, useState } from 'react'

export function useAudioPlayer() {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    const audio = new Audio()
    audio.preload = 'metadata'
    audio.addEventListener('ended', () => { setIsPlaying(false); setActiveId(null) })
    audio.addEventListener('pause', () => setIsPlaying(false))
    audio.addEventListener('play', () => setIsPlaying(true))
    audioRef.current = audio
    return () => { audio.pause(); audio.src = ''; audioRef.current = null }
  }, [])

  const toggle = useCallback(async (id: string, url: string) => {
    const audio = audioRef.current
    if (!audio || !url) return
    if (activeId === id && !audio.paused) {
      audio.pause()
      return
    }
    if (activeId !== id) {
      audio.pause()
      audio.src = url
      audio.currentTime = 0
      setActiveId(id)
    }
    try { await audio.play() } catch { setIsPlaying(false) }
  }, [activeId])

  const pause = useCallback(() => audioRef.current?.pause(), [])
  return { activeId, isPlaying, toggle, pause }
}
