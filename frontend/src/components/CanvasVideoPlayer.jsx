import React, { useEffect, useRef, useState, useCallback } from "react"

export default function CanvasVideoPlayer({ src, className }) {
  const canvasRef = useRef(null)
  const vidRef = useRef(null)
  const rafRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState(false)

  const paint = useCallback(() => {
    const vid = vidRef.current
    const canvas = canvasRef.current
    if (!vid || !canvas) return
    const ctx = canvas.getContext("2d")
    ctx.drawImage(vid, 0, 0, canvas.width, canvas.height)
    setCurrentTime(vid.currentTime)
    if (!vid.paused && !vid.ended) {
      rafRef.current = requestAnimationFrame(paint)
    }
  }, [])

  useEffect(() => {
    if (!src) return
    setReady(false); setError(false); setPlaying(false)
    setCurrentTime(0); setDuration(0)
    cancelAnimationFrame(rafRef.current)

    const vid = document.createElement("video")
    vid.style.cssText = "position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none"
    vid.muted = true
    vid.playsInline = true
    vid.preload = "auto"
    document.body.appendChild(vid)
    vidRef.current = vid

    const onMeta = () => {
      const canvas = canvasRef.current
      if (canvas) { canvas.width = vid.videoWidth || 640; canvas.height = vid.videoHeight || 360 }
      setDuration(vid.duration)
      vid.currentTime = 0.1
    }
    const onSeeked = () => { setReady(true); paint() }
    const onEnded = () => { setPlaying(false); cancelAnimationFrame(rafRef.current) }
    const onError = () => setError(true)

    vid.addEventListener("loadedmetadata", onMeta)
    vid.addEventListener("seeked", onSeeked)
    vid.addEventListener("ended", onEnded)
    vid.addEventListener("error", onError)
    vid.src = src
    vid.load()

    return () => {
      cancelAnimationFrame(rafRef.current)
      vid.pause()
      vid.removeEventListener("loadedmetadata", onMeta)
      vid.removeEventListener("seeked", onSeeked)
      vid.removeEventListener("ended", onEnded)
      vid.removeEventListener("error", onError)
      vid.src = ""
      if (vid.parentNode) vid.parentNode.removeChild(vid)
    }
  }, [src, paint])

  const togglePlay = () => {
    const vid = vidRef.current
    if (!vid || !ready) return
    if (vid.paused) {
      vid.muted = false
      vid.play().then(() => { setPlaying(true); rafRef.current = requestAnimationFrame(paint) }).catch(e => console.warn(e))
    } else {
      vid.pause(); setPlaying(false); cancelAnimationFrame(rafRef.current)
    }
  }

  const seek = (e) => {
    const vid = vidRef.current
    if (!vid || !duration) return
    const rect = e.currentTarget.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    vid.currentTime = ratio * duration
    setCurrentTime(ratio * duration)
    setTimeout(() => paint(), 80)
  }

  const fmt = (s) => {
    if (!isFinite(s) || isNaN(s)) return "0:00"
    return `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, "0")}`
  }

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0

  if (error) return (
    <div className={`flex flex-col items-center justify-center bg-[#0a0e1a] rounded-xl py-10 gap-2 ${className}`}>
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="1.5">
        <rect x="2" y="2" width="20" height="20" rx="3"/><path d="M10 8l6 4-6 4V8z"/>
      </svg>
      <p className="text-slate-500 text-sm">Video preview unavailable</p>
      <p className="text-slate-600 text-xs">File is ready for analysis</p>
    </div>
  )

  return (
    <div className={`relative bg-black rounded-xl overflow-hidden ${className}`}>
      <canvas ref={canvasRef} className="w-full" style={{ display: "block", maxHeight: "320px" }} />
      {!ready && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70 rounded-xl">
          <div className="w-8 h-8 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
        </div>
      )}
      {ready && (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-3 pb-2 pt-6">
          <div className="w-full h-1.5 bg-white/20 rounded-full cursor-pointer mb-2" onClick={seek}>
            <div className="h-full bg-purple-400 rounded-full" style={{ width: `${progress}%` }} />
          </div>
          <div className="flex items-center gap-3">
            <button onClick={togglePlay} className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center transition-colors">
              {playing
                ? <svg width="12" height="12" viewBox="0 0 24 24" fill="white"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                : <svg width="12" height="12" viewBox="0 0 24 24" fill="white"><path d="M8 5v14l11-7z"/></svg>
              }
            </button>
            <span className="text-white/70 text-xs tabular-nums">{fmt(currentTime)} / {fmt(duration)}</span>
          </div>
        </div>
      )}
    </div>
  )
}