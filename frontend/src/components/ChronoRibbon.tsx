/**
 * Multi-track timeline on a single canvas, with zoom and pan.
 *
 * Tracks, top to bottom: scene boundaries and type, a filmstrip of keyframes at
 * their real timestamps, diarisation lanes, a visual-change area over a
 * speech-density line, and a band marking the current answer's citations.
 *
 * A fixed full-duration view only works for short clips. At two hours a scene
 * occupies under three pixels and the strip can show 3% of the keyframes, so
 * the ribbon renders a *window* of the timeline: scroll to zoom around the
 * pointer, drag to pan, and a minimap underneath keeps the whole recording in
 * view. The filmstrip resamples inside the window, so zooming in reveals more
 * frames rather than stretching the same few.
 *
 * Canvas rather than DOM: a long recording produces thousands of marks, and a
 * redraw is one paint where thousands of positioned elements would stall each
 * seek. Redraws are coalesced through rAF and scaled by devicePixelRatio.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../lib/api'
import { fmtTime } from '../lib/format'
import type { Citation, Keyframe, Scene, Segment, VideoChunk } from '../lib/types'
import { Button, FrameImage, Icon, Tip } from './ui'

const H = { scenes: 16, strip: 46, speakers: 22, activity: 34, evidence: 12, gap: 5 }
const TOTAL_H = H.scenes + H.strip + H.speakers + H.activity + H.evidence + H.gap * 4
const MINIMAP_H = 22
const PAD = 10
const MIN_WINDOW_S = 2

interface View {
  start: number
  end: number
}

interface Props {
  duration: number
  scenes: Scene[]
  keyframes: Keyframe[]
  chunks: VideoChunk[]
  segments: Segment[]
  citations: Citation[]
  currentTime: number
  onSeek: (t: number) => void
  onOpenFrame?: (frameId: string) => void
}

export function ChronoRibbon({
  duration, scenes, keyframes, chunks, segments, citations, currentTime, onSeek, onOpenFrame,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const miniRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  const [view, setView] = useState<View>({ start: 0, end: Math.max(duration, 1) })
  const [hover, setHover] = useState<{ x: number; t: number } | null>(null)
  const drag = useRef<{ mode: 'seek' | 'pan' | 'window'; startX: number; view: View } | null>(null)
  const images = useRef(new Map<string, HTMLImageElement>())
  const rafId = useRef(0)

  useEffect(() => { setView({ start: 0, end: Math.max(duration, 1) }) }, [duration])

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)))
    ro.observe(el)
    setWidth(Math.floor(el.getBoundingClientRect().width))
    return () => ro.disconnect()
  }, [])

  const span = Math.max(view.end - view.start, MIN_WINDOW_S)
  const zoomed = span < duration - 0.5

  const clampView = useCallback(
    (v: View): View => {
      const width_ = Math.min(Math.max(v.end - v.start, MIN_WINDOW_S), duration)
      let start = Math.max(0, Math.min(v.start, duration - width_))
      if (!Number.isFinite(start)) start = 0
      return { start, end: start + width_ }
    },
    [duration],
  )

  const zoomAt = useCallback(
    (factor: number, anchorFraction: number) => {
      setView((v) => {
        const current = v.end - v.start
        const next = Math.min(Math.max(current * factor, MIN_WINDOW_S), duration)
        const anchorTime = v.start + current * anchorFraction
        return clampView({ start: anchorTime - next * anchorFraction, end: anchorTime - next * anchorFraction + next })
      })
    },
    [clampView, duration],
  )

  // ---- thumbnails, resampled for the visible window ------------------------
  const stripFrames = useMemo(() => {
    if (!keyframes.length || !width) return []
    const visible = keyframes.filter((k) => k.timestamp >= view.start - 1 && k.timestamp <= view.end + 1)
    const pool = visible.length ? visible : keyframes
    const capacity = Math.max(4, Math.floor((width - PAD * 2) / 76))
    if (pool.length <= capacity) return pool
    const step = pool.length / capacity
    return Array.from({ length: capacity }, (_, i) => pool[Math.floor(i * step)])
  }, [keyframes, width, view])

  const scheduleDraw = useCallback(() => {
    cancelAnimationFrame(rafId.current)
    rafId.current = requestAnimationFrame(() => drawRef.current())
  }, [])

  useEffect(() => {
    let cancelled = false
    for (const kf of stripFrames) {
      if (images.current.has(kf.id)) continue
      const img = new Image()
      img.decoding = 'async'
      img.src = api.thumbUrl(kf.path)
      img.onerror = () => { img.onerror = null; img.src = api.frameUrl(kf.path) }
      img.onload = () => { if (!cancelled) scheduleDraw() }
      images.current.set(kf.id, img)
    }
    return () => { cancelled = true }
  }, [stripFrames, scheduleDraw])

  const speakers = useMemo(() => {
    const seen: string[] = []
    for (const s of segments) if (s.speaker && !seen.includes(s.speaker)) seen.push(s.speaker)
    return seen.sort()
  }, [segments])

  const activity = useMemo(() => {
    if (!chunks.length || !duration) return { visual: [] as number[], speech: [] as number[], buckets: 0 }
    const buckets = Math.max(24, Math.min(480, Math.floor(width / 3) || 120))
    const visual = new Array(buckets).fill(0)
    const speech = new Array(buckets).fill(0)
    for (const c of chunks) {
      const from = Math.floor(((c.span.start - view.start) / span) * buckets)
      const to = Math.ceil(((c.span.end - view.start) / span) * buckets)
      for (let i = Math.max(0, from); i <= Math.min(buckets - 1, to); i++) {
        visual[i] = Math.max(visual[i], c.visual_activity)
        speech[i] = Math.max(speech[i], c.speech_rate)
      }
    }
    const norm = (arr: number[]) => {
      const max = Math.max(...arr, 1e-6)
      return arr.map((v) => v / max)
    }
    return { visual: norm(visual), speech: norm(speech), buckets }
  }, [chunks, duration, width, view, span])

  // ---- painting ------------------------------------------------------------
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !width || !duration) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = width * dpr
    canvas.height = TOTAL_H * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${TOTAL_H}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, TOTAL_H)

    const inner = width - PAD * 2
    const x = (t: number) => PAD + ((t - view.start) / span) * inner
    const visible = (a: number, b: number) => b >= view.start && a <= view.end
    let y = 0

    // scenes
    for (const sc of scenes) {
      if (!visible(sc.span.start, sc.span.end)) continue
      const x0 = Math.max(PAD, x(sc.span.start))
      const x1 = Math.min(PAD + inner, x(sc.span.end))
      const w = Math.max(1.5, x1 - x0)
      ctx.fillStyle = sc.kind === 'static' ? 'rgba(147, 204, 126, 0.22)' : 'rgba(255, 255, 255, 0.07)'
      roundRect(ctx, x0 + 0.5, y, w - 1, H.scenes, 3)
      ctx.fill()
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.09)'
      ctx.lineWidth = 1
      roundRect(ctx, x0 + 0.5, y + 0.5, w - 1, H.scenes - 1, 3)
      ctx.stroke()
      if (sc.index > 0 && x(sc.span.start) >= PAD) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.38)'
        ctx.fillRect(x(sc.span.start), y - 1, 1, H.scenes + 2)
      }
      if (w > 26) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'
        ctx.font = '600 9px "JetBrains Mono", monospace'
        ctx.textBaseline = 'middle'
        ctx.fillText(String(sc.index + 1), x0 + 5, y + H.scenes / 2 + 0.5)
      }
    }
    y += H.scenes + H.gap

    // filmstrip
    ctx.fillStyle = 'rgba(255, 255, 255, 0.025)'
    roundRect(ctx, PAD, y, inner, H.strip, 5)
    ctx.fill()
    const thumbW = Math.min(72, Math.max(34, inner / Math.max(stripFrames.length, 1) - 4))
    for (const kf of stripFrames) {
      const img = images.current.get(kf.id)
      const cx = x(kf.timestamp)
      if (cx < PAD - thumbW || cx > PAD + inner + thumbW) continue
      const x0 = Math.max(PAD + 1, Math.min(cx - thumbW / 2, PAD + inner - thumbW - 1))
      ctx.save()
      roundRect(ctx, x0, y + 2, thumbW, H.strip - 4, 4)
      ctx.clip()
      if (img?.complete && img.naturalWidth) {
        const scale = Math.max(thumbW / img.naturalWidth, (H.strip - 4) / img.naturalHeight)
        const dw = img.naturalWidth * scale
        const dh = img.naturalHeight * scale
        ctx.globalAlpha = 0.92
        ctx.drawImage(img, x0 + (thumbW - dw) / 2, y + 2 + (H.strip - 4 - dh) / 2, dw, dh)
        ctx.globalAlpha = 1
      } else {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.05)'
        ctx.fillRect(x0, y + 2, thumbW, H.strip - 4)
      }
      ctx.restore()
      ctx.strokeStyle = kf.is_slide ? 'rgba(187, 154, 247, 0.6)' : 'rgba(255, 255, 255, 0.1)'
      ctx.lineWidth = 1
      roundRect(ctx, x0 + 0.5, y + 2.5, thumbW - 1, H.strip - 5, 4)
      ctx.stroke()
    }
    y += H.strip + H.gap

    // speakers
    ctx.fillStyle = 'rgba(255, 255, 255, 0.025)'
    roundRect(ctx, PAD, y, inner, H.speakers, 4)
    ctx.fill()
    const laneCount = Math.max(1, Math.min(speakers.length, 4))
    const laneH = (H.speakers - 2) / laneCount
    for (const seg of segments) {
      if (!seg.speaker || !visible(seg.start, seg.end)) continue
      const lane = Math.min(speakers.indexOf(seg.speaker), laneCount - 1)
      const x0 = Math.max(PAD, x(seg.start))
      const w = Math.max(1.5, Math.min(PAD + inner, x(seg.end)) - x0)
      ctx.fillStyle = speakerHue(seg.speaker)
      ctx.globalAlpha = 0.34 + 0.55 * seg.confidence
      roundRect(ctx, x0, y + 1 + lane * laneH, w, Math.max(2.5, laneH - 1.5), 2)
      ctx.fill()
      ctx.globalAlpha = 1
    }
    y += H.speakers + H.gap

    // activity
    const { visual: vis, speech, buckets } = activity
    if (buckets) {
      const bw = inner / buckets
      ctx.beginPath()
      ctx.moveTo(PAD, y + H.activity)
      vis.forEach((v, i) => ctx.lineTo(PAD + i * bw, y + H.activity - v * (H.activity - 3)))
      ctx.lineTo(PAD + inner, y + H.activity)
      ctx.closePath()
      const area = ctx.createLinearGradient(0, y, 0, y + H.activity)
      area.addColorStop(0, 'rgba(187, 154, 247, 0.34)')
      area.addColorStop(1, 'rgba(187, 154, 247, 0.02)')
      ctx.fillStyle = area
      ctx.fill()
      ctx.beginPath()
      speech.forEach((v, i) => {
        const px = PAD + i * bw
        const py = y + H.activity - v * (H.activity - 3)
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)
      })
      ctx.strokeStyle = 'rgba(122, 162, 247, 0.85)'
      ctx.lineWidth = 1.25
      ctx.stroke()
    }
    y += H.activity + H.gap

    // evidence
    ctx.fillStyle = 'rgba(255, 255, 255, 0.02)'
    roundRect(ctx, PAD, y, inner, H.evidence, 3)
    ctx.fill()
    for (const cite of citations) {
      if (!visible(cite.start, cite.end)) continue
      const x0 = Math.max(PAD, x(cite.start))
      const w = Math.max(3, Math.min(PAD + inner, x(cite.end)) - x0)
      ctx.fillStyle = 'rgba(224, 175, 104, 0.85)'
      roundRect(ctx, x0, y + 1, w, H.evidence - 2, 2)
      ctx.fill()
    }

    // playhead
    if (currentTime >= view.start && currentTime <= view.end) {
      const px = x(currentTime)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(px, 0)
      ctx.lineTo(px, TOTAL_H)
      ctx.stroke()
      ctx.fillStyle = '#ffffff'
      ctx.beginPath()
      ctx.moveTo(px - 4, 0)
      ctx.lineTo(px + 4, 0)
      ctx.lineTo(px, 6)
      ctx.closePath()
      ctx.fill()
    }

    if (hover) {
      ctx.strokeStyle = 'rgba(122, 162, 247, 0.55)'
      ctx.lineWidth = 1
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(hover.x, 0)
      ctx.lineTo(hover.x, TOTAL_H)
      ctx.stroke()
      ctx.setLineDash([])
    }
  }, [width, duration, scenes, stripFrames, segments, speakers, activity, citations, currentTime, hover, view, span])

  // ---- minimap -------------------------------------------------------------
  const drawMini = useCallback(() => {
    const canvas = miniRef.current
    if (!canvas || !width || !duration) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = width * dpr
    canvas.height = MINIMAP_H * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${MINIMAP_H}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, MINIMAP_H)

    const inner = width - PAD * 2
    const fx = (t: number) => PAD + (t / duration) * inner

    ctx.fillStyle = 'rgba(255, 255, 255, 0.03)'
    roundRect(ctx, PAD, 4, inner, MINIMAP_H - 8, 3)
    ctx.fill()

    for (const sc of scenes) {
      const x0 = fx(sc.span.start)
      const w = Math.max(0.6, fx(sc.span.end) - x0)
      ctx.fillStyle = sc.kind === 'static' ? 'rgba(147, 204, 126, 0.4)' : 'rgba(255, 255, 255, 0.16)'
      ctx.fillRect(x0, 5, w - 0.4, MINIMAP_H - 10)
    }
    for (const cite of citations) {
      ctx.fillStyle = 'rgba(224, 175, 104, 0.9)'
      ctx.fillRect(fx(cite.start), 5, Math.max(1.5, fx(cite.end) - fx(cite.start)), MINIMAP_H - 10)
    }

    // current window
    const wx0 = fx(view.start)
    const wx1 = fx(view.end)
    ctx.fillStyle = 'rgba(122, 162, 247, 0.16)'
    ctx.fillRect(wx0, 3, Math.max(3, wx1 - wx0), MINIMAP_H - 6)
    ctx.strokeStyle = 'rgba(122, 162, 247, 0.85)'
    ctx.lineWidth = 1
    ctx.strokeRect(wx0 + 0.5, 3.5, Math.max(3, wx1 - wx0) - 1, MINIMAP_H - 7)

    ctx.fillStyle = 'rgba(255, 255, 255, 0.85)'
    ctx.fillRect(fx(currentTime) - 0.5, 3, 1, MINIMAP_H - 6)
  }, [width, duration, scenes, citations, view, currentTime])

  const drawRef = useRef(draw)
  useEffect(() => { drawRef.current = () => { draw(); drawMini() } }, [draw, drawMini])
  useEffect(() => {
    scheduleDraw()
    return () => cancelAnimationFrame(rafId.current)
  }, [draw, drawMini, scheduleDraw])

  // ---- interaction ---------------------------------------------------------
  const timeAt = useCallback(
    (clientX: number) => {
      const rect = canvasRef.current!.getBoundingClientRect()
      const rel = (clientX - rect.left - PAD) / (rect.width - PAD * 2)
      return view.start + Math.max(0, Math.min(1, rel)) * span
    },
    [view.start, span],
  )

  const miniTimeAt = useCallback((clientX: number) => {
    const rect = miniRef.current!.getBoundingClientRect()
    const rel = (clientX - rect.left - PAD) / (rect.width - PAD * 2)
    return Math.max(0, Math.min(1, rel)) * duration
  }, [duration])

  // Wheel: zoom around the pointer. Native listener so preventDefault works,
  // which React's passive wheel handler cannot do.
  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey && Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        e.preventDefault()
        setView((v) => clampView({ start: v.start + (e.deltaX / 400) * span, end: v.end + (e.deltaX / 400) * span }))
        return
      }
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const anchor = Math.max(0, Math.min(1, (e.clientX - rect.left - PAD) / (rect.width - PAD * 2)))
      zoomAt(e.deltaY > 0 ? 1.18 : 1 / 1.18, anchor)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [clampView, span, zoomAt])

  useEffect(() => {
    const up = () => { drag.current = null }
    const move = (e: PointerEvent) => {
      const d = drag.current
      if (!d) return
      if (d.mode === 'seek') onSeek(timeAt(e.clientX))
      else if (d.mode === 'pan') {
        const rect = canvasRef.current!.getBoundingClientRect()
        const dt = ((e.clientX - d.startX) / (rect.width - PAD * 2)) * (d.view.end - d.view.start)
        setView(clampView({ start: d.view.start - dt, end: d.view.end - dt }))
      } else {
        const centre = miniTimeAt(e.clientX)
        const w = d.view.end - d.view.start
        setView(clampView({ start: centre - w / 2, end: centre + w / 2 }))
      }
    }
    window.addEventListener('pointerup', up)
    window.addEventListener('pointermove', move)
    return () => {
      window.removeEventListener('pointerup', up)
      window.removeEventListener('pointermove', move)
    }
  }, [clampView, miniTimeAt, onSeek, timeAt])

  const hoverFrame = useMemo(() => {
    if (!hover || !keyframes.length) return null
    return keyframes.reduce((best, kf) =>
      Math.abs(kf.timestamp - hover.t) < Math.abs(best.timestamp - hover.t) ? kf : best)
  }, [hover, keyframes])

  const fit = () => setView({ start: 0, end: Math.max(duration, 1) })

  return (
    <div className="select-none" ref={wrapRef}>
      <div className="mb-1.5 flex items-center gap-1">
        <Tip label="Zoom out (scroll down on the ribbon)">
          <Button size="sm" icon onClick={() => zoomAt(1.6, 0.5)} disabled={!zoomed} aria-label="Zoom out">
            <span className="mono text-[13px] leading-none">-</span>
          </Button>
        </Tip>
        <Tip label="Zoom in (scroll up on the ribbon)">
          <Button size="sm" icon onClick={() => zoomAt(1 / 1.6, 0.5)} aria-label="Zoom in">
            <span className="mono text-[13px] leading-none">+</span>
          </Button>
        </Tip>
        <Tip label="Fit the whole recording">
          <Button size="sm" onClick={fit} disabled={!zoomed}>Fit</Button>
        </Tip>
        <span className="mono ml-1 text-[10.5px] text-[var(--color-fg-4)]">
          {fmtTime(view.start)} - {fmtTime(view.end)}
          {zoomed && <span className="ml-1.5">({(duration / span).toFixed(1)}x)</span>}
        </span>
        <span className="ml-auto text-[10.5px] text-[var(--color-fg-4)]">
          {zoomed ? 'drag to pan - shift-drag to scrub' : 'drag to scrub - scroll to zoom'}
        </span>
      </div>

      <canvas
        ref={canvasRef}
        className={`w-full ${zoomed ? 'cursor-grab active:cursor-grabbing' : 'cursor-crosshair'}`}
        style={{ height: TOTAL_H }}
        onPointerDown={(e) => {
          const rect = canvasRef.current!.getBoundingClientRect()
          const localY = e.clientY - rect.top
          const stripTop = H.scenes + H.gap
          if (onOpenFrame && hoverFrame && localY >= stripTop && localY <= stripTop + H.strip && !zoomed) {
            onOpenFrame(hoverFrame.id)
            return
          }
          // When zoomed, dragging pans by default and shift scrubs; unzoomed
          // there is nowhere to pan to, so dragging always scrubs.
          const mode = zoomed && !e.shiftKey ? 'pan' : 'seek'
          drag.current = { mode, startX: e.clientX, view }
          if (mode === 'seek') onSeek(timeAt(e.clientX))
        }}
        onMouseMove={(e) => {
          const rect = canvasRef.current!.getBoundingClientRect()
          setHover({ x: e.clientX - rect.left, t: timeAt(e.clientX) })
        }}
        onMouseLeave={() => setHover(null)}
        onDoubleClick={fit}
        role="slider"
        aria-label="Video timeline"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(currentTime)}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'ArrowRight') onSeek(Math.min(duration, currentTime + (e.shiftKey ? 10 : 2)))
          else if (e.key === 'ArrowLeft') onSeek(Math.max(0, currentTime - (e.shiftKey ? 10 : 2)))
          else if (e.key === '+' || e.key === '=') zoomAt(1 / 1.6, 0.5)
          else if (e.key === '-') zoomAt(1.6, 0.5)
          else if (e.key === '0') fit()
        }}
      />

      {hover && (
        <div
          className="pointer-events-none absolute z-20 -translate-x-1/2"
          style={{ left: Math.max(56, Math.min(hover.x, width - 56)), top: 22, transform: 'translate(-50%, -100%)' }}
        >
          <div className="surface-raised overflow-hidden p-0">
            {hoverFrame && <FrameImage path={hoverFrame.path} className="block h-[62px] w-[110px] object-cover" />}
            <div className="mono px-2 py-1 text-[10.5px] text-[var(--color-fg-2)]">{fmtTime(hover.t)}</div>
          </div>
        </div>
      )}

      <canvas
        ref={miniRef}
        className="mt-1 w-full cursor-pointer"
        style={{ height: MINIMAP_H }}
        aria-label="Timeline overview"
        onPointerDown={(e) => {
          const centre = miniTimeAt(e.clientX)
          const w = span
          drag.current = { mode: 'window', startX: e.clientX, view }
          setView(clampView({ start: centre - w / 2, end: centre + w / 2 }))
        }}
      />

      <div className="mt-1.5 flex flex-wrap items-center gap-x-3.5 gap-y-1 px-2 text-[10.5px] text-[var(--color-fg-4)]">
        <Legend color="rgba(255,255,255,0.24)" label="scene" />
        <Legend color="rgba(147,204,126,0.6)" label="slide" />
        <Legend color="rgba(187,154,247,0.7)" label="visual change" />
        <Legend color="rgba(122,162,247,0.85)" label="speech" />
        <Legend color="rgba(224,175,104,0.9)" label="evidence" />
        <span className="ml-auto inline-flex items-center gap-1">
          <Icon.Search /> click a frame to open it
        </span>
      </div>
    </div>
  )
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-[6px] w-[6px] rounded-[2px]" style={{ background: color }} />
      {label}
    </span>
  )
}

const SPEAKER_HUES = [188, 268, 42, 150, 330, 20, 210, 96]
function speakerHue(speaker: string): string {
  let hash = 0
  for (let i = 0; i < speaker.length; i++) hash = (hash * 31 + speaker.charCodeAt(i)) >>> 0
  return `hsl(${SPEAKER_HUES[hash % SPEAKER_HUES.length]} 78% 66%)`
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.roundRect(x, y, w, h, Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2))
}
