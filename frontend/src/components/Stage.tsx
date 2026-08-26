/** Video stage: the player and the transcript, driven by one shared clock. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../lib/api'
import { copyText, downloadBlob, slug } from '../lib/download'
import { fmtTime, speakerColor } from '../lib/format'
import { useStore } from '../lib/store'
import type { Segment } from '../lib/types'
import { Button, CopyButton, Empty, Icon, Menu, PanelHeader, Segmented, Tip } from './ui'

const SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 2]

export function Player({ videoId, poster, fps = 25, height, collapsed = false, onToggleCollapse }: {
  videoId: string
  poster?: string | null
  fps?: number
  /** Height of the picture area in pixels. Undefined keeps the 16:9 box. */
  height?: number
  /** Hide the picture but keep playback and the controls. */
  collapsed?: boolean
  onToggleCollapse?: () => void
}) {
  const ref = useRef<HTMLVideoElement>(null)
  const { currentTime, duration, playing, seekRequest, setTime, setDuration, setPlaying, seek } = useStore()
  const [rate, setRate] = useState(1)
  const [muted, setMuted] = useState(false)
  const [volume, setVolume] = useState(1)
  const [failed, setFailed] = useState(false)
  /** Optional A->B loop, for studying a moment repeatedly. */
  const [loop, setLoop] = useState<{ a: number; b: number } | null>(null)

  // Seeks are requested through the store so the ribbon, transcript, tables
  // and citations all drive one element without prop-drilling a ref.
  useEffect(() => {
    const el = ref.current
    if (!el || !seekRequest) return
    if (Math.abs(el.currentTime - seekRequest.t) > 0.08) el.currentTime = seekRequest.t
  }, [seekRequest])

  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (playing && el.paused) el.play().catch(() => setPlaying(false))
    if (!playing && !el.paused) el.pause()
  }, [playing, setPlaying])

  useEffect(() => {
    const el = ref.current
    if (el) el.volume = volume
  }, [volume])

  const onTime = useCallback(
    (t: number) => {
      setTime(t)
      if (loop && (t >= loop.b || t < loop.a - 0.25)) seek(loop.a)
    },
    [loop, seek, setTime],
  )

  const step = (frames: number) => {
    const el = ref.current
    if (!el) return
    setPlaying(false)
    seek(Math.max(0, Math.min(el.duration || 0, el.currentTime + frames / (fps || 25))))
  }

  return (
    <div className="surface flex shrink-0 flex-col overflow-hidden">
      <div
        className="stage-box relative w-full shrink-0 bg-black"
        data-collapsed={collapsed}
        style={height !== undefined ? ({ '--stage-h': `${height}px` } as React.CSSProperties) : undefined}
      >
        {failed ? (
          <div className="flex h-full flex-col items-center justify-center gap-1.5 px-6 text-center">
            <Icon.Warn className="text-[var(--color-caution)]" />
            <span className="text-[13px] text-[var(--color-fg-2)]">This container can't be decoded by the browser.</span>
            <span className="text-[11.5px] text-[var(--color-fg-3)]">
              Analysis is unaffected, the engine decodes it server-side.
            </span>
          </div>
        ) : (
          <video
            ref={ref}
            src={api.mediaUrl(videoId)}
            className="h-full w-full object-contain"
            poster={poster ? api.posterUrl(poster) : undefined}
            preload="metadata"
            muted={muted}
            playsInline
            onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
            onTimeUpdate={(e) => onTime(e.currentTarget.currentTime)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onError={() => setFailed(true)}
            onClick={() => setPlaying(!playing)}
          />
        )}

        {loop && (
          <div className="absolute left-3 top-3 flex items-center gap-2 rounded-[var(--radius)] border border-[var(--color-line)] bg-[rgba(10,10,11,0.85)] px-2 py-1 text-[11px]">
            <span className="mono text-[var(--color-fg-2)]">loop {fmtTime(loop.a)}-{fmtTime(loop.b)}</span>
            <button className="text-[var(--color-fg-3)] hover:text-[var(--color-fg)]" onClick={() => setLoop(null)} aria-label="Clear loop">
              <Icon.Close />
            </button>
          </div>
        )}
      </div>

      {/* scrubber */}
      <div className="px-3 pt-2.5">
        <input
          type="range"
          min={0}
          max={Math.max(duration, 0.1)}
          step={0.05}
          value={Math.min(currentTime, duration || 0)}
          onChange={(e) => seek(Number(e.target.value))}
          aria-label="Seek"
          className="h-1 w-full cursor-pointer appearance-none rounded-full bg-[var(--color-line)] accent-[var(--color-fg)] [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--color-fg)]"
        />
      </div>

      <div className="flex flex-wrap items-center gap-1.5 px-3 py-2.5">
        {onToggleCollapse && (
          <Tip label={collapsed ? 'Show the video (v)' : 'Hide the video, keep playing (v)'}>
            <Button size="sm" icon onClick={onToggleCollapse} aria-label={collapsed ? 'Show video' : 'Hide video'}>
              {collapsed ? <Icon.Eye /> : <Icon.EyeOff />}
            </Button>
          </Tip>
        )}
        <Tip label={playing ? 'Pause (space)' : 'Play (space)'}>
          <Button icon onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause' : 'Play'}>
            {playing ? <Icon.Pause /> : <Icon.Play />}
          </Button>
        </Tip>
        <Tip label="Previous frame">
          <Button size="sm" icon onClick={() => step(-1)} aria-label="Previous frame">
            <Icon.StepBack />
          </Button>
        </Tip>
        <Tip label="Next frame">
          <Button size="sm" icon onClick={() => step(1)} aria-label="Next frame">
            <Icon.StepForward />
          </Button>
        </Tip>

        <span className="mono ml-1 text-[12px] text-[var(--color-fg-2)]">
          {fmtTime(currentTime)}
          <span className="text-[var(--color-fg-4)]"> / {fmtTime(duration)}</span>
        </span>

        <div className="ml-auto flex items-center gap-1.5">
          <Tip label={loop ? 'Clear the A-B loop' : 'Loop the next 10 seconds from here'}>
            <Button
              size="sm"
              onClick={() => setLoop(loop ? null : { a: currentTime, b: Math.min(duration, currentTime + 10) })}
            >
              {loop ? 'looping' : 'loop'}
            </Button>
          </Tip>

          <Menu
            width={130}
            trigger={({ toggle }) => (
              <Button size="sm" onClick={toggle} className="mono">{rate}x</Button>
            )}
            items={SPEEDS.map((s) => ({
              label: `${s}x speed`,
              onSelect: () => { setRate(s); if (ref.current) ref.current.playbackRate = s },
            }))}
          />

          <Tip label={muted ? 'Unmute (m)' : 'Mute (m)'}>
            <Button size="sm" icon onClick={() => setMuted(!muted)} aria-label={muted ? 'Unmute' : 'Mute'}>
              {muted ? <Icon.Muted /> : <Icon.Volume />}
            </Button>
          </Tip>
          <input
            type="range" min={0} max={1} step={0.05} value={volume}
            onChange={(e) => setVolume(Number(e.target.value))}
            aria-label="Volume"
            className="h-1 w-14 cursor-pointer appearance-none rounded-full bg-[var(--color-line)] accent-[var(--color-fg)] [&::-webkit-slider-thumb]:h-2.5 [&::-webkit-slider-thumb]:w-2.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--color-fg-2)]"
          />
          <Tip label="Full screen">
            <Button size="sm" icon onClick={() => ref.current?.requestFullscreen?.()} aria-label="Full screen">
              <Icon.Expand />
            </Button>
          </Tip>
        </div>
      </div>
    </div>
  )
}

/* --------------------------------------------------------------- transcript */

export function Transcript({ segments, title }: { segments: Segment[]; title: string }) {
  const { currentTime, seek } = useStore()
  const [filter, setFilter] = useState('')
  const [speaker, setSpeaker] = useState<string>('all')
  const [follow, setFollow] = useState(true)
  const activeRef = useRef<HTMLDivElement>(null)

  const speakers = useMemo(() => {
    const seen: string[] = []
    for (const s of segments) if (s.speaker && !seen.includes(s.speaker)) seen.push(s.speaker)
    return seen.sort()
  }, [segments])

  // Segments are time-sorted, so a binary search keeps this O(log n) per tick.
  const activeIndex = useMemo(() => {
    let lo = 0
    let hi = segments.length - 1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      const s = segments[mid]
      if (currentTime < s.start) hi = mid - 1
      else if (currentTime >= s.end) lo = mid + 1
      else return mid
    }
    return -1
  }, [segments, currentTime])

  useEffect(() => {
    if (follow && activeIndex >= 0) activeRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [activeIndex, follow])

  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase()
    return segments.filter(
      (s) =>
        (speaker === 'all' || s.speaker === speaker) &&
        (!q || s.text.toLowerCase().includes(q)),
    )
  }, [segments, filter, speaker])

  const plainText = useMemo(
    () => visible.map((s) => `[${fmtTime(s.start)}] ${s.speaker ? `${s.speaker}: ` : ''}${s.text}`).join('\n'),
    [visible],
  )

  /*
    Windowed rendering. A two-hour recording is roughly 1,700 segments, and
    mounting them all costs about 10,000 DOM nodes, which makes both the
    initial paint and every follow-scroll janky. Only the rows near the
    viewport are mounted; spacers above and below preserve the scrollbar.

    Row height is an estimate, so a long line can drift slightly. The overscan
    absorbs that, and the follow-scroll uses scrollIntoView on the mounted row
    rather than arithmetic, so the active line always lands correctly.
  */
  const scrollRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportH, setViewportH] = useState(600)
  const ROW_H = 52
  const OVERSCAN = 12
  const virtualise = visible.length > 120

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setViewportH(el.clientHeight))
    ro.observe(el)
    setViewportH(el.clientHeight)
    return () => ro.disconnect()
  }, [])

  const windowRange = useMemo(() => {
    if (!virtualise) return { from: 0, to: visible.length, padTop: 0, padBottom: 0 }
    const from = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN)
    const to = Math.min(visible.length, Math.ceil((scrollTop + viewportH) / ROW_H) + OVERSCAN)
    return { from, to, padTop: from * ROW_H, padBottom: Math.max(0, (visible.length - to) * ROW_H) }
  }, [virtualise, scrollTop, viewportH, visible.length])

  // Following playback must be able to reach a row that is not mounted, so jump
  // the scroll position first and let the effect above mount the right window.
  useEffect(() => {
    if (!follow || activeIndex < 0 || !virtualise) return
    const pos = visible.findIndex((s) => s.index === segments[activeIndex]?.index)
    if (pos < 0) return
    const el = scrollRef.current
    if (!el) return
    const target = pos * ROW_H - el.clientHeight / 2
    if (Math.abs(el.scrollTop - target) > el.clientHeight * 0.4) el.scrollTop = Math.max(0, target)
  }, [follow, activeIndex, virtualise, visible, segments])

  return (
    <div className="surface flex min-h-0 flex-1 flex-col overflow-hidden">
      <PanelHeader title="Transcript" count={visible.length}>
        <Tip label="Copy what's shown">
          <Button variant="quiet" size="sm" icon onClick={() => copyText(plainText, 'Transcript copied')} aria-label="Copy transcript">
            <Icon.Copy />
          </Button>
        </Tip>
        <Menu
          width={190}
          trigger={({ toggle }) => (
            <Tip label="Download transcript">
              <Button variant="quiet" size="sm" icon onClick={toggle} aria-label="Download transcript"><Icon.Download /></Button>
            </Tip>
          )}
          items={[
            { label: 'Filtered view', hint: '.txt', onSelect: () => downloadBlob(`${slug(title)}-transcript.txt`, plainText, 'text/plain') },
            { label: 'Filtered view', hint: '.json', onSelect: () => downloadBlob(`${slug(title)}-transcript.json`, JSON.stringify(visible, null, 2), 'application/json') },
          ]}
        />
        <Tip label="Keep the active line centred">
          <button
            onClick={() => setFollow((f) => !f)}
            data-active={follow}
            className="chip data-[active=true]:border-[var(--color-line-strong)] data-[active=true]:text-[var(--color-fg)]"
          >
            follow
          </button>
        </Tip>
      </PanelHeader>

      <div className="flex shrink-0 items-center gap-1.5 border-b border-[var(--color-line-soft)] px-3 py-2">
        <div className="relative flex-1">
          <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--color-fg-4)]">
            <Icon.Search />
          </span>
          <input
            data-transcript-filter
            className="field !h-7 !pl-8 !text-[12.5px]"
            placeholder="Filter lines..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            aria-label="Filter transcript"
          />
        </div>
        {speakers.length > 1 && (
          <Segmented
            value={speaker}
            onChange={setSpeaker}
            options={[
              { value: 'all', label: 'All' },
              ...speakers.map((s) => ({ value: s, label: s.replace('SPEAKER_', 'S') })),
            ]}
          />
        )}
      </div>

      <div
        ref={scrollRef}
        onScroll={(e) => virtualise && setScrollTop(e.currentTarget.scrollTop)}
        className="min-h-0 flex-1 overflow-y-auto px-1.5 py-1.5"
      >
        {virtualise && <div style={{ height: windowRange.padTop }} aria-hidden />}
        {!visible.length && (
          <Empty
            title={segments.length ? 'No matching lines' : 'No speech transcribed'}
            body={segments.length ? undefined : 'Upload a caption file with the video, or install faster-whisper.'}
          />
        )}
        {visible.slice(windowRange.from, windowRange.to).map((s) => {
          const isActive = activeIndex >= 0 && segments[activeIndex].index === s.index
          const color = speakerColor(s.speaker)
          return (
            <div
              key={s.index}
              ref={isActive ? activeRef : undefined}
              className={`group flex gap-2 rounded-[var(--radius)] px-2 py-1.5 transition-colors ${
                isActive ? 'bg-[var(--color-surface-3)]' : 'hover:bg-[var(--color-surface-2)]'
              }`}
            >
              <button
                onClick={() => seek(s.start)}
                className="mono shrink-0 pt-[1px] text-[11px] text-[var(--color-fg-4)] transition-colors hover:text-[var(--color-fg)]"
                title="Jump here"
              >
                {fmtTime(s.start)}
              </button>
              <span className="mt-[5px] h-[13px] w-[2px] shrink-0 rounded-full" style={{ background: color, opacity: isActive ? 1 : 0.55 }} />
              <div className="min-w-0 flex-1">
                {s.speaker && (
                  <span className="mono mr-1.5 text-[11px]" style={{ color }}>{s.speaker.replace('SPEAKER_', 'S')}</span>
                )}
                <span className={`text-[13px] leading-[1.55] ${isActive ? 'text-[var(--color-fg)]' : 'text-[var(--color-fg-2)]'}`}>
                  {highlight(s.text, filter)}
                </span>
              </div>
              <div className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100">
                <CopyButton text={s.text} what="Line copied" />
              </div>
            </div>
          )
        })}
        {virtualise && <div style={{ height: windowRange.padBottom }} aria-hidden />}
      </div>
    </div>
  )
}

function highlight(text: string, query: string) {
  const q = query.trim()
  if (!q) return text
  const idx = text.toLowerCase().indexOf(q.toLowerCase())
  if (idx < 0) return text
  return (
    <>
      {text.slice(0, idx)}
      <mark className="rounded-[3px] bg-[rgba(224,175,104,0.25)] px-0.5 text-[var(--color-fg)]">
        {text.slice(idx, idx + q.length)}
      </mark>
      {text.slice(idx + q.length)}
    </>
  )
}
