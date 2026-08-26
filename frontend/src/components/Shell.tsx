/** Top bar, capability panel, and the studio layout. */

import { useCallback, useEffect, useRef, useState } from 'react'
import { usePaneSize } from '../lib/panes'
import { api } from '../lib/api'
import { copyText } from '../lib/download'
import { fmtDuration, fmtTime, relativeTime } from '../lib/format'
import { prettyCombo } from '../lib/shortcuts'
import { useStore } from '../lib/store'
import { toast } from '../lib/toast'
import { Ask } from './Ask'
import { ChapterRail } from './ChapterRail'
import { ChronoRibbon } from './ChronoRibbon'
import { DataTables, ExportMenu } from './DataTables'
import { useLightbox } from '../hooks/useLightbox'
import { Player, Transcript } from './Stage'
import { ErrorBoundary } from './ErrorBoundary'
import { Button, Icon, Kbd, Menu, Segmented, Spinner, Stat, Surface, Tip } from './ui'
import type { MenuItem } from './ui'

interface TopBarProps {
  capabilitiesOpen: boolean
  onToggleCapabilities: () => void
  onOpenPalette: () => void
  onOpenHelp: () => void
}

/** Past conversations, resumable. Kept in the bar rather than the studio so a
 *  thread survives moving between videos. */
function Threads() {
  const { sessions, sessionId, loadSessions, resumeSession, forgetSession, newThread } = useStore()

  const items: MenuItem[] = [
    { label: 'New thread', hint: 'start fresh', onSelect: () => newThread() },
    ...(sessions.length ? [{ separator: true, label: '' }] : []),
    ...sessions.slice(0, 8).map((s) => ({
      label: s.title || 'Untitled thread',
      // Threads are titled by their opening question, so two that begin the
      // same way are told apart by when they were last touched.
      hint: [`${s.turn_count} turn${s.turn_count === 1 ? '' : 's'}`, relativeTime(s.updated_at)]
        .filter(Boolean)
        .join(' \u00b7 '),
      onSelect: () => {
        resumeSession(s.id)
          .then((opened) => {
            if (!opened) {
              toast.info('Thread restored', 'Open a video to follow its citations.')
            }
          })
          .catch(() => toast.error('That thread could not be opened.'))
      },
    })),
    ...(sessionId
      ? [
          { separator: true, label: '' },
          {
            label: 'Delete this thread',
            danger: true,
            onSelect: () => {
              forgetSession(sessionId).catch(() => toast.error('That thread could not be deleted.'))
            },
          },
        ]
      : []),
  ]

  return (
    <Menu
      width={260}
      trigger={({ open, toggle }) => (
        <Tip label="Conversations, resume or start a new one">
          <Button
            size="sm"
            onClick={() => { if (!open) loadSessions(); toggle() }}
            aria-expanded={open}
            className="hidden sm:inline-flex"
          >
            <Icon.Ask />
            <span className="hidden md:inline">{sessionId ? 'Thread' : 'Threads'}</span>
          </Button>
        </Tip>
      )}
      items={items}
    />
  )
}

export function TopBar({ capabilitiesOpen, onToggleCapabilities, onOpenPalette, onOpenHelp }: TopBarProps) {
  const { view, select, health, healthReachable, loadHealth, timeline } = useStore()

  useEffect(() => {
    loadHealth()
    const id = setInterval(loadHealth, 30_000)
    return () => clearInterval(id)
  }, [loadHealth])

  const degraded = health?.degraded?.length ?? 0
  const offline = !healthReachable

  return (
    <>
      <header className="relative z-30 flex h-12 shrink-0 items-center gap-2 border-b border-[var(--color-line-soft)] bg-[var(--color-surface)] px-3">
        {view === 'studio' && (
          <Tip label="Back to the library (g)">
            <Button variant="quiet" size="sm" icon onClick={() => select(null)} aria-label="Back"><Icon.Back /></Button>
          </Tip>
        )}

        <div className="flex items-center gap-2">
          <svg viewBox="0 0 16 16" className="h-4 w-4 text-[var(--color-fg)]" aria-hidden>
            <circle cx="8" cy="8" r="6.4" fill="none" stroke="currentColor" strokeWidth="1.3" opacity="0.5" />
            <path d="M8 4.2V8l2.8 1.6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          <span className="text-[13.5px] font-semibold tracking-[-0.02em]">Chronoscope</span>
        </div>

        {view === 'studio' && timeline && (
          <div className="ml-2 hidden min-w-0 items-center gap-2 md:flex">
            <span className="h-3.5 w-px bg-[var(--color-line)]" />
            <span className="truncate text-[12.5px] text-[var(--color-fg-2)]">{timeline.video.title}</span>
            <span className="chip">{fmtDuration(timeline.video.duration)}</span>
          </div>
        )}

        <div className="flex-1" />

        <Threads />

        <Tip label="Search videos and commands">
          <Button size="sm" onClick={onOpenPalette} className="hidden sm:inline-flex">
            <Icon.Search />
            <span className="hidden text-[var(--color-fg-3)] md:inline">Search</span>
            <Kbd>{prettyCombo('mod+k')}</Kbd>
          </Button>
        </Tip>

        <Tip label="Keyboard shortcuts">
          <Button variant="quiet" size="sm" icon onClick={onOpenHelp} aria-label="Keyboard shortcuts">
            <span className="mono text-[12px]">?</span>
          </Button>
        </Tip>

        <Tip
          label={
            offline ? 'The API is not responding'
            : degraded ? `${degraded} components running at reduced fidelity`
            : 'All components at full fidelity'
          }
        >
          <Button size="sm" onClick={onToggleCapabilities} aria-expanded={capabilitiesOpen}>
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{
                background: offline ? 'var(--color-critical)'
                  : degraded ? 'var(--color-caution)'
                  : 'var(--color-positive)',
              }}
            />
            <span className="hidden sm:inline">
              {offline ? 'offline' : health ? (degraded ? `${degraded} degraded` : 'nominal') : '...'}
            </span>
          </Button>
        </Tip>
      </header>

      {capabilitiesOpen && <CapabilityPanel onClose={onToggleCapabilities} />}
    </>
  )
}

function CapabilityPanel({ onClose }: { onClose: () => void }) {
  const health = useStore((s) => s.health)
  const [stats, setStats] = useState<Record<string, any> | null>(null)

  useEffect(() => { api.stats().then(setStats).catch(() => null) }, [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-40 flex justify-end" onClick={onClose}>
      <div className="absolute inset-0 bg-[rgba(6,6,7,0.6)] animate-fade-in" />
      <aside
        className="surface-raised relative m-2 flex w-[380px] max-w-[92vw] flex-col overflow-hidden animate-rise"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Engine capabilities"
      >
        <div className="flex h-11 shrink-0 items-center justify-between border-b border-[var(--color-line-soft)] px-3.5">
          <h3 className="label">Engine capabilities</h3>
          <div className="flex gap-1">
            <Tip label="Copy diagnostics">
              <Button
                variant="quiet" size="sm" icon aria-label="Copy diagnostics"
                onClick={() => copyText(JSON.stringify({ health, stats }, null, 2), 'Diagnostics copied')}
              >
                <Icon.Copy />
              </Button>
            </Tip>
            <Button variant="quiet" size="sm" icon onClick={onClose} aria-label="Close"><Icon.Close /></Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-3.5 text-[12.5px]">
          {!health ? (
            <div className="flex justify-center py-8 text-[var(--color-fg-3)]"><Spinner size={18} /></div>
          ) : (
            <div className="flex flex-col gap-3.5">
              {health.degraded.length > 0 && (
                <div className="rounded-[var(--radius)] border border-[rgba(224,175,104,0.3)] bg-[rgba(224,175,104,0.06)] p-2.5">
                  <div className="mb-1.5 flex items-center gap-1.5 text-[12px] font-medium text-[var(--color-caution)]">
                    <Icon.Warn /> Reduced capability
                  </div>
                  <ul className="ml-3 flex list-outside list-disc flex-col gap-1 text-[11.5px] leading-relaxed text-[var(--color-fg-2)]">
                    {health.degraded.map((d) => <li key={d}>{d}</li>)}
                  </ul>
                  <p className="mt-2 text-[11px] text-[var(--color-fg-4)]">
                    Every stage still runs, results come from the deterministic fallbacks and are labelled as such.
                  </p>
                </div>
              )}

              <Group title="Encoders">
                <Row label="Text" value={health.encoders.text?.name} warn={health.encoders.text?.degraded} />
                <Row label="Vision" value={health.encoders.image?.name} warn={health.encoders.image?.degraded} />
                <Row label="Dimensions" value={`${health.encoders.text?.dim} / ${health.encoders.image?.dim}`} />
              </Group>

              <Group title="Vector store">
                <Row label="Backend" value={health.vector_store?.backend} warn={!health.vector_store?.ok} />
                {Object.entries(health.vector_store?.collections ?? {}).map(([name, info]: [string, any]) => (
                  <Row key={name} label={name} value={String(info.points ?? info.size ?? JSON.stringify(info))} />
                ))}
              </Group>

              <Group title="Language models">
                <Row label="Chain" value={health.llm.chain.join(' -> ') || 'none'} warn={!health.llm.any_available} />
                {health.llm.providers.map((p: any) => (
                  <Row key={p.name} label={p.name} value={`${p.available ? 'available' : 'unreachable'} · ${p.state}`} warn={!p.available} />
                ))}
              </Group>

              <Group title="Index & jobs">
                <Row label="BM25 documents" value={health.lexical.documents} />
                <Row label="BM25 terms" value={health.lexical.terms} />
                <Row label="Queued jobs" value={health.jobs?.queued ?? 0} />
                {stats?.query_cache && <Row label="Query cache hits" value={`${((stats.query_cache.hit_rate ?? 0) * 100).toFixed(0)}%`} />}
                {stats?.uptime_s !== undefined && <Row label="Uptime" value={fmtDuration(stats.uptime_s)} />}
              </Group>

              <p className="text-[11px] leading-relaxed text-[var(--color-fg-4)]">
                v{health.version} · {health.env}. Install <span className="mono">sentence-transformers</span>,{' '}
                <span className="mono">open_clip_torch</span> and <span className="mono">faster-whisper</span> for full
                fidelity; run <span className="mono">ollama</span> locally for generated answers.
              </p>
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="label mb-1.5">{title}</h4>
      <div className="flex flex-col gap-1 rounded-[var(--radius)] border border-[var(--color-line-soft)] p-2.5">{children}</div>
    </div>
  )
}

function Row({ label, value, warn }: { label: string; value: React.ReactNode; warn?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-[var(--color-fg-3)]">{label}</span>
      <span className={`mono truncate text-[11.5px] ${warn ? 'text-[var(--color-caution)]' : 'text-[var(--color-fg-2)]'}`}>
        {value ?? '-'}
      </span>
    </div>
  )
}

/* ------------------------------------------------------------------ studio */

type Pane = 'ask' | 'transcript' | 'data'

export function Studio() {
  const { timeline, loadingTimeline, selectedId, currentTime, seek, answer } = useStore()
  const [scoped, setScoped] = useState(true)
  const [pane, setPane] = useState<Pane>('ask')
  const [videoHidden, setVideoHidden] = useState(false)
  const { open: openFrame, node: lightbox } = useLightbox()
  const split = useSplit()

  // The picture area is user-resizable. Capped at 58% of the viewport so the
  // ribbon and the panels below it always keep room.
  const stage = usePaneSize('chronoscope:stage', {
    fallback: 0.44,
    min: 0.12,
    max: 0.6,
    container: split.containerRef,
  })

  useEffect(() => {
    const toggle = () => setVideoHidden((v) => !v)
    window.addEventListener('chronoscope:toggle-video', toggle)
    return () => window.removeEventListener('chronoscope:toggle-video', toggle)
  }, [])

  if (loadingTimeline || !timeline || !selectedId) {
    return (
      <div className="flex flex-1 items-center justify-center text-[var(--color-fg-3)]"><Spinner size={20} /></div>
    )
  }

  const { video, scenes, keyframes, chunks, segments, chapters } = timeline
  const stats = video.stats ?? {}
  const activeScene = scenes.find((s) => currentTime >= s.span.start && currentTime < s.span.end)

  return (
    <>
      <div ref={split.containerRef} className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-2 lg:flex-row lg:overflow-hidden">
        {/* -------------------------------------------------------- left */}
        <div
          className="split-left flex flex-col gap-2 lg:min-h-0 lg:pr-0.5"
          style={split.leftStyle}
        >
          <Player
            videoId={selectedId}
            poster={video.poster}
            fps={video.fps}
            height={videoHidden ? undefined : stage.size}
            collapsed={videoHidden}
            onToggleCollapse={() => setVideoHidden((v) => !v)}
          />

          {/* Horizontal grip: drag to trade picture height against everything
              below it. Double-click restores the default. */}
          {!videoHidden && (
            <div
              role="separator"
              aria-orientation="horizontal"
              aria-label="Resize the video"
              onPointerDown={(e) => stage.onGrab(e, 'y')}
              onDoubleClick={stage.reset}
              title="Drag to resize the video, double-click to reset"
              className="group -my-1 hidden h-2 shrink-0 cursor-row-resize items-center justify-center lg:flex"
            >
              <span className="h-[3px] w-10 rounded-full bg-[var(--color-line)] transition-colors group-hover:bg-[var(--color-line-strong)]" />
            </div>
          )}

          {chapters.length > 1 && (
            <Surface className="shrink-0 p-2.5">
              <ChapterRail chapters={chapters} duration={video.duration} currentTime={currentTime} onSeek={seek} />
            </Surface>
          )}

          <Surface className="relative shrink-0 px-2 py-2">
            <ChronoRibbon
              duration={video.duration}
              scenes={scenes}
              keyframes={keyframes}
              chunks={chunks}
              segments={segments}
              citations={answer?.citations ?? []}
              currentTime={currentTime}
              onSeek={seek}
              onOpenFrame={(id) => openFrame(keyframes, id)}
            />
          </Surface>

          <div className="flex min-h-0 flex-col gap-2 lg:overflow-y-auto">
          <Surface className="grid shrink-0 grid-cols-2 gap-y-3 p-3 sm:grid-cols-4">
            <Stat label="Scenes" value={scenes.length} hint="Detected cuts, points where the picture changes substantially" />
            <Stat label="Keyframes" value={`${keyframes.length}${stats.slides ? ` · ${stats.slides} slide` : ''}`} hint="Frames kept for visual search: chosen for sharpness and spread, then de-duplicated" />
            <Stat label="Chunks" value={chunks.length} hint="Searchable units, a stretch of speech plus the frames on screen while it was said" />
            <Stat label="Speakers" value={video.speakers.length || '-'} hint={video.speakers.length ? `Distinct voices: ${video.speakers.join(', ')}` : 'No diarisation available'} />
            <Stat label="Words" value={stats.transcript?.words ?? '-'} hint="Total words transcribed" />
            <Stat label="Speech" value={stats.transcript?.speech_seconds ? fmtDuration(stats.transcript.speech_seconds) : '-'} hint="Time containing speech, excluding silence" />
            <Stat label="Source" value={stats.transcript?.source ?? '-'} hint="Where the transcript came from: an uploaded caption file, Whisper, or voice-activity detection only" />
            <Stat
              label="At playhead"
              value={activeScene ? `scene ${activeScene.index + 1}` : '-'}
              hint={activeScene ? `${activeScene.kind} · ${fmtTime(activeScene.span.start)}-${fmtTime(activeScene.span.end)}` : ''}
            />
          </Surface>

          {video.topics.length > 0 && (
            <Surface className="shrink-0 p-3">
              <h4 className="label mb-1.5">Topics</h4>
              <div className="flex flex-wrap gap-1">
                {video.topics.map((t) => <span key={t} className="chip">{t}</span>)}
              </div>
            </Surface>
          )}
          </div>
        </div>

        {/* ------------------------------------------------------- handle */}
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize panels"
          onPointerDown={split.onGrab}
          onDoubleClick={split.reset}
          title="Drag to resize · double-click to reset"
          className="group hidden w-1.5 shrink-0 cursor-col-resize items-center justify-center lg:flex"
        >
          <span className="h-10 w-[3px] rounded-full bg-[var(--color-line)] transition-colors group-hover:bg-[var(--color-line-strong)]" />
        </div>

        {/* ------------------------------------------------------- right */}
        <div className="flex min-h-[520px] flex-1 flex-col gap-2 lg:min-h-0 lg:overflow-hidden">
          <div className="flex shrink-0 items-center gap-2">
            <Segmented
              value={pane}
              onChange={setPane}
              options={[
                { value: 'ask', label: 'Ask', title: 'Question the footage' },
                { value: 'transcript', label: `Transcript`, title: 'Every spoken line, synced to playback' },
                { value: 'data', label: 'Data', title: 'Scenes, chunks and keyframes as sortable tables' },
              ]}
            />
            {pane === 'ask' && (
              <Tip label="Restrict retrieval to this video, or search the whole library">
                <label className="ml-auto flex cursor-pointer select-none items-center gap-1.5 text-[11.5px] text-[var(--color-fg-3)]">
                  <input
                    type="checkbox"
                    checked={scoped}
                    onChange={(e) => setScoped(e.target.checked)}
                    className="accent-[var(--color-fg-2)]"
                  />
                  this video only
                </label>
              </Tip>
            )}
            {pane !== 'ask' && pane !== 'data' && (
              <div className="ml-auto">
                <ExportMenu videoId={selectedId} title={video.title} scenes={scenes} chunks={chunks} keyframes={keyframes} segments={segments} />
              </div>
            )}
          </div>

          {/* All panes stay mounted: switching must not restart a running query
              or lose the transcript's scroll position. */}
          <div className={`min-h-0 flex-1 flex-col ${pane === 'ask' ? 'flex' : 'hidden'}`}>
            <ErrorBoundary label="The question panel stopped responding">
              <Ask videoId={selectedId} scoped={scoped} onOpenFrame={openFrame} />
            </ErrorBoundary>
          </div>
          <div className={`min-h-0 flex-1 flex-col ${pane === 'transcript' ? 'flex' : 'hidden'}`}>
            <ErrorBoundary label="The transcript stopped responding">
              <Transcript segments={segments} title={video.title} />
            </ErrorBoundary>
          </div>
          <div className={`min-h-0 flex-1 flex-col ${pane === 'data' ? 'flex' : 'hidden'}`}>
            <Surface className="flex min-h-0 flex-1 flex-col overflow-hidden">
              <DataTables
                scenes={scenes}
                chunks={chunks}
                keyframes={keyframes}
                segments={segments}
                videoId={selectedId}
                title={video.title}
                onOpenFrame={openFrame}
              />
            </Surface>
          </div>
        </div>
      </div>
      {lightbox}
    </>
  )
}

/**
 * Draggable split between the stage and the working panes.
 *
 * Stored as a fraction so the layout survives a window resize, and persisted
 * to localStorage.
 */
function useSplit() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [fraction, setFraction] = useState(() => {
    const saved = Number(localStorage.getItem('chronoscope:split'))
    return saved >= 0.3 && saved <= 0.75 ? saved : 0.56
  })
  const dragging = useRef(false)

  useEffect(() => {
    const move = (e: PointerEvent) => {
      if (!dragging.current || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const next = Math.min(0.75, Math.max(0.3, (e.clientX - rect.left) / rect.width))
      setFraction(next)
    }
    const up = () => {
      if (!dragging.current) return
      dragging.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      localStorage.setItem('chronoscope:split', String(fraction))
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [fraction])

  const onGrab = useCallback(() => {
    dragging.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [])

  const reset = useCallback(() => {
    setFraction(0.56)
    localStorage.setItem('chronoscope:split', '0.56')
  }, [])

  return {
    containerRef,
    onGrab,
    reset,
    leftStyle: { '--split': `${(fraction * 100).toFixed(2)}%` } as React.CSSProperties,
    fraction,
  }
}
