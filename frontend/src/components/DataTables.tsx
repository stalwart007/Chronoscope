/**
 * Sortable tables over the derived datasets: scenes, chunks and keyframes.
 *
 * Every row seeks the player, and each view exports to CSV or JSON.
 */

import { useMemo, useState } from 'react'
import { downloadCsv, slug } from '../lib/download'
import { fmtTime, speakerColor } from '../lib/format'
import { useStore } from '../lib/store'
import type { Keyframe, Scene, Segment, VideoChunk } from '../lib/types'
import { useSort } from '../lib/table'
import { Button, CopyButton, Empty, FrameImage, Icon, Menu, Segmented, Th, Tip } from './ui'

type Tab = 'scenes' | 'chunks' | 'frames'

export function DataTables({ scenes, chunks, keyframes, segments, videoId, title, onOpenFrame }: {
  scenes: Scene[]
  chunks: VideoChunk[]
  keyframes: Keyframe[]
  segments: Segment[]
  videoId: string
  title: string
  onOpenFrame: (frames: Keyframe[], id: string) => void
}) {
  const [tab, setTab] = useState<Tab>('scenes')
  const { currentTime, seek } = useStore()

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-[var(--color-line-soft)] px-3">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { value: 'scenes', label: `Scenes ${scenes.length}`, title: 'Detected cuts and their character' },
            { value: 'chunks', label: `Chunks ${chunks.length}`, title: 'The units that get embedded and retrieved' },
            { value: 'frames', label: `Frames ${keyframes.length}`, title: 'Keyframes kept for visual search' },
          ]}
        />
        <div className="ml-auto">
          <ExportMenu videoId={videoId} title={title} tab={tab} scenes={scenes} chunks={chunks} keyframes={keyframes} segments={segments} />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {tab === 'scenes' && <ScenesTable scenes={scenes} keyframes={keyframes} currentTime={currentTime} onSeek={seek} />}
        {tab === 'chunks' && <ChunksTable chunks={chunks} currentTime={currentTime} onSeek={seek} />}
        {tab === 'frames' && <FramesTable keyframes={keyframes} currentTime={currentTime} onSeek={seek} onOpenFrame={onOpenFrame} />}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ exports */

export function ExportMenu({ videoId, title, tab, scenes, chunks, keyframes, segments }: {
  videoId: string; title: string; tab?: Tab
  scenes: Scene[]; chunks: VideoChunk[]; keyframes: Keyframe[]; segments: Segment[]
}) {
  const base = `/api/videos/${videoId}/export`
  const grab = (path: string) => { window.location.href = `${base}${path}` }

  // Client-side CSV for the table currently on screen keeps "export what I see"
  // honest, the server export is the full dataset, this is the current view.
  const exportVisible = () => {
    const name = slug(title)
    if (tab === 'chunks') {
      downloadCsv(
        `${name}-chunks.csv`,
        chunks.map((c) => [c.index, c.span.start, c.span.end, c.speakers.join('|'), c.keywords.join('|'), c.text]),
        ['index', 'start_s', 'end_s', 'speakers', 'keywords', 'text'],
      )
    } else if (tab === 'frames') {
      downloadCsv(
        `${name}-frames.csv`,
        keyframes.map((k) => [k.id, k.scene_index, k.timestamp, k.quality, k.sharpness, k.entropy, k.text_density, k.is_slide ? 1 : 0]),
        ['id', 'scene', 'timestamp_s', 'quality', 'sharpness', 'entropy', 'text_density', 'is_slide'],
      )
    } else {
      downloadCsv(
        `${name}-scenes.csv`,
        scenes.map((s) => [s.index, s.span.start, s.span.end, s.span.duration, s.kind, s.cut_score, s.static_ratio]),
        ['index', 'start_s', 'end_s', 'duration_s', 'kind', 'cut_score', 'static_ratio'],
      )
    }
  }

  return (
    <Menu
      width={250}
      trigger={({ toggle, open }) => (
        <Button size="sm" onClick={toggle} aria-expanded={open}>
          <Icon.Download /> Export
        </Button>
      )}
      items={[
        { label: 'Transcript · SubRip', hint: '.srt', onSelect: () => grab('/transcript?format=srt') },
        { label: 'Transcript · WebVTT', hint: '.vtt', onSelect: () => grab('/transcript?format=vtt') },
        { label: 'Transcript · plain text', hint: '.txt', onSelect: () => grab('/transcript?format=txt') },
        { label: 'Transcript · spreadsheet', hint: '.csv', onSelect: () => grab('/transcript?format=csv') },
        { separator: true, label: '' },
        { label: 'Scenes', hint: '.csv', onSelect: () => grab('/scenes?format=csv') },
        { label: 'Chunks', hint: '.csv', onSelect: () => grab('/chunks?format=csv') },
        { label: 'Keyframe measurements', hint: '.csv', onSelect: () => grab('/keyframes?format=csv') },
        { label: 'All keyframe images', hint: '.zip', onSelect: () => grab('/frames.zip') },
        { separator: true, label: '' },
        { label: 'Complete analysis bundle', hint: '.json', onSelect: () => grab('/bundle?format=json') },
        ...(tab ? [{ label: 'This table as shown', hint: '.csv', onSelect: exportVisible }] : []),
        { separator: true, label: '' },
        {
          label: 'Segments as JSON',
          hint: `${segments.length}`,
          onSelect: () => grab('/transcript?format=json'),
        },
      ]}
    />
  )
}

/* ------------------------------------------------------------------- tables */

function ScenesTable({ scenes, keyframes, currentTime, onSeek }: {
  scenes: Scene[]; keyframes: Keyframe[]; currentTime: number; onSeek: (t: number) => void
}) {
  const rows = useMemo(
    () => scenes.map((s) => ({
      index: s.index, start: s.span.start, end: s.span.end, duration: s.span.duration,
      kind: s.kind, cut: s.cut_score, staticRatio: s.static_ratio,
      frame: keyframes.find((k) => k.scene_index === s.index),
    })),
    [scenes, keyframes],
  )
  const { sorted, key, dir, toggle } = useSort(rows, 'index')
  if (!scenes.length) return <Empty title="No scenes detected" />

  return (
    <table className="table">
      <thead>
        <tr>
          <Th width={54}>Frame</Th>
          <Th sortKey="index" active={key === 'index'} dir={dir} onSort={toggle} width={44}>#</Th>
          <Th sortKey="start" active={key === 'start'} dir={dir} onSort={toggle}>Start</Th>
          <Th sortKey="duration" active={key === 'duration'} dir={dir} onSort={toggle} align="right">Length</Th>
          <Th sortKey="kind" active={key === 'kind'} dir={dir} onSort={toggle}>Type</Th>
          <Th sortKey="cut" active={key === 'cut'} dir={dir} onSort={toggle} align="right">Cut strength</Th>
          <Th sortKey="staticRatio" active={key === 'staticRatio'} dir={dir} onSort={toggle} align="right">Static</Th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((r) => (
          <tr
            key={r.index}
            data-active={currentTime >= r.start && currentTime < r.end}
            onClick={() => onSeek(r.start)}
            className="cursor-pointer"
          >
            <td>
              {r.frame && (
                <FrameImage path={r.frame.path} className="h-6 w-10 rounded-[3px] border border-[var(--color-line)] object-cover" />
              )}
            </td>
            <td className="mono text-[var(--color-fg-3)]">{r.index + 1}</td>
            <td className="mono">{fmtTime(r.start)}</td>
            <td className="mono text-right text-[var(--color-fg-2)]">{r.duration.toFixed(1)}s</td>
            <td>
              <span className="chip">{r.kind}</span>
            </td>
            <td className="mono text-right text-[var(--color-fg-2)]">{r.cut.toFixed(1)}</td>
            <td className="mono text-right text-[var(--color-fg-2)]">{(r.staticRatio * 100).toFixed(0)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ChunksTable({ chunks, currentTime, onSeek }: {
  chunks: VideoChunk[]; currentTime: number; onSeek: (t: number) => void
}) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const rows = useMemo(
    () => chunks.map((c) => ({
      id: c.id, index: c.index, start: c.span.start, end: c.span.end, duration: c.span.duration,
      speakers: c.speakers.join(', '), keywords: c.keywords.slice(0, 4).join(', '),
      rate: c.speech_rate, frames: c.keyframe_ids.length, text: c.text,
    })),
    [chunks],
  )
  const { sorted, key, dir, toggle } = useSort(rows, 'index')
  if (!chunks.length) return <Empty title="No chunks built" />

  return (
    <table className="table">
      <thead>
        <tr>
          <Th sortKey="index" active={key === 'index'} dir={dir} onSort={toggle} width={44}>#</Th>
          <Th sortKey="start" active={key === 'start'} dir={dir} onSort={toggle}>Span</Th>
          <Th sortKey="speakers" active={key === 'speakers'} dir={dir} onSort={toggle}>Speaker</Th>
          <Th sortKey="keywords" active={key === 'keywords'} dir={dir} onSort={toggle}>Keywords</Th>
          <Th sortKey="rate" active={key === 'rate'} dir={dir} onSort={toggle} align="right">Words/s</Th>
          <Th sortKey="frames" active={key === 'frames'} dir={dir} onSort={toggle} align="right">Frames</Th>
          <Th width={64} align="right">Text</Th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((r) => (
          <>
            <tr
              key={r.id}
              data-active={currentTime >= r.start && currentTime < r.end}
              onClick={() => onSeek(r.start)}
              className="cursor-pointer"
            >
              <td className="mono text-[var(--color-fg-3)]">{r.index + 1}</td>
              <td className="mono whitespace-nowrap">{fmtTime(r.start)}-{fmtTime(r.end)}</td>
              <td>
                {r.speakers ? (
                  <span className="mono text-[11.5px]" style={{ color: speakerColor(r.speakers.split(',')[0]) }}>
                    {r.speakers}
                  </span>
                ) : (
                  <span className="text-[var(--color-fg-4)]">-</span>
                )}
              </td>
              <td className="max-w-[240px] truncate text-[var(--color-fg-2)]">{r.keywords || '-'}</td>
              <td className="mono text-right text-[var(--color-fg-2)]">{r.rate.toFixed(1)}</td>
              <td className="mono text-right text-[var(--color-fg-2)]">{r.frames}</td>
              <td className="text-right">
                <div className="flex justify-end gap-0.5" onClick={(e) => e.stopPropagation()}>
                  <CopyButton text={r.text} what="Chunk text copied" />
                  <Tip label={expanded === r.id ? 'Collapse' : 'Read full text'}>
                    <Button
                      variant="quiet" size="sm" icon
                      onClick={() => setExpanded((v) => (v === r.id ? null : r.id))}
                      aria-label="Toggle text"
                    >
                      <Icon.Expand />
                    </Button>
                  </Tip>
                </div>
              </td>
            </tr>
            {expanded === r.id && (
              <tr key={`${r.id}-text`}>
                <td colSpan={7} className="bg-[var(--color-surface-2)] text-[12.5px] leading-relaxed text-[var(--color-fg-2)]">
                  {r.text || <span className="text-[var(--color-fg-4)]">No speech in this chunk.</span>}
                </td>
              </tr>
            )}
          </>
        ))}
      </tbody>
    </table>
  )
}

function FramesTable({ keyframes, currentTime, onSeek, onOpenFrame }: {
  keyframes: Keyframe[]; currentTime: number; onSeek: (t: number) => void
  onOpenFrame: (frames: Keyframe[], id: string) => void
}) {
  const rows = useMemo(
    () => keyframes.map((k) => ({
      id: k.id, timestamp: k.timestamp, scene: k.scene_index, quality: k.quality,
      sharpness: k.sharpness, entropy: k.entropy, density: k.text_density,
      kind: k.is_slide ? 'slide' : 'shot', path: k.path,
    })),
    [keyframes],
  )
  const { sorted, key, dir, toggle } = useSort(rows, 'timestamp')
  if (!keyframes.length) return <Empty title="No keyframes selected" />

  return (
    <table className="table">
      <thead>
        <tr>
          <Th width={62}>Frame</Th>
          <Th sortKey="timestamp" active={key === 'timestamp'} dir={dir} onSort={toggle}>Time</Th>
          <Th sortKey="scene" active={key === 'scene'} dir={dir} onSort={toggle} align="right">Scene</Th>
          <Th sortKey="kind" active={key === 'kind'} dir={dir} onSort={toggle}>Type</Th>
          <Th sortKey="quality" active={key === 'quality'} dir={dir} onSort={toggle} align="right">Quality</Th>
          <Th sortKey="sharpness" active={key === 'sharpness'} dir={dir} onSort={toggle} align="right">Sharpness</Th>
          <Th sortKey="entropy" active={key === 'entropy'} dir={dir} onSort={toggle} align="right">Entropy</Th>
          <Th sortKey="density" active={key === 'density'} dir={dir} onSort={toggle} align="right">Text</Th>
          <Th width={40} />
        </tr>
      </thead>
      <tbody>
        {sorted.map((r) => (
          <tr
            key={r.id}
            data-active={Math.abs(currentTime - r.timestamp) < 1.5}
            onClick={() => onSeek(r.timestamp)}
            className="cursor-pointer"
          >
            <td>
              <FrameImage
                path={r.path}
                className="h-7 w-12 rounded-[3px] border border-[var(--color-line)] object-cover transition-opacity hover:opacity-80"
                onClick={() => onOpenFrame(keyframes, r.id)}
              />
            </td>
            <td className="mono">{fmtTime(r.timestamp)}</td>
            <td className="mono text-right text-[var(--color-fg-3)]">{r.scene + 1}</td>
            <td><span className="chip">{r.kind}</span></td>
            <td className="mono text-right text-[var(--color-fg-2)]">{r.quality.toFixed(3)}</td>
            <td className="mono text-right text-[var(--color-fg-2)]">{r.sharpness.toFixed(4)}</td>
            <td className="mono text-right text-[var(--color-fg-2)]">{r.entropy.toFixed(2)}</td>
            <td className="mono text-right text-[var(--color-fg-2)]">{r.density.toFixed(4)}</td>
            <td className="text-right">
              <div onClick={(e) => e.stopPropagation()}>
                <Tip label="Open full size">
                  <Button variant="quiet" size="sm" icon onClick={() => onOpenFrame(keyframes, r.id)} aria-label="Open frame">
                    <Icon.Expand />
                  </Button>
                </Tip>
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

