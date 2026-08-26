/** The library: add footage, watch it process, open one to analyse. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiError, api, ingestStream } from '../lib/api'
import { downloadCsv } from '../lib/download'
import { fmtBytes, fmtDuration, relativeTime, speakerColor } from '../lib/format'
import { useStore } from '../lib/store'
import { explainError, toast } from '../lib/toast'
import type { VideoSummary } from '../lib/types'
import { useDemoLoader } from '../hooks/useDemoLoader'
import { Button, Empty, Icon, Menu, ProgressRing, Segmented, Spinner, Stat, Tip } from './ui'

const VIDEO_EXT = /\.(mp4|mov|mkv|webm|avi|m4v|mpe?g|wmv|flv)$/i
const SUB_EXT = /\.(srt|vtt|json)$/i
type SortKey = 'recent' | 'title' | 'duration'

export function Library() {
  const { videos, loadingVideos, libraryError, loadVideos, select } = useStore()
  const [filter, setFilter] = useState('')
  const [sort, setSort] = useState<SortKey>('recent')
  const loadDemo = useDemoLoader(loadVideos)

  useEffect(() => { loadVideos() }, [loadVideos])

  useEffect(() => {
    const run = () => void loadDemo()
    window.addEventListener('chronoscope:demo', run)
    return () => window.removeEventListener('chronoscope:demo', run)
  }, [loadDemo])

  // Poll only while something is processing, so an idle library is silent.
  const anyActive = videos.some((v) => v.status === 'pending' || v.status === 'running')
  useEffect(() => {
    if (!anyActive) return
    const id = setInterval(loadVideos, 4000)
    return () => clearInterval(id)
  }, [anyActive, loadVideos])

  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase()
    const matched = q
      ? videos.filter((v) =>
          [v.title, v.filename, v.summary, ...v.topics, ...v.speakers].join(' ').toLowerCase().includes(q))
      : videos
    const sorted = [...matched]
    if (sort === 'title') sorted.sort((a, b) => a.title.localeCompare(b.title))
    else if (sort === 'duration') sorted.sort((a, b) => b.duration - a.duration)
    else sorted.sort((a, b) => (a.created_at < b.created_at ? 1 : -1))
    return sorted
  }, [videos, filter, sort])

  const exportLibrary = () =>
    downloadCsv(
      'chronoscope-library.csv',
      videos.map((v) => [v.id, v.title, v.filename, v.status, v.duration, v.size_bytes,
        v.speakers.join('|'), v.topics.join('|'), v.created_at]),
      ['id', 'title', 'filename', 'status', 'duration_s', 'size_bytes', 'speakers', 'topics', 'created_at'],
    )

  return (
    <div className="mx-auto flex h-full w-full max-w-[1360px] flex-col gap-4 overflow-y-auto px-4 py-5 sm:px-6">
      <Dropzone onUploaded={loadVideos} />
      <UrlBar onQueued={loadVideos} />


      {videos.length > 0 && <Overview videos={videos} />}

      {videos.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-[180px] flex-1">
            <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--color-fg-4)]">
              <Icon.Search />
            </span>
            <input
              className="field !pl-8"
              placeholder="Filter by title, topic or speaker..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              aria-label="Filter library"
            />
          </div>
          <Segmented
            value={sort}
            onChange={setSort}
            options={[
              { value: 'recent', label: 'Recent' },
              { value: 'title', label: 'Title' },
              { value: 'duration', label: 'Longest' },
            ]}
          />
          <span className="mono text-[11px] text-[var(--color-fg-4)]">{visible.length}/{videos.length}</span>
          <Tip label="Export the library index as CSV">
            <Button size="sm" icon onClick={exportLibrary} aria-label="Export library"><Icon.Download /></Button>
          </Tip>
        </div>
      )}

      {libraryError && videos.length > 0 && (
        <div className="surface flex items-center gap-2 border-[rgba(240,113,120,0.35)] p-2.5 text-[12.5px] text-[var(--color-critical)]">
          <Icon.Warn />
          <span className="flex-1">{libraryError}. Showing the last known list.</span>
          <Button size="sm" onClick={loadVideos}><Icon.Refresh /> Retry</Button>
        </div>
      )}

      {loadingVideos && videos.length === 0 ? (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {[0, 1, 2].map((i) => <div key={i} className="skeleton h-[226px] rounded-[var(--radius-lg)]" />)}
        </div>
      ) : libraryError && videos.length === 0 ? (
        <Unreachable message={libraryError} onRetry={loadVideos} />
      ) : videos.length === 0 ? (
        <FirstRun onLoadDemo={loadDemo} />
      ) : visible.length === 0 ? (
        <div className="surface"><Empty title={`Nothing matches "${filter}"`} /></div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {visible.map((v) => (
            <VideoCard key={v.id} video={v} onOpen={() => select(v.id)} onChanged={loadVideos} />
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * Corpus summary.
 *
 * A library of one card leaves most of the page empty and says nothing about
 * what has been indexed. These are the figures worth knowing before searching:
 * how much footage there is, how much of it is speech, and who is in it.
 */
function Overview({ videos }: { videos: VideoSummary[] }) {
  const done = videos.filter((v) => v.status === 'completed')
  const totals = done.reduce(
    (acc, v) => {
      const s = v.stats ?? {}
      acc.duration += v.duration || 0
      acc.scenes += s.scenes ?? 0
      acc.chunks += s.chunks ?? 0
      acc.frames += s.keyframes ?? 0
      acc.words += s.transcript?.words ?? 0
      acc.speech += s.transcript?.speech_seconds ?? 0
      for (const sp of v.speakers) acc.speakers.add(`${v.id}:${sp}`)
      for (const t of v.topics) acc.topics.set(t, (acc.topics.get(t) ?? 0) + 1)
      return acc
    },
    { duration: 0, scenes: 0, chunks: 0, frames: 0, words: 0, speech: 0,
      speakers: new Set<string>(), topics: new Map<string, number>() },
  )
  const processing = videos.filter((v) => v.status === 'pending' || v.status === 'running').length
  const failed = videos.filter((v) => v.status === 'failed').length
  const speechShare = totals.duration ? totals.speech / totals.duration : 0
  const topTopics = [...totals.topics.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8)

  return (
    <div className="surface flex flex-col gap-3 p-3.5">
      <div className="grid grid-cols-2 gap-y-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Indexed" value={done.length} hint="Videos ready to question" />
        <Stat label="Footage" value={fmtDuration(totals.duration)} hint="Total duration across the library" />
        <Stat
          label="Speech"
          value={`${(speechShare * 100).toFixed(0)}%`}
          hint={`${fmtDuration(totals.speech)} of speech, ${totals.words.toLocaleString()} words transcribed`}
        />
        <Stat label="Scenes" value={totals.scenes} hint="Detected cuts across the library" />
        <Stat label="Chunks" value={totals.chunks} hint="Retrievable units in the index" />
        <Stat label="Frames" value={totals.frames} hint="Keyframes held for visual search" />
      </div>

      {(topTopics.length > 0 || processing > 0 || failed > 0) && (
        <div className="flex flex-wrap items-center gap-1 border-t border-[var(--color-line-soft)] pt-2.5">
          {processing > 0 && (
            <span className="chip" style={{ color: 'var(--color-fg-2)' }}>
              <Spinner size={10} /> {processing} processing
            </span>
          )}
          {failed > 0 && (
            <span className="chip" style={{ color: 'var(--color-critical)' }}>{failed} failed</span>
          )}
          {topTopics.map(([topic, count]) => (
            <Tip key={topic} label={count > 1 ? `Appears in ${count} videos` : 'Appears in 1 video'}>
              <span className="chip">
                {topic}
                {count > 1 && <span className="mono text-[var(--color-fg-4)]">{count}</span>}
              </span>
            </Tip>
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * Shown when the library could not be loaded.
 *
 * Distinct from the empty state on purpose: an unreachable server must never
 * be presented as "you have no videos", which would suggest data loss.
 */
function Unreachable({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="surface px-6 py-10 text-center">
      <Icon.Warn className="mx-auto mb-2 text-[var(--color-caution)]" />
      <h2 className="text-[15px] font-medium">{message}</h2>
      <p className="mx-auto mt-1.5 max-w-md text-[12.5px] leading-relaxed text-[var(--color-fg-3)]">
        Your indexed videos are safe. The interface just cannot reach the API right now.
        Check that the server is running, then retry.
      </p>
      <div className="mt-4 flex justify-center gap-1.5">
        <Button variant="primary" onClick={onRetry}><Icon.Refresh /> Retry</Button>
      </div>
      <p className="mono mt-3 text-[11px] text-[var(--color-fg-4)]">make dev</p>
    </div>
  )
}

/** Empty state: explains the flow and offers a ready-made example. */
function FirstRun({ onLoadDemo }: { onLoadDemo: () => void }) {
  const steps = [
    ['Add footage', 'Drop a video above, or load the demo talk.'],
    ['It gets understood', 'Scenes, speech, speakers and keyframes, a couple of seconds per minute of video.'],
    ['Ask in plain language', '"When do they show the architecture diagram?", you get the moment, the frame and the quote.'],
  ]
  return (
    <div className="surface overflow-hidden">
      <div className="border-b border-[var(--color-line-soft)] px-6 py-8 text-center">
        <h2 className="text-[18px] font-semibold">Nothing indexed yet</h2>
        <p className="mx-auto mt-1.5 max-w-md text-[12.5px] leading-relaxed text-[var(--color-fg-3)]">
          Chronoscope turns video into something you can question. Try it in one click, no file needed.
        </p>
        <Button variant="primary" className="mx-auto mt-4" onClick={onLoadDemo}>
          <Icon.Ask /> Load the demo video
        </Button>
        <p className="mt-2 text-[11px] text-[var(--color-fg-4)]">
          Generates a 58-second talk with five scenes and two speakers, then indexes it.
        </p>
      </div>
      <div className="grid gap-px bg-[var(--color-line-soft)] sm:grid-cols-3">
        {steps.map(([title, body], i) => (
          <div key={title} className="bg-[var(--color-surface)] p-5">
            <div className="mono mb-1.5 text-[11px] text-[var(--color-fg-4)]">0{i + 1}</div>
            <div className="text-[13px] font-medium">{title}</div>
            <p className="mt-1 text-[11.5px] leading-relaxed text-[var(--color-fg-3)]">{body}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

function Dropzone({ onUploaded }: { onUploaded: () => void }) {
  const [over, setOver] = useState(false)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [note, setNote] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const send = useCallback(
    async (files: File[]) => {
      const video = files.find((f) => VIDEO_EXT.test(f.name))
      if (!video) {
        toast.error('No video in that drop', 'Drop an mp4, mov, mkv, webm or avi file.')
        return
      }
      // A caption file dropped alongside is used verbatim, faster and more
      // accurate than re-running ASR on footage that already has one.
      const transcript = files.find((f) => SUB_EXT.test(f.name)) ?? null
      setBusy(true)
      setNote(null)
      setProgress(0)
      try {
        const res = await api.upload(video, {
          title: video.name.replace(VIDEO_EXT, ''),
          transcript,
          onProgress: setProgress,
        })
        const message = res.duplicate
          ? 'Identical content is already indexed.'
          : `Processing started${transcript ? ` with ${transcript.name}` : ''}.`
        setNote(message)
        toast.success(res.duplicate ? 'Already in your library' : `"${video.name}" queued`, message)
        onUploaded()
      } catch (e) {
        const { title, body } = explainError(e instanceof ApiError ? e : { message: String(e) })
        toast.error(title, body)
        setNote(title)
      } finally {
        setBusy(false)
        setProgress(0)
      }
    },
    [onUploaded],
  )

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); send(Array.from(e.dataTransfer.files)) }}
      onClick={() => !busy && inputRef.current?.click()}
      onKeyDown={(e) => { if (e.key === 'Enter') inputRef.current?.click() }}
      role="button"
      tabIndex={0}
      className={`surface relative cursor-pointer overflow-hidden px-6 py-5 text-center transition-colors ${
        over ? '!border-[var(--color-focus)] bg-[rgba(122,162,247,0.06)]' : 'hover:!border-[var(--color-line)]'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="video/*,.srt,.vtt,.json"
        className="hidden"
        onChange={(e) => { send(Array.from(e.target.files ?? [])); e.target.value = '' }}
      />
      {busy && (
        <div className="absolute inset-x-0 top-0 h-0.5 bg-[var(--color-line)]">
          <div className="h-full bg-[var(--color-fg)] transition-[width]" style={{ width: `${progress}%` }} />
        </div>
      )}
      <div className="flex flex-col items-center gap-1.5">
        <span className={over ? 'text-[var(--color-focus)]' : 'text-[var(--color-fg-4)]'}>
          {busy ? <Spinner size={18} /> : <Icon.Upload />}
        </span>
        <p className="text-[13px] font-medium">
          {busy ? `Uploading... ${progress}%` : 'Drop a video here, or click to browse'}
        </p>
        <p className="text-[11.5px] text-[var(--color-fg-4)]">
          Add an .srt / .vtt / .json caption file to skip transcription
        </p>
        {note && <p className="mt-0.5 text-[11.5px] text-[var(--color-fg-3)]">{note}</p>}
      </div>
    </div>
  )
}

/**
 * Ingest from a direct media URL.
 *
 * Only direct links are accepted: the server refuses page URLs from video
 * platforms by name, and refuses private or reserved addresses so it cannot be
 * used to reach internal services.
 */
function UrlBar({ onQueued }: { onQueued: () => void }) {
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    const value = url.trim()
    if (!value || busy) return
    setBusy(true)
    const id = toast.progress('Downloading', new URL(value, window.location.href).hostname)
    try {
      const res = await api.ingestUrl(value)
      toast.update(id, {
        kind: 'success',
        title: res.duplicate ? 'Already in your library' : 'Download complete',
        body: res.duplicate ? 'Identical content is already indexed.' : 'Processing has started.',
        ttl: 4000,
      })
      setUrl('')
      onQueued()
    } catch (err) {
      const { title, body } = explainError(err)
      toast.update(id, { kind: 'error', title, body, ttl: 10000 })
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="flex gap-1.5">
      <div className="relative flex-1">
        <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--color-fg-4)]">
          <Icon.External />
        </span>
        <input
          className="field !pl-8"
          placeholder="Or paste a direct link to a video file (https://.../talk.mp4)"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          aria-label="Media URL"
          inputMode="url"
        />
      </div>
      <Tip label="Direct media links only. Pages from video sites are not downloaded.">
        <Button type="submit" disabled={busy || !url.trim()}>
          {busy ? <Spinner /> : <Icon.Download />} Fetch
        </Button>
      </Tip>
    </form>
  )
}

function VideoCard({ video, onOpen, onChanged }: { video: VideoSummary; onOpen: () => void; onChanged: () => void }) {
  const patchVideo = useStore((s) => s.patchVideo)
  const [stage, setStage] = useState(video.stage)
  const [progress, setProgress] = useState(video.progress)
  const [busy, setBusy] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [done, setDone] = useState<string[]>([])

  const active = video.status === 'pending' || video.status === 'running'

  // While processing, follow the server-sent progress. The stream replays from
  // its buffer, so mounting late never misses earlier stages.
  useEffect(() => {
    if (!active) return
    const controller = new AbortController()
    ingestStream(
      video.id,
      (kind, data) => {
        if (typeof data?.progress === 'number') setProgress(data.progress)
        if (data?.label) setStage(data.label)
        if (kind === 'stage_done' && data?.label) setDone((d) => (d.includes(data.label) ? d : [...d, data.label]))
        if (kind === 'stage_error' && data?.fatal) toast.error('Processing failed', `${data.stage}: ${data.error}`)
        if (kind === 'job_done' || kind === 'job_failed' || kind === 'job_cancelled') {
          setStage('')
          if (kind === 'job_done') toast.success('Ready to question', `"${video.title}" finished processing.`)
          onChanged()
        }
      },
      controller.signal,
    ).catch(() => {})
    return () => controller.abort()
  }, [active, video.id, video.title, onChanged])

  const remove = async () => {
    if (!confirmDelete) {
      // Two-step delete instead of a modal: it stays in place, is reversible
      // by clicking away, and needs no keyboard trap.
      setConfirmDelete(true)
      window.setTimeout(() => setConfirmDelete(false), 4000)
      return
    }
    setBusy(true)
    try {
      await api.deleteVideo(video.id)
      toast.success('Deleted', `"${video.title}" and its derived data were removed.`)
      onChanged()
    } catch (err) {
      const { title, body } = explainError(err)
      toast.error(title, body)
    } finally {
      setBusy(false)
      setConfirmDelete(false)
    }
  }

  const reindex = async () => {
    setBusy(true)
    try {
      await api.reindex(video.id)
      patchVideo(video.id, { status: 'pending', progress: 0 })
      toast.info('Re-indexing', 'The video will be processed again from its original file.')
      onChanged()
    } catch (err) {
      const { title, body } = explainError(err)
      toast.error(title, body)
    } finally {
      setBusy(false)
    }
  }

  const stats = video.stats ?? {}
  const base = `/api/videos/${video.id}/export`

  return (
    <article
      onClick={video.status === 'completed' ? onOpen : undefined}
      className={`surface group flex flex-col overflow-hidden transition-colors ${
        video.status === 'completed' ? 'cursor-pointer hover:border-[var(--color-line-strong)]' : ''
      }`}
    >
      <div className="relative aspect-video w-full overflow-hidden bg-black">
        {video.poster ? (
          <img
            src={api.posterUrl(video.poster)}
            alt=""
            className="h-full w-full object-cover opacity-90 transition-opacity duration-300 group-hover:opacity-100"
            loading="lazy"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-[var(--color-fg-4)]"><Icon.Film /></div>
        )}

        {active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 bg-[rgba(10,10,11,0.82)]">
            <div className="relative">
              <ProgressRing value={progress} size={40} />
              <span className="mono absolute inset-0 flex items-center justify-center text-[10px]">
                {Math.round(progress)}
              </span>
            </div>
            <span className="text-[11.5px] text-[var(--color-fg-2)]">{stage || 'queued'}</span>
            {done.length > 0 && (
              <span className="mono text-[10px] text-[var(--color-fg-4)]">{done.length} of 10 stages</span>
            )}
          </div>
        )}

        {video.status === 'failed' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-[rgba(20,8,10,0.86)] p-4 text-center">
            <Icon.Warn className="text-[var(--color-critical)]" />
            <span className="text-[12px] font-medium text-[var(--color-critical)]">Processing failed</span>
            <span className="line-clamp-2 text-[10.5px] text-[var(--color-fg-3)]">{video.error}</span>
            <Button size="sm" className="mt-1" onClick={(e) => { e.stopPropagation(); reindex() }}>Retry</Button>
          </div>
        )}

        {video.duration > 0 && (
          <span className="mono absolute bottom-1.5 right-1.5 rounded-[4px] bg-[rgba(10,10,11,0.8)] px-1.5 py-0.5 text-[10.5px] text-[var(--color-fg-2)]">
            {fmtDuration(video.duration)}
          </span>
        )}
      </div>

      <div className="flex flex-1 flex-col gap-2 p-3">
        <div className="flex items-start justify-between gap-2">
          <h3 className="line-clamp-1 text-[13.5px] font-medium">{video.title}</h3>
          <div className="flex shrink-0 gap-0.5" onClick={(e) => e.stopPropagation()}>
            <Menu
              width={220}
              trigger={({ toggle }) => (
                <Button
                  variant="quiet" size="sm" icon onClick={toggle} aria-label="Video actions"
                  className="opacity-0 transition-opacity group-hover:opacity-100 data-[open=true]:opacity-100"
                >
                  <Icon.More />
                </Button>
              )}
              items={[
                { label: 'Open', disabled: video.status !== 'completed', onSelect: onOpen },
                { separator: true, label: '' },
                { label: 'Transcript', hint: '.srt', disabled: video.status !== 'completed', onSelect: () => { window.location.href = `${base}/transcript?format=srt` } },
                { label: 'Analysis bundle', hint: '.json', disabled: video.status !== 'completed', onSelect: () => { window.location.href = `${base}/bundle?format=json` } },
                { label: 'Keyframe images', hint: '.zip', disabled: video.status !== 'completed', onSelect: () => { window.location.href = `${base}/frames.zip` } },
                { separator: true, label: '' },
                { label: 'Re-index', disabled: busy || active, onSelect: reindex },
                { label: confirmDelete ? 'Click again to confirm' : 'Delete', danger: true, disabled: busy, onSelect: remove },
              ]}
            />
          </div>
        </div>

        {video.summary && (
          <p className="line-clamp-2 text-[11.5px] leading-relaxed text-[var(--color-fg-3)]">{video.summary}</p>
        )}

        <div className="mt-auto flex flex-wrap items-center gap-1 pt-0.5">
          {video.speakers.slice(0, 3).map((s) => (
            <span key={s} className="chip" style={{ color: speakerColor(s) }}>{s.replace('SPEAKER_', 'S')}</span>
          ))}
          {stats.scenes ? <span className="chip">{stats.scenes} scenes</span> : null}
          {stats.chunks ? <span className="chip">{stats.chunks} chunks</span> : null}
          {stats.keyframes ? <span className="chip">{stats.keyframes} frames</span> : null}
        </div>

        <div className="flex items-center justify-between border-t border-[var(--color-line-soft)] pt-2 text-[10.5px] text-[var(--color-fg-4)]">
          <span className="mono">{fmtBytes(video.size_bytes)}</span>
          <span>{relativeTime(video.created_at)}</span>
        </div>
      </div>
    </article>
  )
}
