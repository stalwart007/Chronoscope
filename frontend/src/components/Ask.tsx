/** The question surface: query, live reasoning, cited answer, ranked evidence. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, askStream } from '../lib/api'
import { copyText, downloadBlob, downloadCsv, downloadJson, slug } from '../lib/download'
import { MODALITY_COLOR, MODALITY_LABEL, fmtNumber, fmtTime, parseTimestamp, speakerColor } from '../lib/format'
import { useStore } from '../lib/store'
import { explainError, toast } from '../lib/toast'
import type { AgentEvent, AnswerBundle, Computation, GraphTopology, Keyframe, ScoredHit } from '../lib/types'
import { AgentSwarm } from './AgentSwarm'
import { Thread } from './Thread'
import {
  Button, ChannelChip, ChannelDot, CopyButton, Empty, FrameImage, FusionBar,
  Icon, Kbd, Menu, Meter, Segmented, Spinner, Tip,
} from './ui'

const SUGGESTIONS = [
  { label: 'Locate a visual', q: 'When does the speaker show the architecture diagram?' },
  { label: 'Attribute a claim', q: 'What did the second speaker say about Kubernetes?' },
  { label: 'Read a chart', q: 'Find the revenue chart and calculate the year over year growth' },
  { label: 'Summarise', q: 'Summarise the main points of this talk' },
]

/** Once a thread exists, the useful prompts are the ones that lean on it. */
const FOLLOW_UPS = [
  { label: 'What next', q: 'What did they say right after that?' },
  { label: 'Who', q: 'Who said that?' },
  { label: 'Go deeper', q: 'Tell me more about it' },
  { label: 'Before', q: 'And what came before that?' },
]

export function Ask({ videoId, scoped, onOpenFrame }: {
  videoId: string | null
  scoped: boolean
  onOpenFrame?: (frames: Keyframe[], id: string) => void
}) {
  const { answer, setAnswer, seek, sessionId, thread, recordTurn, newThread } = useStore()
  const [query, setQuery] = useState('')
  const [busy, setBusy] = useState(false)
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [topology, setTopology] = useState<GraphTopology | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<'answer' | 'evidence' | 'trace'>('answer')
  const [showGraph, setShowGraph] = useState(true)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => {
    api.graph().then(setTopology).catch(() => setTopology(null))
    return () => abort.current?.abort()
  }, [])

  // Clearing the thread clears what was said about it: leaving the previous
  // run's reasoning graph behind implies an answer that is no longer on screen.
  useEffect(() => {
    if (!thread.length) {
      setEvents([])
      setError(null)
    }
  }, [thread.length])

  // A shared link carries its question, so the recipient lands on the same view.
  useEffect(() => {
    const shared = new URL(window.location.href).searchParams.get('q')
    if (shared) setQuery(shared)
  }, [])

  const run = useCallback(
    async (q: string) => {
      const text = q.trim()
      if (!text || busy) return
      abort.current?.abort()
      const controller = new AbortController()
      abort.current = controller
      setBusy(true)
      setError(null)
      setEvents([])
      setAnswer(null)
      setTab('answer')
      setShowGraph(true)
      try {
        await askStream(
          {
            q: text,
            video_ids: scoped && videoId ? [videoId] : [],
            top_k: 8,
            session_id: sessionId ?? undefined,
          },
          {
            open: (d) => { if (d?.graph) setTopology(d.graph) },
            agent: (e) => setEvents((prev) => [...prev, e]),
            answer: (a: AnswerBundle) => { recordTurn(a); setShowGraph(false) },
            error: (e) => setError(typeof e === 'string' ? e : e?.message ?? 'The query failed.'),
          },
          controller.signal,
        )
      } catch (e) {
        if (!controller.signal.aborted) {
          const { title, body } = explainError(e)
          setError(`${title}, ${body}`)
          toast.error(title, body)
        }
      } finally {
        if (!controller.signal.aborted) setBusy(false)
      }
    },
    [busy, scoped, videoId, sessionId, recordTurn, setAnswer],
  )

  const nodeStates = useMemo(() => {
    const map: Record<string, string> = {}
    for (const e of events) {
      if (e.node !== '__end__') map[e.node] = e.kind === 'error' ? 'error' : 'done'
    }
    return map
  }, [events])

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      {/* ------------------------------------------------------------ input */}
      <div className="surface p-2.5">
        <form onSubmit={(e) => { e.preventDefault(); run(query) }} className="flex gap-1.5">
          <div className="relative flex-1">
            <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--color-fg-4)]">
              <Icon.Search />
            </span>
            <input
              data-ask-input
              className="field !pl-8 !pr-9"
              placeholder={
                thread.length
                  ? 'Ask a follow-up, "what happened right after that?"'
                  : scoped ? 'Ask about this video...' : 'Ask across every indexed video...'
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Question"
            />
            {!query && (
              <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2">
                <Kbd>/</Kbd>
              </span>
            )}
          </div>
          {busy ? (
            <Button onClick={() => { abort.current?.abort(); setBusy(false) }} title="Stop the query">
              <Spinner /> Stop
            </Button>
          ) : (
            <Button variant="primary" type="submit" disabled={!query.trim()}>
              <Icon.Ask /> Ask
            </Button>
          )}
        </form>

        <div className="mt-2 flex flex-wrap items-center gap-1">
          {(thread.length ? FOLLOW_UPS : SUGGESTIONS).map((s) => (
            <Tip key={s.q} label={s.q}>
              <button
                className="chip transition-colors hover:border-[var(--color-line-strong)] hover:text-[var(--color-fg)]"
                onClick={() => { setQuery(s.q); run(s.q) }}
                disabled={busy}
              >
                {s.label}
              </button>
            </Tip>
          ))}
          {thread.length > 0 && (
            <Tip label="Forget this thread so the next question is read on its own">
              <button
                className="chip ml-auto text-[var(--color-fg-4)] transition-colors hover:text-[var(--color-fg)]"
                onClick={() => { newThread(); setQuery('') }}
                disabled={busy}
              >
                clear context
              </button>
            </Tip>
          )}
        </div>
      </div>

      {/* ----------------------------------------------------------- thread */}
      <Thread onSeek={seek} onAsk={(q) => { setQuery(q); run(q) }} />

      {/* ------------------------------------------------------------ swarm */}
      {(busy || events.length > 0) && (
        <div className="surface shrink-0 animate-rise">
          {showGraph ? (
            <div className="px-1.5 pb-1 pt-1.5">
              <AgentSwarm topology={topology} events={events} active={busy && !answer} />
              <div className="no-scrollbar flex items-center gap-1.5 overflow-x-auto px-1.5 pb-1.5">
                {events.filter((e) => e.kind === 'result' && e.node !== '__end__').slice(-3).map((e) => (
                  <span key={e.seq} className="chip animate-fade-in">{e.message}</span>
                ))}
                {!busy && (
                  <Button variant="quiet" size="sm" className="ml-auto" onClick={() => setShowGraph(false)}>
                    hide
                  </Button>
                )}
              </div>
            </div>
          ) : (
            <button
              className="flex w-full items-center gap-2 px-3 py-2 text-left text-[12px] text-[var(--color-fg-3)] transition-colors hover:text-[var(--color-fg-2)]"
              onClick={() => setShowGraph(true)}
            >
              <span className="flex gap-1">
                {['plan', 'retrieve', 'visual_qa', 'analyst', 'synthesize'].map((n) => (
                  <span
                    key={n}
                    className="h-1.5 w-1.5 rounded-full"
                    style={{
                      background:
                        nodeStates[n] === 'error' ? 'var(--color-critical)'
                        : nodeStates[n] ? 'var(--color-positive)'
                        : 'var(--color-line-strong)',
                    }}
                  />
                ))}
              </span>
              reasoning graph · {Object.keys(nodeStates).length} nodes ran
              <span className="ml-auto">show</span>
            </button>
          )}
        </div>
      )}

      {error && (
        <div className="surface flex items-start gap-2 border-[rgba(240,113,120,0.35)] p-2.5 text-[12.5px] text-[var(--color-critical)]">
          <Icon.Warn className="mt-0.5 shrink-0" /> <span>{error}</span>
        </div>
      )}

      {/* ----------------------------------------------------------- output */}
      {answer ? (
        <div className="surface flex min-h-0 flex-1 flex-col overflow-hidden animate-rise">
          <div className="flex h-10 shrink-0 items-center gap-2 border-b border-[var(--color-line-soft)] px-2">
            <Segmented
              value={tab}
              onChange={setTab}
              options={[
                { value: 'answer', label: 'Answer', title: 'The response, with clickable timestamps' },
                {
                  value: 'evidence',
                  label: answer.restored ? 'Evidence' : `Evidence ${answer.hits.length}`,
                  title: 'Every retrieved moment and which channels found it',
                },
                { value: 'trace', label: 'Trace', title: 'How the question was planned, routed and timed' },
              ]}
            />
            <div className="ml-auto flex items-center gap-1.5">
              <Meter value={answer.confidence} />
              <Tip label="End-to-end: planning, retrieval, specialist agents and synthesis">
                <span className="mono hidden whitespace-nowrap text-[11px] text-[var(--color-fg-4)] xl:inline">
                  {answer.elapsed_ms.toFixed(0)}ms
                </span>
              </Tip>
              <AnswerMenu answer={answer} />
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto p-3.5">
            {tab === 'answer' && <AnswerBody answer={answer} onSeek={seek} onOpenFrame={onOpenFrame} />}
            {tab === 'evidence' && (
              answer.restored
                ? <Restored what="Retrieved evidence" onReask={() => run(answer.query)} />
                : <Evidence hits={answer.hits} onSeek={seek} onOpenFrame={onOpenFrame} />
            )}
            {tab === 'trace' && (
              answer.restored
                ? <Restored what="The reasoning trace" onReask={() => run(answer.query)} />
                : <Trace answer={answer} events={events} />
            )}
          </div>
        </div>
      ) : (
        !busy && (
          <div className="surface flex min-h-0 flex-1 items-center justify-center">
            <Empty
              title="Ask anything about the footage"
              body="The planner decomposes your question, the retriever fuses transcript, vision and keyword channels, and specialists read frames or run calculations only when they are needed."
            />
          </div>
        )
      )}
    </div>
  )
}

/* --------------------------------------------------------- restored turns */

/** A thread stores answers and citations, not the evidence behind them.
 *  Say what is missing and offer the one action that brings it back. */
function Restored({ what, onReask }: { what: string; onReask: () => void }) {
  return (
    <Empty
      title={`${what} is not kept with a thread`}
      body="Threads store each question, its answer and its citations. Ask the question again to rebuild the full retrieval."
      action={<Button variant="primary" size="sm" onClick={onReask}><Icon.Refresh /> Ask again</Button>}
    />
  )
}

/* ------------------------------------------------------------- resolution */

/** What a follow-up was taken to mean. Stated plainly, because a question
 *  answered against the wrong antecedent is worse than one left unanswered. */
function Resolution({ answer }: { answer: AnswerBundle }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded-md border border-[var(--color-line-soft)] bg-[var(--color-bg-1)]">
      <button
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-[11.5px] text-[var(--color-fg-3)] transition-colors hover:text-[var(--color-fg-2)]"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="text-[var(--color-fg-4)]">&#8627;</span>
        read as a follow-up
        <span className="ml-auto text-[var(--color-fg-4)]">{open ? 'hide' : 'why'}</span>
      </button>
      {open && (
        <div className="border-t border-[var(--color-line-soft)] px-2.5 py-2">
          <p className="text-[12px] leading-snug text-[var(--color-fg-2)]">{answer.resolved_query}</p>
          <ul className="mt-1.5 space-y-0.5">
            {answer.resolution_notes.map((n) => (
              <li key={n} className="text-[11px] text-[var(--color-fg-4)]">{n}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

/* -------------------------------------------------------------- answer menu */

function AnswerMenu({ answer }: { answer: AnswerBundle }) {
  const name = slug(answer.query.slice(0, 40), 'answer')

  const asMarkdown = () => {
    const lines = [
      `# ${answer.query}`, '',
      answer.answer, '',
      '## Citations', '',
      ...answer.citations.map((c) => `- \`${fmtTime(c.start)}\`${c.speaker ? ` **${c.speaker}**` : ''}, ${c.quote}`),
    ]
    if (answer.computations.length) {
      lines.push('', '## Computation', '')
      for (const c of answer.computations) {
        lines.push(`${c.explanation ?? ''}`, '', '```python', c.code ?? '', '```', '',
          `Result: \`${JSON.stringify(c.result?.value)}\``)
      }
    }
    return lines.join('\n')
  }

  return (
    <Menu
      width={230}
      trigger={({ toggle }) => (
        <Tip label="Copy, share or export this answer">
          <Button variant="quiet" size="sm" icon onClick={toggle} aria-label="Answer actions"><Icon.More /></Button>
        </Tip>
      )}
      items={[
        {
          label: 'Copy answer with citations',
          onSelect: () => copyText(
            [`Q: ${answer.query}`, '', answer.answer, '',
             ...answer.citations.map((c) => `[${fmtTime(c.start)}] ${c.speaker ? `${c.speaker}: ` : ''}${c.quote}`)].join('\n'),
            'Answer copied',
          ),
        },
        {
          label: 'Copy a link to this question',
          onSelect: () => {
            const url = new URL(window.location.href)
            url.searchParams.set('q', answer.query)
            copyText(url.toString(), 'Link copied')
          },
        },
        { separator: true, label: '' },
        { label: 'Download as Markdown', hint: '.md', onSelect: () => downloadBlob(`${name}.md`, asMarkdown(), 'text/markdown') },
        { label: 'Download evidence bundle', hint: '.json', onSelect: () => downloadJson(`${name}.json`, answer) },
        {
          label: 'Download evidence table',
          hint: '.csv',
          onSelect: () => downloadCsv(
            `${name}-evidence.csv`,
            answer.hits.map((h, i) => [
              i + 1, h.video_id, h.chunk?.label ?? '', h.chunk?.span.start ?? 0, h.score,
              Object.entries(h.ranks).map(([k, v]) => `${k}:${v}`).join('|'),
              h.chunk?.speakers.join('|') ?? '', h.chunk?.text ?? '',
            ]),
            ['rank', 'video_id', 'span', 'start_s', 'score', 'channel_ranks', 'speakers', 'text'],
          ),
        },
      ]}
    />
  )
}

/* ------------------------------------------------------------------ answer */

function AnswerBody({ answer, onSeek, onOpenFrame }: {
  answer: AnswerBundle; onSeek: (t: number) => void
  onOpenFrame?: (frames: Keyframe[], id: string) => void
}) {
  const frames = useMemo(() => answer.hits.flatMap((h) => h.keyframes), [answer.hits])

  return (
    <div className="flex flex-col gap-4">
      {answer.is_followup && <Resolution answer={answer} />}

      <div className="flex flex-wrap items-center gap-1">
        {answer.plan.answer_style && <span className="chip">{answer.plan.answer_style}</span>}
        {answer.plan.tasks.map((t) => (
          <Tip key={t.id} label={t.rationale || t.query}>
            <span className="chip">{t.kind.replace(/_/g, ' ')}</span>
          </Tip>
        ))}
        <Tip label="Which model produced this answer">
          <span className="mono ml-auto text-[10.5px] text-[var(--color-fg-4)]">{answer.model_used}</span>
        </Tip>
      </div>

      <div className="text-[13.5px] leading-[1.7] text-[var(--color-fg)]">
        <RichText text={answer.answer} onSeek={onSeek} />
      </div>

      {answer.computations.map((c, i) => <ComputationCard key={i} computation={c} />)}

      {answer.visual_findings.length > 0 && (
        <section>
          <h4 className="label mb-2">Frames examined</h4>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {answer.visual_findings.map((f) => (
              <div key={f.frame_id} className="group overflow-hidden rounded-[var(--radius)] border border-[var(--color-line-soft)] transition-colors hover:border-[var(--color-line-strong)]">
                <div className="relative">
                  <FrameImage
                    path={f.image}
                    className="aspect-video w-full cursor-zoom-in object-cover"
                    onClick={() => onOpenFrame?.(frames.length ? frames : [], f.frame_id)}
                  />
                  <button
                    onClick={() => onSeek(f.timestamp)}
                    className="absolute bottom-1 left-1 rounded-[4px] border border-[var(--color-line)] bg-[rgba(10,10,11,0.85)] px-1.5 py-0.5 font-mono text-[10.5px] text-[var(--color-fg-2)] transition-colors hover:text-[var(--color-fg)]"
                  >
                    {f.timestamp_label}
                  </button>
                </div>
                <p className="line-clamp-2 px-2 py-1.5 text-[11.5px] leading-snug text-[var(--color-fg-3)]">
                  {f.on_screen_text || f.description}
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

      {answer.citations.length > 0 && (
        <section>
          <h4 className="label mb-1.5">Citations</h4>
          <div className="flex flex-col">
            {answer.citations.map((c, i) => (
              <div
                key={`${c.chunk_id}-${i}`}
                className="group flex items-start gap-2 rounded-[var(--radius)] px-2 py-1.5 transition-colors hover:bg-[var(--color-surface-2)]"
              >
                <button onClick={() => onSeek(c.start)} className="mono shrink-0 text-[11px] text-[var(--color-fg-3)] hover:text-[var(--color-fg)]">
                  {fmtTime(c.start)}
                </button>
                {c.speaker && (
                  <span className="mono shrink-0 text-[11px]" style={{ color: speakerColor(c.speaker) }}>
                    {c.speaker.replace('SPEAKER_', 'S')}
                  </span>
                )}
                <span className="flex-1 text-[12.5px] leading-snug text-[var(--color-fg-2)]">{c.quote}</span>
                <span className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100">
                  <CopyButton text={c.quote} what="Quote copied" />
                </span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

/** Turns `[mm:ss]` markers in generated prose into seek buttons. */
function RichText({ text, onSeek }: { text: string; onSeek: (t: number) => void }) {
  const parts = useMemo(() => {
    const out: (string | { ts: number; label: string })[] = []
    const re = /\[((?:\d+:)?\d{1,2}:\d{2}(?:\.\d)?)\]/g
    let last = 0
    let m: RegExpExecArray | null
    while ((m = re.exec(text))) {
      const seconds = parseTimestamp(m[1])
      if (seconds === null) continue
      if (m.index > last) out.push(text.slice(last, m.index))
      out.push({ ts: seconds, label: m[1] })
      last = m.index + m[0].length
    }
    if (last < text.length) out.push(text.slice(last))
    return out
  }, [text])

  return (
    <>
      {parts.map((p, i) =>
        typeof p === 'string' ? (
          <span key={i} className="whitespace-pre-wrap">{p}</span>
        ) : (
          <button key={i} className="cite" onClick={() => onSeek(p.ts)} title="Jump to this moment">{p.label}</button>
        ),
      )}
    </>
  )
}

function ComputationCard({ computation }: { computation: Computation }) {
  const series = computation.series
  const value = computation.result?.value
  const max = series?.values.length ? Math.max(...series.values) : 0
  const [showCode, setShowCode] = useState(false)

  return (
    <section className="rounded-[var(--radius)] border border-[var(--color-line-soft)] bg-[var(--color-surface-2)] p-3">
      <div className="mb-2.5 flex items-center gap-2">
        <span className="chip" style={{ color: MODALITY_COLOR.lexical, borderColor: 'rgba(224,175,104,0.3)' }}>
          computed
        </span>
        <span className="text-[11.5px] text-[var(--color-fg-3)]">{computation.explanation}</span>
        {series?.values.length ? (
          <span className="ml-auto">
            <CopyButton
              text={series.labels.map((l, i) => `${l}\t${series.values[i]}`).join('\n')}
              what="Series copied"
            />
          </span>
        ) : null}
      </div>

      {series?.values.length ? (
        <div className="mb-3 flex items-end gap-1.5">
          {series.values.map((v, i) => (
            <Tip key={i} label={`${series.labels[i] ?? i + 1}: ${v.toLocaleString()}`}>
              <div className="flex flex-1 flex-col items-center gap-1">
                <span className="mono text-[10.5px] text-[var(--color-fg-3)]">{fmtNumber(v)}</span>
                <div
                  className="w-full rounded-[2px] bg-[var(--color-fg-3)] transition-[height] duration-500"
                  style={{ height: `${Math.max(3, (v / max) * 56)}px` }}
                />
                <span className="mono text-[10px] text-[var(--color-fg-4)]">{series.labels[i] ?? i + 1}</span>
              </div>
            </Tip>
          ))}
        </div>
      ) : null}

      {value !== undefined && value !== null && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3">
          {typeof value === 'object'
            ? Object.entries(value).map(([k, v]) => (
                <div key={k} className="min-w-0">
                  <div className="label truncate">{k.replace(/_/g, ' ')}</div>
                  <div className="mono truncate text-[12.5px] text-[var(--color-fg)]">
                    {Array.isArray(v) ? v.map((x) => fmtNumber(Number(x))).join(', ') : fmtNumber(Number(v))}
                  </div>
                </div>
              ))
            : <div className="mono text-[15px] text-[var(--color-fg)]">{String(value)}</div>}
        </div>
      )}

      {computation.code && (
        <>
          <button
            className="mt-2.5 text-[11px] text-[var(--color-fg-4)] transition-colors hover:text-[var(--color-fg-2)]"
            onClick={() => setShowCode((v) => !v)}
          >
            {showCode ? 'hide' : 'show'} the program that produced this
          </button>
          {showCode && (
            <div className="relative mt-1.5">
              <pre className="mono overflow-x-auto rounded-[var(--radius-sm)] border border-[var(--color-line-soft)] bg-[var(--color-bg)] p-2.5 text-[11px] leading-relaxed text-[var(--color-fg-2)]">
                {computation.code}
              </pre>
              <span className="absolute right-1 top-1">
                <CopyButton text={computation.code} what="Code copied" />
              </span>
            </div>
          )}
        </>
      )}
    </section>
  )
}

/* ---------------------------------------------------------------- evidence */

function Evidence({ hits, onSeek, onOpenFrame }: {
  hits: ScoredHit[]; onSeek: (t: number) => void
  onOpenFrame?: (frames: Keyframe[], id: string) => void
}) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const [cursor, setCursor] = useState(0)
  const listRef = useRef<HTMLDivElement>(null)
  const frames = useMemo(() => hits.flatMap((h) => h.keyframes), [hits])

  useEffect(() => { setCursor(0) }, [hits])

  // Stepping through results is the common motion after a search, so the list
  // is focusable and arrow keys move a cursor that seeks as it goes.
  const move = (delta: number) => {
    setCursor((c) => {
      const next = Math.max(0, Math.min(hits.length - 1, c + delta))
      const hit = hits[next]
      if (hit?.chunk) onSeek(hit.chunk.span.start)
      listRef.current
        ?.querySelector(`[data-row="${next}"]`)
        ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
      return next
    })
  }

  if (!hits.length) return <Empty title="No evidence" body="Nothing in the index matched this question." />

  return (
    <div
      ref={listRef}
      className="flex flex-col gap-1.5 outline-none"
      tabIndex={0}
      role="listbox"
      aria-label="Retrieved evidence"
      aria-activedescendant={`evidence-${cursor}`}
      onKeyDown={(e) => {
        if (e.key === 'ArrowDown' || e.key === 'j') { e.preventDefault(); move(1) }
        else if (e.key === 'ArrowUp' || e.key === 'k') { e.preventDefault(); move(-1) }
        else if (e.key === 'Enter') {
          e.preventDefault()
          const hit = hits[cursor]
          if (hit?.chunk) onSeek(hit.chunk.span.start)
        }
      }}
    >
      <div className="flex items-center justify-between px-0.5 text-[10.5px] text-[var(--color-fg-4)]">
        <span>{hits.length} moments</span>
        <span className="flex items-center gap-1">
          <Kbd>up</Kbd><Kbd>dn</Kbd> step through <Kbd>enter</Kbd> jump
        </span>
      </div>
      {hits.map((h, i) => (
        <div
          key={h.chunk_id}
          id={`evidence-${i}`}
          data-row={i}
          role="option"
          aria-selected={i === cursor}
          onClick={() => setCursor(i)}
          className={`group cursor-default rounded-[var(--radius)] border p-2.5 transition-colors ${
            i === cursor
              ? 'border-[var(--color-line-strong)] bg-[var(--color-surface-2)]'
              : 'border-[var(--color-line-soft)] hover:border-[var(--color-line)]'
          }`}
        >
          <div className="flex items-start gap-2.5">
            {h.keyframes[0] && (
              <FrameImage
                path={h.keyframes[0].path}
                className="h-[46px] w-[82px] shrink-0 cursor-zoom-in rounded-[var(--radius-sm)] border border-[var(--color-line-soft)] object-cover"
                onClick={() => onOpenFrame?.(frames, h.keyframes[0].id)}
              />
            )}
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="mono text-[10.5px] text-[var(--color-fg-4)]">{i + 1}</span>
                <button className="mono text-[11.5px] text-[var(--color-fg-2)] hover:text-[var(--color-fg)]" onClick={() => onSeek(h.chunk?.span.start ?? 0)}>
                  {h.chunk?.label}
                </button>
                {h.chunk?.speakers.map((s) => (
                  <span key={s} className="mono text-[10.5px]" style={{ color: speakerColor(s) }}>
                    {s.replace('SPEAKER_', 'S')}
                  </span>
                ))}
                <Tip label="Fused relevance score">
                  <span className="mono ml-auto text-[10.5px] text-[var(--color-fg-4)]">{h.score.toFixed(4)}</span>
                </Tip>
                <span className="flex shrink-0 gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
                  <CopyButton text={h.chunk?.text ?? ''} what="Text copied" />
                  <Tip label={expanded === h.chunk_id ? 'Collapse' : 'Read the full chunk'}>
                    <Button variant="quiet" size="sm" icon onClick={() => setExpanded((v) => (v === h.chunk_id ? null : h.chunk_id))} aria-label="Expand">
                      <Icon.Expand />
                    </Button>
                  </Tip>
                </span>
              </div>
              <p className={`mt-1 text-[12.5px] leading-snug text-[var(--color-fg-2)] ${expanded === h.chunk_id ? '' : 'line-clamp-2'}`}>
                {h.chunk?.text}
              </p>
              <div className="mt-1.5 flex flex-wrap items-center gap-1">
                {Object.entries(h.ranks).sort((a, b) => a[1]! - b[1]!).map(([m, r]) => (
                  <ChannelChip key={m} modality={m} rank={r} />
                ))}
              </div>
              <div className="mt-1.5">
                <FusionBar fusion={h.fusion as Record<string, number>} />
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------- trace */

function Trace({ answer, events }: { answer: AnswerBundle; events: AgentEvent[] }) {
  const timings = useMemo(() => {
    const map: Record<string, number> = {}
    for (const e of events) {
      const ms = e.data?.elapsed_ms
      if (typeof ms === 'number' && e.node !== '__end__') map[e.node] = ms
    }
    return map
  }, [events])
  const total = Math.max(...Object.values(timings), 1)

  return (
    <div className="flex flex-col gap-5 text-[12.5px]">
      <section>
        <div className="mb-1.5 flex items-center gap-2">
          <h4 className="label">Plan</h4>
          <CopyButton text={JSON.stringify(answer.plan, null, 2)} what="Plan copied" />
        </div>
        <p className="mb-2 text-[var(--color-fg-3)]">{answer.plan.intent}</p>
        <div className="flex flex-col gap-1.5">
          {answer.plan.tasks.map((t) => (
            <div key={t.id} className="rounded-[var(--radius)] border border-[var(--color-line-soft)] p-2.5">
              <div className="flex items-center gap-2">
                <span className="mono text-[10.5px] text-[var(--color-fg-4)]">{t.id}</span>
                <span className="font-medium text-[var(--color-fg)]">{t.kind.replace(/_/g, ' ')}</span>
                {t.depends_on.length > 0 && (
                  <span className="mono text-[10px] text-[var(--color-fg-4)]">after {t.depends_on.join(', ')}</span>
                )}
              </div>
              {t.rationale && <p className="mt-1 text-[11.5px] text-[var(--color-fg-3)]">{t.rationale}</p>}
              {Object.keys(t.modality_bias).length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {Object.entries(t.modality_bias).map(([m, v]) => (
                    <span key={m} className="chip">
                      <ChannelDot modality={m} />
                      {MODALITY_LABEL[m] ?? m} x{v.toFixed(2)}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      <section>
        <h4 className="label mb-1.5">Node timings</h4>
        <div className="flex flex-col gap-1">
          {Object.entries(timings).map(([node, ms]) => (
            <div key={node} className="flex items-center gap-2.5">
              <span className="w-24 shrink-0 truncate text-[var(--color-fg-2)]">{node}</span>
              <span className="h-1 flex-1 overflow-hidden rounded-full bg-[var(--color-line)]">
                <span className="block h-full rounded-full bg-[var(--color-fg-3)]" style={{ width: `${(ms / total) * 100}%` }} />
              </span>
              <span className="mono w-14 shrink-0 text-right text-[11px] text-[var(--color-fg-4)]">{ms.toFixed(0)}ms</span>
            </div>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-1.5 flex items-center gap-2">
          <h4 className="label">Event log</h4>
          <CopyButton text={events.map((e) => `${e.seq} ${e.node} ${e.kind} ${e.message}`).join('\n')} what="Log copied" />
        </div>
        <div className="mono flex flex-col gap-0.5 rounded-[var(--radius)] border border-[var(--color-line-soft)] bg-[var(--color-bg)] p-2.5 text-[11px] text-[var(--color-fg-3)]">
          {events.map((e) => (
            <div key={e.seq} className="flex gap-2">
              <span className="w-5 text-right text-[var(--color-fg-4)]">{e.seq}</span>
              <span className={e.kind === 'error' ? 'text-[var(--color-critical)]' : 'text-[var(--color-fg-2)]'}>{e.node}</span>
              <span className="text-[var(--color-fg-4)]">{e.kind}</span>
              <span className="truncate">{e.message}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
