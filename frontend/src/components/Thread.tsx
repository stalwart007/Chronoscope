/** The conversation so far: what was asked, what came back, and what a
 *  follow-up was understood to mean. */

import { useState } from 'react'
import { fmtTime } from '../lib/format'
import { useStore } from '../lib/store'
import type { AnswerBundle } from '../lib/types'
import { Button, Icon, Tip } from './ui'

export function Thread({ onSeek, onAsk }: {
  onSeek: (t: number) => void
  onAsk: (q: string) => void
}) {
  const { thread, answer, openTurn, newThread } = useStore()
  if (thread.length < 2) return null

  // The turn on screen is rendered in full below; the thread shows its history.
  const currentIndex = thread.findIndex((t) => t === answer)

  return (
    <div className="surface shrink-0 animate-rise">
      <div className="flex h-9 items-center gap-2 border-b border-[var(--color-line-soft)] px-2.5">
        <span className="text-[11px] uppercase tracking-[0.08em] text-[var(--color-fg-4)]">
          Thread
        </span>
        <span className="chip">{thread.length} turns</span>
        <Tip label="Follow-ups are read against this thread. Start a new one to drop the context.">
          <Button variant="quiet" size="sm" className="ml-auto" onClick={newThread}>
            New thread
          </Button>
        </Tip>
      </div>

      <ol className="max-h-[30vh] overflow-y-auto p-1.5">
        {thread.map((turn, i) => (
          <TurnRow
            key={`${turn.session_id}-${i}`}
            turn={turn}
            index={i}
            current={i === currentIndex}
            onOpen={() => openTurn(i)}
            onSeek={onSeek}
            onReask={() => onAsk(turn.query)}
          />
        ))}
      </ol>
    </div>
  )
}

function TurnRow({ turn, index, current, onOpen, onSeek, onReask }: {
  turn: AnswerBundle
  index: number
  current: boolean
  onOpen: () => void
  onSeek: (t: number) => void
  onReask: () => void
}) {
  const [showResolution, setShowResolution] = useState(false)
  const carried = turn.is_followup && turn.resolved_query && turn.resolved_query !== turn.query

  return (
    <li
      className={`rounded-md px-2 py-1.5 transition-colors ${
        current ? 'bg-[var(--color-bg-3)]' : 'hover:bg-[var(--color-bg-2)]'
      }`}
    >
      <div className="flex items-start gap-2">
        <span className="mono mt-0.5 w-4 shrink-0 text-right text-[10.5px] text-[var(--color-fg-4)]">
          {index + 1}
        </span>

        <div className="min-w-0 flex-1">
          <button
            className="block w-full text-left text-[12.5px] leading-snug text-[var(--color-fg)] hover:underline"
            onClick={onOpen}
            title="Show this answer"
          >
            {turn.is_followup && (
              <Tip label="Read as a follow-up to the previous turn">
                <span className="mr-1 text-[var(--color-fg-4)]">&#8627;</span>
              </Tip>
            )}
            {turn.query}
          </button>

          <p className="mt-0.5 line-clamp-2 text-[11.5px] leading-snug text-[var(--color-fg-3)]">
            {turn.answer}
          </p>

          <div className="mt-1 flex flex-wrap items-center gap-1">
            {turn.citations.slice(0, 4).map((c, ci) => (
              <button
                key={`${c.chunk_id}-${ci}`}
                className="chip mono transition-colors hover:border-[var(--color-line-strong)] hover:text-[var(--color-fg)]"
                onClick={() => onSeek(c.start)}
                title={c.quote}
              >
                {fmtTime(c.start)}
              </button>
            ))}
            {carried && (
              <button
                className="chip text-[var(--color-fg-4)] transition-colors hover:text-[var(--color-fg-2)]"
                onClick={() => setShowResolution((v) => !v)}
                aria-expanded={showResolution}
              >
                {showResolution ? 'hide' : 'read as'}
              </button>
            )}
            <Tip label="Ask this again as a fresh question">
              <button
                className="ml-auto shrink-0 rounded p-1 text-[var(--color-fg-4)] transition-colors hover:text-[var(--color-fg)]"
                onClick={onReask}
                aria-label="Ask again"
              >
                <Icon.Refresh />
              </button>
            </Tip>
          </div>

          {showResolution && carried && (
            <div className="mt-1.5 rounded border border-[var(--color-line-soft)] bg-[var(--color-bg-1)] p-1.5">
              <p className="text-[11.5px] leading-snug text-[var(--color-fg-2)]">
                {turn.resolved_query}
              </p>
              {turn.resolution_notes.length > 0 && (
                <ul className="mt-1 space-y-0.5">
                  {turn.resolution_notes.map((n) => (
                    <li key={n} className="text-[11px] text-[var(--color-fg-4)]">
                      {n}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </div>
    </li>
  )
}
