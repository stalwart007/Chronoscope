/** Cross-cutting surfaces: toasts, command palette, shortcut help. */

import { useEffect, useMemo, useRef, useState } from 'react'
import { fmtDuration } from '../lib/format'
import { prettyCombo, registeredShortcuts } from '../lib/shortcuts'
import { useStore } from '../lib/store'
import { useToasts } from '../lib/toast'
import { Button, Icon, Kbd, Spinner } from './ui'

/* -------------------------------------------------------------------- toasts */

export function Toasts() {
  const { toasts, dismiss } = useToasts()
  if (!toasts.length) return null
  return (
    <div
      className="pointer-events-none fixed bottom-4 right-4 z-[90] flex w-[min(360px,calc(100vw-2rem))] flex-col gap-1.5"
      role="status"
      aria-live="polite"
    >
      {toasts.map((t) => {
        const tone =
          t.kind === 'error' ? 'var(--color-critical)'
          : t.kind === 'success' ? 'var(--color-positive)'
          : 'var(--color-fg-3)'
        return (
          <div key={t.id} className="surface-raised pointer-events-auto flex animate-rise items-start gap-2.5 p-2.5">
            <span className="mt-0.5 shrink-0" style={{ color: tone }}>
              {t.kind === 'progress' ? <Spinner size={13} /> : t.kind === 'error' ? <Icon.Warn /> : <Icon.Check />}
            </span>
            <div className="min-w-0 flex-1">
              <div className="text-[12.5px] font-medium leading-snug">{t.title}</div>
              {t.body && <div className="mt-0.5 text-[11.5px] leading-relaxed text-[var(--color-fg-3)]">{t.body}</div>}
              {t.action && (
                <Button variant="quiet" size="sm" className="mt-1.5 !px-1.5" onClick={t.action.run}>
                  {t.action.label}
                </Button>
              )}
            </div>
            <Button variant="quiet" size="sm" icon onClick={() => dismiss(t.id)} aria-label="Dismiss">
              <Icon.Close />
            </Button>
          </div>
        )
      })}
    </div>
  )
}

/* ----------------------------------------------------------- command palette */

export interface Command {
  id: string
  label: string
  hint?: string
  group: string
  keywords?: string
  run: () => void
}

/**
 * Scored fuzzy subsequence match. Contiguous runs and word-start hits score
 * higher, so "adiag" ranks "architecture diagram" above coincidental matches.
 */
function fuzzyScore(needle: string, haystack: string): number {
  if (!needle) return 1
  const n = needle.toLowerCase()
  const h = haystack.toLowerCase()
  if (h.includes(n)) return 1000 - h.indexOf(n)
  let score = 0
  let hi = 0
  let streak = 0
  for (const ch of n) {
    const found = h.indexOf(ch, hi)
    if (found < 0) return -1
    streak = found === hi ? streak + 1 : 0
    score += 10 + streak * 5 + (found === 0 || h[found - 1] === ' ' ? 8 : 0)
    hi = found + 1
  }
  return score
}

export function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { videos, select, setView } = useStore()
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (open) {
      setQuery('')
      setCursor(0)
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [open])

  const commands = useMemo<Command[]>(() => {
    const items: Command[] = [
      {
        id: 'library',
        label: 'Go to library',
        group: 'Navigate',
        keywords: 'home videos back',
        run: () => { select(null); setView('library') },
      },
      {
        id: 'help',
        label: 'Show keyboard shortcuts',
        group: 'Help',
        keywords: 'keys hotkeys ?',
        run: () => window.dispatchEvent(new CustomEvent('chronoscope:help')),
      },
      {
        id: 'capabilities',
        label: 'Open engine capabilities',
        group: 'Help',
        keywords: 'health status degraded models',
        run: () => window.dispatchEvent(new CustomEvent('chronoscope:capabilities')),
      },
      {
        id: 'demo',
        label: 'Load the demo video',
        hint: 'Generates a sample talk and indexes it',
        group: 'Library',
        keywords: 'sample example try',
        run: () => window.dispatchEvent(new CustomEvent('chronoscope:demo')),
      },
      {
        id: 'docs',
        label: 'Open the API reference',
        group: 'Help',
        keywords: 'openapi swagger docs',
        run: () => window.open('/api/docs', '_blank', 'noopener'),
      },
    ]
    for (const v of videos) {
      items.push({
        id: `v:${v.id}`,
        label: v.title,
        hint: `${fmtDuration(v.duration)} · ${v.status}`,
        group: 'Videos',
        keywords: [v.filename, ...v.topics, ...v.speakers].join(' '),
        run: () => select(v.id),
      })
    }
    return items
  }, [videos, select, setView])

  const results = useMemo(() => {
    const scored = commands
      .map((c) => ({ c, score: fuzzyScore(query, `${c.label} ${c.keywords ?? ''}`) }))
      .filter((r) => r.score >= 0)
      .sort((a, b) => b.score - a.score)
    return scored.slice(0, 12).map((r) => r.c)
  }, [commands, query])

  useEffect(() => { setCursor(0) }, [query])
  useEffect(() => {
    listRef.current?.querySelector('[data-active="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [cursor, results])

  if (!open) return null

  const choose = (cmd?: Command) => {
    if (!cmd) return
    onClose()
    cmd.run()
  }

  return (
    <div className="fixed inset-0 z-[70] flex items-start justify-center pt-[12vh]" onClick={onClose}>
      <div className="absolute inset-0 bg-[rgba(6,6,7,0.7)] animate-fade-in" />
      <div
        className="surface-raised relative w-[min(600px,calc(100vw-2rem))] overflow-hidden animate-rise"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
      >
        <div className="flex items-center gap-2.5 border-b border-[var(--color-line-soft)] px-3.5 py-2.5">
          <span className="text-[var(--color-fg-4)]"><Icon.Search /></span>
          <input
            ref={inputRef}
            className="flex-1 bg-transparent text-[13.5px] outline-none placeholder:text-[var(--color-fg-4)]"
            placeholder="Search videos and commands..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') { e.preventDefault(); setCursor((c) => Math.min(c + 1, results.length - 1)) }
              else if (e.key === 'ArrowUp') { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)) }
              else if (e.key === 'Enter') { e.preventDefault(); choose(results[cursor]) }
              else if (e.key === 'Escape') onClose()
            }}
          />
          <Kbd>esc</Kbd>
        </div>

        <div ref={listRef} className="max-h-[46vh] overflow-y-auto p-1.5">
          {results.length === 0 && (
            <p className="px-3 py-8 text-center text-[12.5px] text-[var(--color-fg-4)]">
              Nothing matches "{query}".
            </p>
          )}
          {results.map((cmd, i) => {
            const first = i === 0 || results[i - 1].group !== cmd.group
            return (
              <div key={cmd.id}>
                {first && (
                  <div className="label px-2.5 pb-1 pt-2">{cmd.group}</div>
                )}
                <button
                  data-active={i === cursor}
                  onMouseEnter={() => setCursor(i)}
                  onClick={() => choose(cmd)}
                  className={`flex w-full items-center gap-3 rounded-[var(--radius-sm)] px-2.5 py-1.5 text-left transition-colors ${
                    i === cursor ? 'bg-[var(--color-surface-3)]' : 'hover:bg-[var(--color-surface-2)]'
                  }`}
                >
                  <span className="min-w-0 flex-1 truncate text-[12.5px]">{cmd.label}</span>
                  {cmd.hint && <span className="mono shrink-0 text-[10.5px] text-[var(--color-fg-4)]">{cmd.hint}</span>}
                </button>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------- shortcut help */

export function ShortcutHelp({ open, onClose }: { open: boolean; onClose: () => void }) {
  const shortcuts = useMemo(() => (open ? registeredShortcuts() : []), [open])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    if (open) window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null
  const groups = ['Global', 'Search', 'Playback'] as const

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-[rgba(6,6,7,0.7)] animate-fade-in" />
      <div
        className="surface-raised relative w-[min(540px,100%)] overflow-hidden animate-rise"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
      >
        <div className="flex h-11 items-center justify-between border-b border-[var(--color-line-soft)] px-3.5">
          <h3 className="label">Keyboard shortcuts</h3>
          <Button variant="quiet" size="sm" icon onClick={onClose} aria-label="Close"><Icon.Close /></Button>
        </div>
        <div className="max-h-[70vh] overflow-y-auto p-4">
          {groups.map((group) => {
            const rows = shortcuts.filter((s) => s.group === group)
            if (!rows.length) return null
            return (
              <div key={group} className="mb-4 last:mb-0">
                <div className="label mb-1.5">{group}</div>
                <div className="flex flex-col gap-1">
                  {rows.map((s) => (
                    <div key={s.combo} className="flex items-center justify-between gap-4 rounded-[var(--radius-sm)] px-2 py-1 hover:bg-[var(--color-surface-2)]">
                      <span className="text-[12.5px] text-[var(--color-fg-2)]">{s.label}</span>
                      <Kbd>{prettyCombo(s.combo)}</Kbd>
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
          <p className="mt-2 border-t border-[var(--color-line-soft)] pt-3 text-[11.5px] leading-relaxed text-[var(--color-fg-4)]">
            Timestamps in an answer are buttons, click one to jump the player there. The ribbon is
            draggable and its filmstrip opens frames full size. Every table sorts, and every panel exports.
          </p>
        </div>
      </div>
    </div>
  )
}
