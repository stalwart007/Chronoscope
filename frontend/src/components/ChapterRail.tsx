/**
 * Chapters as a proportional rail.
 *
 * Scenes track the picture and chunks track retrieval; neither tells you what
 * a recording is *about* at a glance. Chapters come from topic segmentation
 * over the chunk embeddings, and this lays them out proportionally so the
 * shape of the material is visible: one long argument, or six short items.
 */

import { fmtTime } from '../lib/format'
import type { Chapter } from '../lib/types'
import { Tip } from './ui'

export function ChapterRail({ chapters, duration, currentTime, onSeek }: {
  chapters: Chapter[]
  duration: number
  currentTime: number
  onSeek: (t: number) => void
}) {
  if (chapters.length < 2) return null
  const activeIndex = chapters.findIndex((c) => currentTime >= c.start && currentTime < c.end)

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <span className="label">Chapters</span>
        <span className="mono text-[10.5px] text-[var(--color-fg-4)]">{chapters.length}</span>
        {activeIndex >= 0 && (
          <span className="ml-auto truncate text-[11.5px] text-[var(--color-fg-2)]">
            {chapters[activeIndex].title}
          </span>
        )}
      </div>

      <div className="flex h-8 w-full gap-[2px] overflow-hidden rounded-[var(--radius-sm)]">
        {chapters.map((c, i) => {
          const width = Math.max(2, ((c.end - c.start) / Math.max(duration, 1)) * 100)
          const active = i === activeIndex
          return (
            <Tip
              key={c.index}
              label={
                <span className="block">
                  <span className="mono block text-[10.5px] text-[var(--color-fg-3)]">
                    {fmtTime(c.start)} - {fmtTime(c.end)}
                  </span>
                  <span className="block">{c.title}</span>
                  {c.speakers.length > 0 && (
                    <span className="mono block text-[10.5px] text-[var(--color-fg-4)]">
                      {c.speakers.join(', ')}
                    </span>
                  )}
                </span>
              }
            >
              <button
                onClick={() => onSeek(c.start)}
                aria-label={`Chapter ${i + 1}: ${c.title}`}
                aria-current={active}
                style={{ width: `${width}%` }}
                className={`group relative flex h-8 min-w-0 shrink-0 items-center overflow-hidden px-2 text-left transition-colors ${
                  active
                    ? 'bg-[var(--color-surface-3)] text-[var(--color-fg)]'
                    : 'bg-[var(--color-surface-2)] text-[var(--color-fg-3)] hover:bg-[var(--color-surface-3)] hover:text-[var(--color-fg-2)]'
                }`}
              >
                {/* A stronger topic shift gets a brighter leading edge. */}
                {i > 0 && (
                  <span
                    className="absolute left-0 top-0 h-full w-[2px]"
                    style={{ background: `rgba(122,162,247,${0.25 + 0.6 * c.boundary_strength})` }}
                  />
                )}
                <span className="truncate text-[11px]">{c.title}</span>
              </button>
            </Tip>
          )
        })}
      </div>
    </div>
  )
}
