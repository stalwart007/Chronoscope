/**
 * Full-resolution keyframe viewer.
 *
 * Frames are thumbnails everywhere else, so this shows one at full size with
 * the measurements derived during ingestion, plus actions to seek the player
 * or save the image. Arrow keys move between frames without closing.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api'
import { copyText } from '../lib/download'
import { fmtTime } from '../lib/format'
import { useStore } from '../lib/store'
import type { Keyframe } from '../lib/types'
import { Button, Icon, Kbd, Stat, Tip } from './ui'

export function Lightbox({ frames, index, onIndex, onClose }: {
  frames: Keyframe[]
  index: number
  onIndex: (i: number) => void
  onClose: () => void
}) {
  const seek = useStore((s) => s.seek)
  const [loaded, setLoaded] = useState(false)
  const frame = frames[index]

  const go = useCallback(
    (delta: number) => {
      const next = (index + delta + frames.length) % frames.length
      setLoaded(false)
      onIndex(next)
    },
    [index, frames.length, onIndex],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose() }
      else if (e.key === 'ArrowRight') { e.preventDefault(); go(1) }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1) }
      else if (e.key === 'Enter') { e.preventDefault(); seek(frame.timestamp); onClose() }
    }
    // Capture phase: the global shortcut registry also listens for arrows, and
    // while the lightbox is open it must win.
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [go, onClose, seek, frame])

  const measurements = useMemo(
    () => [
      { label: 'Time', value: fmtTime(frame.timestamp), hint: 'Position in the source video' },
      { label: 'Scene', value: `#${frame.scene_index + 1}`, hint: 'Which detected scene this frame belongs to' },
      { label: 'Size', value: `${frame.width}x${frame.height}`, hint: 'Stored resolution' },
      { label: 'Quality', value: frame.quality.toFixed(3), hint: 'Composite of sharpness, entropy, exposure and text density, used to pick this frame over its neighbours' },
      { label: 'Sharpness', value: frame.sharpness.toFixed(4), hint: 'Variance of the Laplacian: high means crisp, low means motion blur' },
      { label: 'Entropy', value: frame.entropy.toFixed(2), hint: 'Shannon entropy of the luma histogram, visual complexity' },
      { label: 'Text density', value: frame.text_density.toFixed(4), hint: 'Fraction of strong fine-scale edges: a proxy for text and diagrams' },
      { label: 'Type', value: frame.is_slide ? 'Slide' : 'Shot', hint: 'Classified from text density and entropy' },
    ],
    [frame],
  )

  const download = async () => {
    const res = await fetch(api.frameUrl(frame.path))
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `frame-${fmtTime(frame.timestamp).replace(/[:.]/g, '-')}.jpg`
    a.click()
    requestAnimationFrame(() => URL.revokeObjectURL(url))
  }

  return (
    <div className="fixed inset-0 z-[80] flex flex-col bg-[rgba(7,7,8,0.975)] animate-fade-in" onClick={onClose}>
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-[var(--color-line-soft)] px-4">
        <span className="mono text-[13px] text-[var(--color-fg)]">{fmtTime(frame.timestamp)}</span>
        <span className="text-[12px] text-[var(--color-fg-3)]">
          Frame {index + 1} of {frames.length}
        </span>
        <div className="ml-auto flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
          <Button size="sm" onClick={() => { seek(frame.timestamp); onClose() }}>
            <Icon.Play /> Jump to moment
          </Button>
          <Button size="sm" onClick={download}><Icon.Download /> Save frame</Button>
          <Button size="sm" variant="quiet" onClick={() => copyText(api.frameUrl(frame.path), 'Frame URL copied')}>
            <Icon.Copy /> URL
          </Button>
          <Button size="sm" variant="quiet" icon onClick={onClose} aria-label="Close"><Icon.Close /></Button>
        </div>
      </header>

      <div className="relative flex min-h-0 flex-1 items-center justify-center p-6" onClick={(e) => e.stopPropagation()}>
        {frames.length > 1 && (
          <>
            <NavButton side="left" onClick={() => go(-1)} />
            <NavButton side="right" onClick={() => go(1)} />
          </>
        )}
        <img
          key={frame.id}
          src={api.frameUrl(frame.path)}
          alt={`Keyframe at ${fmtTime(frame.timestamp)}`}
          onLoad={() => setLoaded(true)}
          className={`max-h-full max-w-full rounded-[var(--radius-lg)] border border-[var(--color-line)] object-contain transition-opacity duration-200 ${
            loaded ? 'opacity-100' : 'opacity-0'
          }`}
        />
      </div>

      <footer
        className="shrink-0 border-t border-[var(--color-line-soft)] px-4 py-3"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4 lg:grid-cols-8">
          {measurements.map((m) => (
            <Stat key={m.label} label={m.label} value={m.value} hint={m.hint} />
          ))}
        </div>
        <div className="mt-3 flex items-center gap-3 border-t border-[var(--color-line-soft)] pt-2.5 text-[11px] text-[var(--color-fg-4)]">
          <span className="flex items-center gap-1"><Kbd>&larr;</Kbd><Kbd>&rarr;</Kbd> browse</span>
          <span className="flex items-center gap-1"><Kbd>↵</Kbd> jump to moment</span>
          <span className="flex items-center gap-1"><Kbd>esc</Kbd> close</span>
        </div>
      </footer>
    </div>
  )
}

function NavButton({ side, onClick }: { side: 'left' | 'right'; onClick: () => void }) {
  return (
    <Tip label={side === 'left' ? 'Previous frame' : 'Next frame'}>
      <button
        onClick={onClick}
        aria-label={side === 'left' ? 'Previous frame' : 'Next frame'}
        className={`absolute top-1/2 z-10 flex h-10 w-10 -translate-y-1/2 items-center justify-center rounded-full border border-[var(--color-line)] bg-[var(--color-surface-2)] text-[var(--color-fg-2)] transition-colors hover:border-[var(--color-line-strong)] hover:text-[var(--color-fg)] ${
          side === 'left' ? 'left-4' : 'right-4'
        }`}
      >
        <span className={side === 'left' ? 'rotate-180' : ''}><Icon.Chevron /></span>
      </button>
    </Tip>
  )
}
