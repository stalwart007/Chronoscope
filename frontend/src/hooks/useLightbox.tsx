import { useCallback, useState } from 'react'
import { Lightbox } from '../components/Lightbox'
import type { Keyframe } from '../lib/types'

/** Shared lightbox state, so any frame in any panel opens the same viewer. */
export function useLightbox() {
  const [state, setState] = useState<{ frames: Keyframe[]; index: number } | null>(null)

  const open = useCallback((frames: Keyframe[], frameId: string) => {
    const index = Math.max(0, frames.findIndex((f) => f.id === frameId))
    setState({ frames, index })
  }, [])

  const node = state ? (
    <Lightbox
      frames={state.frames}
      index={state.index}
      onIndex={(index) => setState((s) => (s ? { ...s, index } : s))}
      onClose={() => setState(null)}
    />
  ) : null

  return { open, node }
}
