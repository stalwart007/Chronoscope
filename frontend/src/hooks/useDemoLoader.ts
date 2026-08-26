import { useMemo } from 'react'
import { api } from '../lib/api'
import { useStore } from '../lib/store'
import { explainError, toast } from '../lib/toast'

/** Shared "load the demo" flow, used by the empty state and the palette. */
export function useDemoLoader(onDone: () => void) {
  const select = useStore((s) => s.select)
  return useMemo(
    () => async () => {
      const id = toast.progress('Building the demo video', 'Rendering five scenes and two speakers.')
      try {
        const res = await api.loadDemo()
        toast.update(id, {
          kind: 'success',
          title: res.already_loaded ? 'Demo already indexed' : 'Demo queued',
          body: res.already_loaded ? 'Opening it now.' : 'Processing takes a few seconds.',
          ttl: 3500,
        })
        onDone()
        if (res.already_loaded) select(res.video_id)
      } catch (err) {
        const { title, body } = explainError(err)
        toast.update(id, { kind: 'error', title, body, ttl: 9000 })
      }
    },
    [onDone, select],
  )
}
