import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'

interface Options {
  /** Default size as a fraction of the container. */
  fallback: number
  min: number
  max: number
  /** Container the fraction is measured against. */
  container: RefObject<HTMLElement | null>
  axis?: 'x' | 'y'
}

const FALLBACK_PX = 720
/**
 * Below this the measurement is not believable: a hidden or still-laying-out
 * container reports a few pixels, and scaling a fraction by that would collapse
 * the pane. A ResizeObserver corrects the size once real layout arrives.
 */
const MIN_TRUSTED_PX = 240

/**
 * A draggable pane size, persisted per key.
 *
 * Stored as a fraction rather than pixels. A pixel value captured at mount can
 * be wrong, since the container is often still settling, and it does not
 * survive a move to a different display. Fractions stay correct through both.
 */
/**
 * Measures the container rather than the window. ``window.innerHeight`` reads
 * zero while a tab is backgrounded, which would otherwise pin the pane to its
 * minimum on the first render.
 */
export function usePaneSize(key: string, { fallback, min, max, container, axis = 'y' }: Options) {
  const measure = useCallback(() => {
    const el = container.current
    const size = el ? (axis === 'y' ? el.clientHeight : el.clientWidth) : 0
    if (size >= MIN_TRUSTED_PX) return size
    const doc = axis === 'y' ? document.documentElement.clientHeight : document.documentElement.clientWidth
    return doc >= MIN_TRUSTED_PX ? doc : FALLBACK_PX
  }, [container, axis])

  const clampFraction = useCallback(
    (f: number) => Math.min(max, Math.max(min, f)),
    [min, max],
  )

  const [fraction, setFraction] = useState(() => {
    const saved = Number(localStorage.getItem(key))
    return Number.isFinite(saved) && saved > 0 ? clampFraction(saved) : fallback
  })
  const [basis, setBasis] = useState(FALLBACK_PX)

  const dragging = useRef(false)
  const origin = useRef({ pointer: 0, fraction: 0 })

  useEffect(() => {
    const update = () => setBasis(measure())
    update()
    const el = container.current
    // A ResizeObserver also fires when the pane becomes visible again, which a
    // window resize listener alone would miss.
    const observer = el ? new ResizeObserver(update) : null
    if (el && observer) observer.observe(el)
    window.addEventListener('resize', update)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', update)
    }
  }, [measure, container])

  const onGrab = useCallback(
    (e: React.PointerEvent, axis: 'x' | 'y') => {
      const base = measure()
      dragging.current = true
      origin.current = { pointer: axis === 'y' ? e.clientY : e.clientX, fraction }
      document.body.style.cursor = axis === 'y' ? 'row-resize' : 'col-resize'
      document.body.style.userSelect = 'none'

      const move = (ev: PointerEvent) => {
        if (!dragging.current) return
        const delta = (axis === 'y' ? ev.clientY : ev.clientX) - origin.current.pointer
        setFraction(clampFraction(origin.current.fraction + delta / base))
      }
      const up = () => {
        dragging.current = false
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
        window.removeEventListener('pointermove', move)
        window.removeEventListener('pointerup', up)
        setFraction((current) => {
          localStorage.setItem(key, String(current))
          return current
        })
      }
      window.addEventListener('pointermove', move)
      window.addEventListener('pointerup', up)
    },
    [clampFraction, fraction, key, measure],
  )

  const reset = useCallback(() => {
    setFraction(fallback)
    localStorage.setItem(key, String(fallback))
  }, [fallback, key])

  return { size: Math.round(fraction * basis), fraction, onGrab, reset }
}
