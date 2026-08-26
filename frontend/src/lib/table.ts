import { useCallback, useMemo, useState } from 'react'

export type SortDir = 'asc' | 'desc'

/** Client-side sorting for the data tables. Numbers compare numerically. */
export function useSort<T>(rows: T[], initial: keyof T & string, initialDir: SortDir = 'asc') {
  const [key, setKey] = useState<string>(initial)
  const [dir, setDir] = useState<SortDir>(initialDir)

  const sorted = useMemo(() => {
    const copy = [...rows]
    copy.sort((a, b) => {
      const av = (a as Record<string, unknown>)[key]
      const bv = (b as Record<string, unknown>)[key]
      if (typeof av === 'number' && typeof bv === 'number') return dir === 'asc' ? av - bv : bv - av
      return dir === 'asc'
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av))
    })
    return copy
  }, [rows, key, dir])

  const toggle = useCallback((next: string) => {
    setKey((prev) => {
      if (prev === next) {
        setDir((d) => (d === 'asc' ? 'desc' : 'asc'))
        return prev
      }
      setDir('asc')
      return next
    })
  }, [])

  return { sorted, key, dir, toggle }
}
