/** One colour per modality, reused across every visualisation. */
export const MODALITY_COLOR: Record<string, string> = {
  text: '#22d3ee',
  summary: '#34d399',
  image: '#c084fc',
  lexical: '#fbbf24',
}

export const MODALITY_LABEL: Record<string, string> = {
  text: 'Transcript',
  summary: 'Summary',
  image: 'Vision',
  lexical: 'BM25',
}

/** Deterministic, well-separated hues for diarised speakers. */
const SPEAKER_HUES = [188, 268, 42, 150, 330, 20, 210, 96]

export function speakerColor(speaker?: string | null): string {
  if (!speaker) return '#63708f'
  let hash = 0
  for (let i = 0; i < speaker.length; i++) hash = (hash * 31 + speaker.charCodeAt(i)) >>> 0
  const hue = SPEAKER_HUES[hash % SPEAKER_HUES.length]
  return `hsl(${hue} 78% 66%)`
}

export function fmtTime(t: number): string {
  if (!Number.isFinite(t) || t < 0) return '0:00'
  const total = Math.floor(t)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`
}

export function fmtDuration(t: number): string {
  if (t >= 3600) return `${(t / 3600).toFixed(1)} h`
  if (t >= 60) return `${Math.round(t / 60)} min`
  return `${Math.round(t)} s`
}

export function fmtBytes(n: number): string {
  if (!n) return '-'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`
}

export function fmtNumber(n: number): string {
  if (!Number.isFinite(n)) return '-'
  const abs = Math.abs(n)
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return abs < 1 && abs > 0 ? n.toFixed(3) : n.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

export function relativeTime(iso: string): string {
  if (!iso) return ''
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ''
  const delta = (Date.now() - then) / 1000
  if (delta < 60) return 'just now'
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`
  return `${Math.floor(delta / 86400)}d ago`
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

/** Parse `[mm:ss]` / `[h:mm:ss]` citations out of generated answers. */
export function parseTimestamp(label: string): number | null {
  const m = label.match(/^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d))?$/)
  if (!m) return null
  const [, h, mm, ss, frac] = m
  return (h ? +h * 3600 : 0) + +mm * 60 + +ss + (frac ? +frac / 10 : 0)
}

