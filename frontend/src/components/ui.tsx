/** Interface primitives. Monochrome by default; colour is passed in explicitly. */

import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { api } from '../lib/api'
import { copyText } from '../lib/download'
import { MODALITY_COLOR, MODALITY_LABEL, clamp } from '../lib/format'

/* ------------------------------------------------------------------ layout */

export function Surface({ children, className = '', raised = false, ...rest }: {
  children: ReactNode; className?: string; raised?: boolean
} & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={`${raised ? 'surface-raised' : 'surface'} ${className}`} {...rest}>
      {children}
    </div>
  )
}

export function PanelHeader({ title, count, children }: { title: ReactNode; count?: number; children?: ReactNode }) {
  return (
    <div className="flex h-10 shrink-0 items-center gap-2 border-b border-[var(--color-line-soft)] px-3">
      <span className="label">{title}</span>
      {count !== undefined && <span className="mono text-[11px] text-[var(--color-fg-4)]">{count}</span>}
      <div className="ml-auto flex items-center gap-1">{children}</div>
    </div>
  )
}

/* ------------------------------------------------------------------ buttons */

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'default' | 'primary' | 'quiet' | 'danger'
  size?: 'md' | 'sm'
  icon?: boolean
}

export function Button({ variant = 'default', size = 'md', icon = false, className = '', ...rest }: ButtonProps) {
  const variants = { default: '', primary: 'btn-primary', quiet: 'btn-quiet', danger: 'btn-quiet btn-danger' }
  return (
    <button
      type="button"
      className={`btn ${variants[variant]} ${size === 'sm' ? 'btn-sm' : ''} ${icon ? 'btn-icon' : ''} ${className}`}
      {...rest}
    />
  )
}

export function Segmented<T extends string>({ value, options, onChange, className = '' }: {
  value: T
  options: { value: T; label: ReactNode; title?: string }[]
  onChange: (v: T) => void
  className?: string
}) {
  return (
    <div className={`segmented ${className}`} role="tablist">
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          data-active={value === o.value}
          title={o.title}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="kbd">{children}</kbd>
}

/* ------------------------------------------------------------------- menus */

export interface MenuItem {
  label: ReactNode
  hint?: string
  onSelect?: () => void
  danger?: boolean
  separator?: boolean
  disabled?: boolean
}

/**
 * A dropdown that closes on outside click, Escape, and scroll, and flips to
 * stay on screen. Missing any of the three leaves the popover stuck open.
 */
export function Menu({ trigger, items, align = 'right', width = 210 }: {
  trigger: (props: { open: boolean; toggle: () => void }) => ReactNode
  items: MenuItem[]
  align?: 'left' | 'right'
  width?: number
}) {
  const [open, setOpen] = useState(false)
  const [flipUp, setFlipUp] = useState(false)
  const root = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const close = (e: Event) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', onKey)
    window.addEventListener('scroll', () => setOpen(false), { once: true, capture: true })
    const rect = root.current?.getBoundingClientRect()
    setFlipUp(!!rect && window.innerHeight - rect.bottom < Math.min(items.length * 30 + 16, 320))
    return () => {
      document.removeEventListener('mousedown', close)
      document.removeEventListener('keydown', onKey)
    }
  }, [open, items.length])

  return (
    <div ref={root} className="relative inline-flex">
      {trigger({ open, toggle: () => setOpen((o) => !o) })}
      {open && (
        <div
          role="menu"
          style={{ width }}
          className={`surface-raised absolute z-50 animate-rise overflow-hidden p-1 ${
            flipUp ? 'bottom-full mb-1' : 'top-full mt-1'
          } ${align === 'right' ? 'right-0' : 'left-0'}`}
        >
          {items.map((item, i) =>
            item.separator ? (
              <div key={i} className="my-1 h-px bg-[var(--color-line-soft)]" />
            ) : (
              <button
                key={i}
                role="menuitem"
                disabled={item.disabled}
                onClick={() => { setOpen(false); item.onSelect?.() }}
                className={`flex w-full items-center gap-2 rounded-[5px] px-2 py-1.5 text-left text-[12.5px] transition-colors disabled:opacity-40 ${
                  item.danger
                    ? 'text-[var(--color-fg-2)] hover:bg-[rgba(240,113,120,0.12)] hover:text-[var(--color-critical)]'
                    : 'text-[var(--color-fg-2)] hover:bg-[var(--color-surface-3)] hover:text-[var(--color-fg)]'
                }`}
              >
                <span className="flex-1 truncate">{item.label}</span>
                {item.hint && <span className="mono shrink-0 text-[10.5px] text-[var(--color-fg-4)]">{item.hint}</span>}
              </button>
            ),
          )}
        </div>
      )}
    </div>
  )
}

/* ----------------------------------------------------------------- tooltips */

export function Tip({ label, children, side = 'top' }: { label: ReactNode; children: ReactNode; side?: 'top' | 'bottom' }) {
  const [show, setShow] = useState(false)
  const id = useId()
  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      onFocus={() => setShow(true)}
      onBlur={() => setShow(false)}
      aria-describedby={show ? id : undefined}
    >
      {children}
      {show && (
        <span
          id={id}
          role="tooltip"
          className={`surface-raised pointer-events-none absolute left-1/2 z-50 w-max max-w-[260px] -translate-x-1/2 animate-fade-in px-2 py-1.5 text-[11.5px] leading-snug text-[var(--color-fg-2)] ${
            side === 'top' ? 'bottom-full mb-1.5' : 'top-full mt-1.5'
          }`}
        >
          {label}
        </span>
      )}
    </span>
  )
}

/* ------------------------------------------------------------------- status */

export function Spinner({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className="animate-spin" aria-hidden>
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2.5" />
      <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  )
}

export function ProgressRing({ value, size = 32, stroke = 2.5 }: { value: number; size?: number; stroke?: number }) {
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90" aria-hidden>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--color-line)" strokeWidth={stroke} />
      <circle
        cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--color-fg)" strokeWidth={stroke}
        strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - clamp(value, 0, 100) / 100)}
        style={{ transition: 'stroke-dashoffset 400ms cubic-bezier(0.16,1,0.3,1)' }}
      />
    </svg>
  )
}

export function Meter({ value, label }: { value: number; label?: string }) {
  const pct = clamp(value, 0, 1)
  return (
    <Tip label={label ?? 'Confidence blends the retrieval score margin, how many channels agreed, and how much evidence was found'}>
      <span className="flex items-center gap-1.5">
        <span className="h-1 w-10 overflow-hidden rounded-full bg-[var(--color-line)] sm:w-16">
          <span
            className="block h-full rounded-full bg-[var(--color-fg-2)] transition-[width] duration-500"
            style={{ width: `${pct * 100}%` }}
          />
        </span>
        <span className="mono text-[11px] text-[var(--color-fg-2)]">{(pct * 100).toFixed(0)}%</span>
      </span>
    </Tip>
  )
}

export function ChannelDot({ modality }: { modality: string }) {
  return (
    <span
      className="inline-block h-[7px] w-[7px] shrink-0 rounded-full"
      style={{ background: MODALITY_COLOR[modality] ?? 'var(--color-fg-4)' }}
    />
  )
}

export function ChannelChip({ modality, rank }: { modality: string; rank?: number }) {
  return (
    <Tip label={`${MODALITY_LABEL[modality] ?? modality} channel${rank ? ` ranked this #${rank}` : ''}`}>
      <span className="chip">
        <ChannelDot modality={modality} />
        {MODALITY_LABEL[modality] ?? modality}
        {rank !== undefined && <span className="mono text-[var(--color-fg-4)]">{rank}</span>}
      </span>
    </Tip>
  )
}

/** Stacked bar of each channel's contribution to a fused score. */
export function FusionBar({ fusion, height = 3 }: { fusion: Record<string, number>; height?: number }) {
  const entries = Object.entries(fusion).filter(([, v]) => v > 0)
  const total = entries.reduce((s, [, v]) => s + v, 0) || 1
  if (!entries.length) return null
  return (
    <span className="flex w-full overflow-hidden rounded-full" style={{ height }} role="img" aria-label="channel contribution">
      {entries.sort((a, b) => b[1] - a[1]).map(([m, v]) => (
        <Tip key={m} label={`${MODALITY_LABEL[m] ?? m}: ${((v / total) * 100).toFixed(0)}% of the fused score`}>
          <span
            className="block h-full"
            style={{ width: `${(v / total) * 100}%`, background: MODALITY_COLOR[m] ?? 'var(--color-fg-4)', minWidth: 2 }}
          />
        </Tip>
      ))}
    </span>
  )
}

export function Stat({ label, value, hint, mono = true }: {
  label: string; value: ReactNode; hint?: string; mono?: boolean
}) {
  const body = (
    <div className="min-w-0">
      <div className="label mb-0.5 truncate">{label}</div>
      <div className={`${mono ? 'mono' : ''} truncate text-[14px] font-medium text-[var(--color-fg)]`}>{value}</div>
    </div>
  )
  return hint ? <Tip label={hint} side="bottom">{body}</Tip> : body
}

export function Empty({ title, body, action }: { title: string; body?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-8 py-14 text-center">
      <div className="text-[14px] font-medium text-[var(--color-fg-2)]">{title}</div>
      {body && <p className="max-w-sm text-[12.5px] leading-relaxed text-[var(--color-fg-3)]">{body}</p>}
      {action}
    </div>
  )
}

/* -------------------------------------------------------------------- media */

/**
 * Keyframe image, preferring the thumbnail written during ingestion.
 *
 * Frames are painted well under 200px in the filmstrip, evidence list and
 * hover preview. Falls back to the full frame if no thumbnail exists.
 */
export function FrameImage({ path, className, alt = '', full = false, onClick }: {
  path: string; className?: string; alt?: string; full?: boolean; onClick?: () => void
}) {
  const [src, setSrc] = useState(() => (full ? api.frameUrl(path) : api.thumbUrl(path)))
  useEffect(() => { setSrc(full ? api.frameUrl(path) : api.thumbUrl(path)) }, [path, full])
  return (
    <img
      src={src}
      alt={alt}
      className={className}
      loading="lazy"
      decoding="async"
      onClick={onClick}
      onError={() => setSrc(api.frameUrl(path))}
    />
  )
}

/* -------------------------------------------------------------------- table */

export function Th({ children = null, sortKey, active, dir, onSort, align = 'left', width }: {
  children?: ReactNode; sortKey?: string; active?: boolean; dir?: 'asc' | 'desc'
  onSort?: (k: string) => void; align?: 'left' | 'right'; width?: number
}) {
  return (
    <th
      data-sortable={!!sortKey}
      onClick={sortKey && onSort ? () => onSort(sortKey) : undefined}
      style={{ textAlign: align, width }}
      aria-sort={active ? (dir === 'asc' ? 'ascending' : 'descending') : undefined}
    >
      <span className="inline-flex items-center gap-1">
        {children}
        {active && <span className="text-[var(--color-fg)]">{dir === 'asc' ? '↑' : '↓'}</span>}
      </span>
    </th>
  )
}

/* -------------------------------------------------------------------- icons */

const stroke = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }

export const Icon = {
  Search: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><circle cx="7" cy="7" r="4.5" /><path d="m13.5 13.5-3-3" /></svg>),
  Play: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" {...p}><path d="M5 3.2v9.6l8-4.8z" /></svg>),
  Pause: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" {...p}><rect x="4" y="3" width="3" height="10" rx="1" /><rect x="9" y="3" width="3" height="10" rx="1" /></svg>),
  Upload: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M8 11V3m0 0L5 6m3-3 3 3" /><path d="M2.5 10.5v1.8a1.2 1.2 0 0 0 1.2 1.2h8.6a1.2 1.2 0 0 0 1.2-1.2v-1.8" /></svg>),
  Download: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M8 3v8m0 0 3-3m-3 3-3-3" /><path d="M2.5 12.5h11" /></svg>),
  Film: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><rect x="2" y="3" width="12" height="10" rx="1.5" /><path d="M5.5 3v10M10.5 3v10M2 8h12" /></svg>),
  Back: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M10 3 5 8l5 5" /></svg>),
  Trash: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.2h5.8l.6-8.2" /></svg>),
  Refresh: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9" /><path d="M13.5 2.5v3h-3" /></svg>),
  Copy: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><rect x="5.5" y="5.5" width="8" height="8" rx="1.5" /><path d="M10.5 5.5v-1a1.5 1.5 0 0 0-1.5-1.5H4a1.5 1.5 0 0 0-1.5 1.5v5A1.5 1.5 0 0 0 4 11h1" /></svg>),
  Check: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="m3.5 8.5 3 3 6-7" /></svg>),
  Warn: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M8 2.5 14.5 13.5h-13z" /><path d="M8 6.5v3M8 11.8v.2" /></svg>),
  Close: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="m4 4 8 8M12 4l-8 8" /></svg>),
  Ask: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M8 2.5 9.2 6l3.5 1.2-3.5 1.3L8 12l-1.2-3.5L3.3 7.2 6.8 6z" /></svg>),
  More: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" {...p}><circle cx="3.5" cy="8" r="1.2" /><circle cx="8" cy="8" r="1.2" /><circle cx="12.5" cy="8" r="1.2" /></svg>),
  Chevron: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="m6 4 4 4-4 4" /></svg>),
  External: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M9 3h4v4M13 3 7.5 8.5" /><path d="M12 9.5v3a.5.5 0 0 1-.5.5h-8a.5.5 0 0 1-.5-.5v-8a.5.5 0 0 1 .5-.5h3" /></svg>),
  Eye: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M1.8 8S4.2 4.2 8 4.2 14.2 8 14.2 8 11.8 11.8 8 11.8 1.8 8 1.8 8Z" /><circle cx="8" cy="8" r="1.8" /></svg>),
  EyeOff: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M6.2 4.6A6 6 0 0 1 8 4.2C11.8 4.2 14.2 8 14.2 8a11 11 0 0 1-2 2.2M9.9 9.9a2 2 0 0 1-2.8-2.8" /><path d="M4.4 5.6A11 11 0 0 0 1.8 8s2.4 3.8 6.2 3.8c.7 0 1.3-.1 1.9-.3" /><path d="m2.6 2.6 10.8 10.8" /></svg>),
  StepBack: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" {...p}><path d="M11.5 3.4v9.2L5.6 8z" /><rect x="3.4" y="3.4" width="1.6" height="9.2" rx="0.6" /></svg>),
  StepForward: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" {...p}><path d="M4.5 3.4v9.2L10.4 8z" /><rect x="11" y="3.4" width="1.6" height="9.2" rx="0.6" /></svg>),
  Volume: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M8 3 5 5.6H3v4.8h2L8 13z" /><path d="M10.4 6.2a2.6 2.6 0 0 1 0 3.6M12.2 4.5a5 5 0 0 1 0 7" /></svg>),
  Muted: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M8 3 5 5.6H3v4.8h2L8 13z" /><path d="m10.6 6.4 3.2 3.2M13.8 6.4l-3.2 3.2" /></svg>),
  Expand: (p: { className?: string }) => (<svg viewBox="0 0 16 16" width="14" height="14" {...stroke} {...p}><path d="M6 2.5H2.5V6M10 13.5h3.5V10M13.5 6V2.5H10M2.5 10v3.5H6" /></svg>),
}

/* ------------------------------------------------------------ copy control */

export function CopyButton({ text, what, size = 'sm' }: { text: string; what?: string; size?: 'md' | 'sm' }) {
  const [done, setDone] = useState(false)
  return (
    <Tip label={what ?? 'Copy'}>
      <Button
        variant="quiet"
        size={size}
        icon
        onClick={(e) => {
          e.stopPropagation()
          void copyText(text, what)
          setDone(true)
          window.setTimeout(() => setDone(false), 1200)
        }}
        aria-label={what ?? 'Copy'}
      >
        {done ? <Icon.Check /> : <Icon.Copy />}
      </Button>
    </Tip>
  )
}
