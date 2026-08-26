import { toast } from './toast'

/** Trigger a browser download for content generated in the page. */
export function downloadBlob(filename: string, content: BlobPart, mime: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mime }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  // Revoke on the next frame: revoking synchronously races the download in
  // Safari, which has not yet read the blob when the click handler returns.
  requestAnimationFrame(() => URL.revokeObjectURL(url))
}

export function downloadJson(filename: string, value: unknown): void {
  downloadBlob(filename, JSON.stringify(value, null, 2), 'application/json')
}

export function downloadCsv(filename: string, rows: (string | number)[][], header: string[]): void {
  const escape = (cell: string | number) => {
    const text = String(cell ?? '')
    // Neutralise spreadsheet formula injection, then quote for CSV.
    const guarded = /^[=+\-@\t\r]/.test(text) ? `'${text}` : text
    return /[",\n]/.test(guarded) ? `"${guarded.replace(/"/g, '""')}"` : guarded
  }
  const body = [header, ...rows].map((r) => r.map(escape).join(',')).join('\n')
  downloadBlob(filename, body, 'text/csv;charset=utf-8')
}

/** Copy to the clipboard with a toast, falling back when the API is blocked. */
export async function copyText(text: string, what = 'Copied'): Promise<void> {
  try {
    await navigator.clipboard.writeText(text)
    toast.success(what)
  } catch {
    // Clipboard API needs a secure context; plain HTTP deployments land here.
    const area = document.createElement('textarea')
    area.value = text
    area.style.position = 'fixed'
    area.style.opacity = '0'
    document.body.appendChild(area)
    area.select()
    try {
      document.execCommand('copy')
      toast.success(what)
    } catch {
      toast.error('Could not copy', 'Your browser blocked clipboard access.')
    }
    area.remove()
  }
}

export function slug(text: string, fallback = 'chronoscope'): string {
  const cleaned = (text || '')
    .normalize('NFKD')
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80)
  return cleaned || fallback
}
