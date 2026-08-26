import { create } from 'zustand'

export type ToastKind = 'info' | 'success' | 'error' | 'progress'

export interface Toast {
  id: string
  kind: ToastKind
  title: string
  body?: string
  /** Optional inline action, e.g. "Undo" or "Open". */
  action?: { label: string; run: () => void }
  /** Milliseconds before auto-dismiss; 0 pins the toast until dismissed. */
  ttl: number
}

interface ToastState {
  toasts: Toast[]
  push: (t: Omit<Toast, 'id' | 'ttl'> & { ttl?: number }) => string
  update: (id: string, patch: Partial<Omit<Toast, 'id'>>) => void
  dismiss: (id: string) => void
}

let seq = 0

export const useToasts = create<ToastState>((set, get) => ({
  toasts: [],
  push(t) {
    const id = `t${++seq}`
    const ttl = t.ttl ?? (t.kind === 'error' ? 9000 : 4500)
    set((s) => ({ toasts: [...s.toasts.slice(-4), { ...t, id, ttl }] }))
    if (ttl > 0) setTimeout(() => get().dismiss(id), ttl)
    return id
  },
  update(id, patch) {
    set((s) => ({ toasts: s.toasts.map((t) => (t.id === id ? { ...t, ...patch } : t)) }))
    if (patch.ttl && patch.ttl > 0) setTimeout(() => get().dismiss(id), patch.ttl)
  },
  dismiss(id) {
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }))
  },
}))

/** Imperative helpers so callers don't need the hook. */
export const toast = {
  info: (title: string, body?: string) => useToasts.getState().push({ kind: 'info', title, body }),
  success: (title: string, body?: string) => useToasts.getState().push({ kind: 'success', title, body }),
  error: (title: string, body?: string) => useToasts.getState().push({ kind: 'error', title, body }),
  /** A pinned toast the caller resolves later, used for long operations. */
  progress: (title: string, body?: string) =>
    useToasts.getState().push({ kind: 'progress', title, body, ttl: 0 }),
  update: (id: string, patch: Partial<Omit<Toast, 'id'>>) => useToasts.getState().update(id, patch),
  dismiss: (id: string) => useToasts.getState().dismiss(id),
}

/**
 * Map a thrown value to a message and a next step.
 *
 * Raw API codes like "unsupported_media" are accurate but not actionable, so
 * each known code is paired with what to do about it.
 */
export function explainError(err: unknown): { title: string; body: string } {
  const anyErr = err as { code?: string; message?: string; status?: number }
  const code = anyErr?.code ?? ''
  const message = anyErr?.message ?? String(err ?? 'Something went wrong')
  const guidance: Record<string, { title: string; body: string }> = {
    unsupported_media: {
      title: "That file is not a video we can read",
      body: `${message}. Try an MP4, MOV, MKV or WebM, re-encode with: ffmpeg -i input output.mp4`,
    },
    quota_exceeded: {
      title: 'Storage is full',
      body: `${message}. Delete a video from the library, or raise CS_STORAGE_QUOTA_GB.`,
    },
    disk_full: { title: 'Not enough disk space', body: message },
    video_too_long: { title: 'Video is too long', body: `${message}. Trim it, or raise CS_MAX_VIDEO_DURATION_S.` },
    resolution_too_high: { title: 'Frame size is too large', body: message },
    rate_limited: { title: 'Slow down a moment', body: `${message}. This protects the server from overload.` },
    unauthorized: { title: 'API key required', body: 'Set the key this server expects, then reload.' },
    payload_too_large: { title: 'Request too large', body: message },
    validation_error: { title: "That request was not valid", body: 'Check the highlighted fields and try again.' },
    needs_extractor: {
      title: 'That is a web page, not a video file',
      body: 'Video sites serve HTML. Download the file first, then upload it, or paste a direct link to the media.',
    },
    blocked_address: {
      title: 'That address is not allowed',
      body: 'Private, loopback and reserved addresses are blocked so the server cannot be used to reach internal services.',
    },
    url_scheme: { title: 'Unsupported link', body: 'Only http and https URLs can be fetched.' },
    dns_failure: { title: 'Host not found', body: 'That domain could not be resolved. Check the link and try again.' },
    fetch_failed: { title: 'The download failed', body: 'The remote server refused the request.' },
    too_large: { title: 'File is too large', body: 'Raise CS_MAX_UPLOAD_MB, or trim the video first.' },
    too_many_redirects: { title: 'Too many redirects', body: 'The link bounced too many times to follow safely.' },
    offline: {
      title: 'Cannot reach the server',
      body: 'The API is not responding. Check that it is running, then retry.',
    },
    network_error: {
      title: 'Network error',
      body: 'The request could not be sent. Check your connection and retry.',
    },
    dependency_unavailable: {
      title: "A component is not available",
      body: `${message}. Check the capabilities panel for what's missing.`,
    },
    pipeline_failed: {
      title: 'Processing failed',
      body: `${message}. The original file is kept, you can re-index from the library.`,
    },
  }
  return guidance[code] ?? { title: 'Something went wrong', body: message }
}
