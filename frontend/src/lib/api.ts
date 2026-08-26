/**
 * Typed API client.
 *
 * JSON for CRUD, and Server-Sent Events for the two long-running flows:
 * ingestion progress and agent reasoning. Both are server-to-client only, and
 * SSE reconnects, replays and passes through proxies without extra handling.
 */

import type { AgentEvent, AnswerBundle, GraphTopology, Health, RetrievalTrace, ScoredHit, SessionDetail, SessionSummary, Timeline, VideoSummary } from './types'

const BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public detail?: unknown) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...init?.headers },
    })
  } catch {
    // fetch only rejects for transport failures: the server is unreachable.
    throw new ApiError(0, 'offline', 'Cannot reach the server')
  }
  if (!res.ok) {
    let code = 'http_error'
    let message = `Request failed (${res.status})`
    let detail: unknown
    try {
      const body = await res.json()
      code = body?.error?.code ?? code
      message = body?.error?.message ?? message
      detail = body?.error?.detail
    } catch {
      // A non-JSON body means the response came from a proxy or crashed
      // process, not the API. Do not surface its markup to the user.
      if (res.status >= 500) {
        code = 'offline'
        message = 'The server is not responding'
      }
    }
    throw new ApiError(res.status, code, message, detail)
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T)
}

export const api = {
  health: () => request<Health>('/api/system/health'),
  stats: () => request<Record<string, any>>('/api/system/stats'),
  config: () => request<Record<string, any>>('/api/system/config'),
  graph: () => request<GraphTopology>('/api/system/graph'),

  sessions: () => request<SessionSummary[]>('/api/sessions'),
  session: (id: string) => request<SessionDetail>(`/api/sessions/${encodeURIComponent(id)}`),
  deleteSession: (id: string) =>
    request<{ deleted: string }>(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }),

  listVideos: () => request<{ videos: VideoSummary[]; total: number }>('/api/videos'),
  getVideo: (id: string) => request<VideoSummary>(`/api/videos/${id}`),
  timeline: (id: string) => request<Timeline>(`/api/videos/${id}/timeline`),
  deleteVideo: (id: string) => request<{ deleted: string }>(`/api/videos/${id}`, { method: 'DELETE' }),
  reindex: (id: string) => request<{ video_id: string }>(`/api/videos/${id}/reindex`, { method: 'POST' }),
  mediaUrl: (id: string) => `${BASE}/api/videos/${id}/media`,
  frameUrl: (path: string) => `${BASE}/frames/${path}`,
  posterUrl: (path: string) => `${BASE}${path}`,

  upload(
    file: File,
    opts: { title?: string; transcript?: File | null; onProgress?: (pct: number) => void } = {},
  ): Promise<{ video_id: string; status: string; duplicate: boolean; message: string }> {
    // XHR rather than fetch: upload progress events are not exposed by fetch.
    const form = new FormData()
    form.append('file', file)
    if (opts.title) form.append('title', opts.title)
    if (opts.transcript) form.append('transcript', opts.transcript)
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE}/api/videos`)
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) opts.onProgress?.(Math.round((e.loaded / e.total) * 100))
      }
      xhr.onload = () => {
        try {
          const body = JSON.parse(xhr.responseText)
          if (xhr.status >= 400) {
            reject(new ApiError(xhr.status, body?.error?.code ?? 'upload_failed', body?.error?.message ?? 'Upload failed'))
          } else resolve(body)
        } catch {
          reject(new ApiError(xhr.status, 'bad_response', 'Malformed server response'))
        }
      }
      xhr.onerror = () => reject(new ApiError(0, 'network_error', 'Network error during upload'))
      xhr.onabort = () => reject(new ApiError(0, 'aborted', 'Upload cancelled'))
      xhr.send(form)
    })
  },

  ingestUrl: (url: string, title = '') =>
    request<{ video_id: string; status: string; duplicate: boolean; message: string }>(
      '/api/videos/from-url',
      { method: 'POST', body: JSON.stringify({ url, title }) },
    ),

  search: (body: Record<string, unknown>) =>
    request<{ hits: ScoredHit[]; trace: RetrievalTrace }>('/api/search', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  ask: (body: Record<string, unknown>) =>
    request<{ answer: AnswerBundle }>('/api/query', { method: 'POST', body: JSON.stringify(body) }),

  history: () => request<{ queries: any[] }>('/api/query/history'),

  loadDemo: () =>
    request<{ video_id: string; status: string; already_loaded: boolean }>('/api/system/demo', {
      method: 'POST',
    }),

  /** Small companion image written at ingest; falls back to the full frame. */
  thumbUrl: (path: string) => `${BASE}/frames/${path.replace(/\.jpg$/, '.thumb.jpg')}`,
}

/** Minimal SSE reader over `fetch`, gives us abort control that EventSource lacks. */
export async function readSSE(
  url: string,
  handlers: { [event: string]: (data: any) => void },
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}${url}`, { signal, headers: { Accept: 'text/event-stream' } })
  if (!res.ok || !res.body) throw new ApiError(res.status, 'stream_failed', 'Could not open event stream')
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let split: number
    while ((split = buffer.indexOf('\n\n')) >= 0) {
      const raw = buffer.slice(0, split)
      buffer = buffer.slice(split + 2)
      let event = 'message'
      const dataLines: string[] = []
      for (const line of raw.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7).trim()
        else if (line.startsWith('data: ')) dataLines.push(line.slice(6))
      }
      if (!dataLines.length) continue
      const handler = handlers[event] ?? handlers['*']
      if (!handler) continue
      try {
        handler(JSON.parse(dataLines.join('\n')))
      } catch {
        handler(dataLines.join('\n'))
      }
    }
  }
}

export function ingestStream(videoId: string, onEvent: (kind: string, data: any) => void, signal?: AbortSignal) {
  const kinds = [
    'hello', 'job_start', 'stage_start', 'stage_done', 'stage_error',
    'job_done', 'job_failed', 'job_cancelled',
  ]
  const handlers = Object.fromEntries(kinds.map((k) => [k, (d: any) => onEvent(k, d)]))
  return readSSE(`/api/videos/${videoId}/events`, handlers, signal)
}

export function askStream(
  params: { q: string; video_ids?: string[]; top_k?: number; session_id?: string },
  on: { open?: (d: any) => void; agent?: (e: AgentEvent) => void; answer?: (a: AnswerBundle) => void; error?: (e: any) => void; done?: () => void },
  signal?: AbortSignal,
) {
  const qs = new URLSearchParams({ q: params.q })
  if (params.video_ids?.length) qs.set('video_ids', params.video_ids.join(','))
  if (params.top_k) qs.set('top_k', String(params.top_k))
  if (params.session_id) qs.set('session_id', params.session_id)
  return readSSE(
    `/api/query/stream?${qs}`,
    {
      open: (d) => on.open?.(d),
      agent: (d) => on.agent?.(d),
      answer: (d) => on.answer?.(d),
      error: (d) => on.error?.(d),
      done: () => on.done?.(),
    },
    signal,
  )
}
