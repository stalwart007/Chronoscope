import { create } from 'zustand'
import { api } from './api'
import type { AnswerBundle, Health, SessionSummary, Timeline, Turn, VideoSummary } from './types'

interface State {
  // library
  videos: VideoSummary[]
  loadingVideos: boolean
  libraryError: string | null

  // studio
  selectedId: string | null
  timeline: Timeline | null
  loadingTimeline: boolean

  // playback (shared so citations, ribbon and transcript stay in lockstep)
  currentTime: number
  duration: number
  playing: boolean
  seekRequest: { t: number; nonce: number } | null

  // answers
  answer: AnswerBundle | null

  // conversation: the thread the next question will be read against
  sessionId: string | null
  thread: AnswerBundle[]
  sessions: SessionSummary[]
  health: Health | null
  healthReachable: boolean

  view: 'library' | 'studio'

  loadVideos: () => Promise<void>
  loadHealth: () => Promise<void>
  select: (id: string | null) => Promise<void>
  refreshTimeline: () => Promise<void>
  setTime: (t: number) => void
  setDuration: (d: number) => void
  setPlaying: (p: boolean) => void
  seek: (t: number) => void
  setAnswer: (a: AnswerBundle | null) => void
  recordTurn: (a: AnswerBundle) => void
  newThread: () => void
  openTurn: (index: number) => void
  loadSessions: () => Promise<void>
  /** Resolves false when the thread's video is no longer in the library. */
  resumeSession: (id: string) => Promise<boolean>
  forgetSession: (id: string) => Promise<void>
  patchVideo: (id: string, patch: Partial<VideoSummary>) => void
  setView: (v: 'library' | 'studio') => void
}

export const useStore = create<State>((set, get) => ({
  videos: [],
  loadingVideos: false,
  libraryError: null,
  selectedId: null,
  timeline: null,
  loadingTimeline: false,
  currentTime: 0,
  duration: 0,
  playing: false,
  seekRequest: null,
  answer: null,
  sessionId: null,
  thread: [],
  sessions: [],
  health: null,
  healthReachable: true,
  view: 'library',

  async loadVideos() {
    set({ loadingVideos: true, libraryError: null })
    try {
      const { videos } = await api.listVideos()
      set({ videos, libraryError: null })
      // Health polls on a 30s timer. After a recovery the badge would sit on
      // "offline" until the next tick, so refresh it as soon as any call works.
      if (!get().healthReachable) void get().loadHealth()
    } catch (e) {
      // Keep whatever was already loaded. Replacing it with an empty list
      // would claim the library is empty when the server is merely down.
      set({ libraryError: e instanceof Error ? e.message : 'Failed to load the library' })
    } finally {
      set({ loadingVideos: false })
    }
  },

  async loadHealth() {
    try {
      set({ health: await api.health(), healthReachable: true })
    } catch {
      // Health is advisory and must never block the UI, but the badge should
      // still show that the engine is unreachable rather than "checking".
      set({ healthReachable: false })
    }
  },

  async select(id) {
    if (!id) {
      set({ selectedId: null, timeline: null, view: 'library' })
      return
    }
    set({ selectedId: id, loadingTimeline: true, view: 'studio', currentTime: 0, playing: false })
    try {
      const timeline = await api.timeline(id)
      set({ timeline, duration: timeline.video.duration })
    } catch {
      set({ timeline: null })
    } finally {
      set({ loadingTimeline: false })
    }
  },

  async refreshTimeline() {
    const id = get().selectedId
    if (!id) return
    try {
      const timeline = await api.timeline(id)
      set({ timeline, duration: timeline.video.duration })
    } catch { /* keep the stale timeline rather than blanking the studio */ }
  },

  setTime: (t) => set({ currentTime: t }),
  setDuration: (d) => set({ duration: d }),
  setPlaying: (p) => set({ playing: p }),
  seek: (t) => set({ seekRequest: { t, nonce: Date.now() + Math.random() }, currentTime: t }),
  setAnswer: (answer) => set({ answer }),

  /** Append a finished answer to the thread, or replace it if it is a re-ask. */
  recordTurn: (a) =>
    set((st) => {
      const thread = st.thread.filter((t) => t.session_id === a.session_id)
      const at = thread.findIndex((t) => t.query === a.query && t.elapsed_ms === a.elapsed_ms)
      if (at >= 0) thread[at] = a
      else thread.push(a)
      return { thread, sessionId: a.session_id || st.sessionId, answer: a }
    }),

  newThread: () => set({ sessionId: null, thread: [], answer: null }),

  /** Bring an earlier turn back into the main view without losing the thread. */
  openTurn: (index) => {
    const t = get().thread[index]
    if (t) set({ answer: t })
  },

  async loadSessions() {
    try {
      set({ sessions: await api.sessions() })
    } catch {
      set({ sessions: [] })
    }
  },

  /** Replay a stored session. Turns are summaries, so evidence is not restored. */
  async resumeSession(id) {
    const detail = await api.session(id)
    const thread = detail.turns.map((t: Turn) => rehydrate(t, id))
    set({ sessionId: id, thread, answer: thread[thread.length - 1] ?? null })

    // Open the video the thread is about so its citations lead somewhere. A
    // thread asked across the whole library records no video of its own, so
    // fall back to whatever its answers actually cited. Never navigate to a
    // video that is gone: that strands the studio on a timeline it cannot load.
    if (!get().videos.length) await get().loadVideos()
    const known = new Set(get().videos.map((v) => v.id))
    const candidates = [
      ...detail.session.video_ids,
      ...thread.flatMap((t) => t.citations.map((c) => c.video_id)),
    ]
    const target = candidates.find((v) => known.has(v))
    if (!target) return false
    if (target !== get().selectedId) await get().select(target)
    return true
  },

  async forgetSession(id) {
    await api.deleteSession(id)
    set((st) => ({
      sessions: st.sessions.filter((s) => s.id !== id),
      ...(st.sessionId === id ? { sessionId: null, thread: [], answer: null } : {}),
    }))
  },
  patchVideo: (id, patch) =>
    set((s) => ({ videos: s.videos.map((v) => (v.id === id ? { ...v, ...patch } : v)) })),
  setView: (view) => set({ view }),
}))

/** A stored turn carries the answer and its citations, but not the retrieval
 *  evidence behind it. Rebuild just enough of a bundle to render the thread. */
function rehydrate(t: Turn, sessionId: string): AnswerBundle {
  return {
    query: t.query,
    answer: t.answer,
    plan: { intent: '', answer_style: '', tasks: [], modalities: [], keywords: [], time_hints: [] } as any,
    citations: t.citations,
    hits: [],
    computations: [],
    visual_findings: [],
    confidence: t.confidence,
    elapsed_ms: t.elapsed_ms,
    trace: [],
    model_used: '',
    session_id: sessionId,
    resolved_query: t.resolved_query,
    is_followup: Boolean(t.resolved_query) && t.resolved_query !== t.query,
    resolution_notes: [],
    restored: true,
  }
}
