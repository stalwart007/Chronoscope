/** Mirrors the backend's pydantic models (see backend/app/core/types.py). */

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
export type Modality = 'text' | 'summary' | 'image' | 'lexical'

export interface TimeSpan {
  start: number
  end: number
  duration: number
}

export interface VideoSummary {
  id: string
  filename: string
  title: string
  status: JobStatus
  stage: string
  progress: number
  error: string
  duration: number
  width: number
  height: number
  fps: number
  size_bytes: number
  language: string
  summary: string
  speakers: string[]
  topics: string[]
  chapters: Chapter[]
  stats: Record<string, any>
  poster: string | null
  created_at: string
}

export interface Chapter {
  index: number
  start: number
  end: number
  title: string
  keywords: string[]
  chunk_ids: string[]
  speakers: string[]
  boundary_strength: number
}

export interface Scene {
  index: number
  span: TimeSpan
  cut_score: number
  static_ratio: number
  kind: 'cut' | 'fade' | 'static' | 'synthetic'
}

export interface Keyframe {
  id: string
  scene_index: number
  timestamp: number
  path: string
  width: number
  height: number
  phash: number
  quality: number
  sharpness: number
  entropy: number
  text_density: number
  is_slide: boolean
}

export interface Sentence {
  start: number
  end: number
  text: string
  speaker: string | null
}

export interface VideoChunk {
  id: string
  video_id: string
  index: number
  span: TimeSpan
  text: string
  summary: string
  speakers: string[]
  keyframe_ids: string[]
  scene_indices: number[]
  keywords: string[]
  sentences: Sentence[]
  speech_rate: number
  visual_activity: number
  token_estimate: number
  label: string
}

export interface Segment {
  index: number
  start: number
  end: number
  text: string
  speaker: string
  confidence: number
}

export interface Timeline {
  video: VideoSummary
  chapters: Chapter[]
  scenes: Scene[]
  keyframes: Keyframe[]
  chunks: VideoChunk[]
  segments: Segment[]
}

export interface ScoredHit {
  chunk_id: string
  video_id: string
  score: number
  ranks: Partial<Record<Modality, number>>
  raw_scores: Partial<Record<Modality, number>>
  fusion: Partial<Record<Modality, number>>
  chunk: VideoChunk | null
  keyframes: Keyframe[]
  modalities: string[]
}

export interface RetrievalTrace {
  per_modality: Record<string, string[]>
  timings_ms: Record<string, number>
  fused_order: string[]
  mmr_order: string[]
  notes: string[]
}

export interface SubTask {
  id: string
  kind: string
  query: string
  depends_on: string[]
  rationale: string
  modality_bias: Record<string, number>
}

export interface QueryPlan {
  intent: string
  tasks: SubTask[]
  needs_computation: boolean
  needs_vision: boolean
  answer_style: 'timestamped' | 'narrative' | 'table' | 'numeric'
}

export interface Citation {
  chunk_id: string
  video_id: string
  start: number
  end: number
  speaker: string | null
  keyframe: string | null
  quote: string
  relevance: number
}

export interface AgentEvent {
  seq: number
  ts: string
  node: string
  kind: 'start' | 'log' | 'delta' | 'result' | 'error' | 'end'
  message: string
  data: Record<string, any>
}

export interface Computation {
  kind: string
  code?: string
  explanation?: string
  series?: { labels: string[]; values: number[]; unit: string }
  result?: { ok: boolean; value: any; error: string; elapsed_ms: number; variables: Record<string, any> }
  facts?: { value: number; raw: string; label: string; unit: string; source: string; context: string }[]
  source?: string
}

export interface VisualFinding {
  frame_id: string
  chunk_id: string
  timestamp: number
  timestamp_label: string
  image: string
  description: string
  on_screen_text: string
  data_points: { label: string; value: number; unit: string }[]
  answers_question: boolean
  confidence: number
  source: string
}

export interface AnswerBundle {
  query: string
  answer: string
  plan: QueryPlan
  citations: Citation[]
  hits: ScoredHit[]
  computations: Computation[]
  visual_findings: VisualFinding[]
  confidence: number
  elapsed_ms: number
  trace: AgentEvent[]
  model_used: string
  session_id: string
  /** The question after pronouns and references were resolved against the thread. */
  resolved_query: string
  is_followup: boolean
  /** Human-readable account of what was carried forward, shown on demand. */
  resolution_notes: string[]
  /** Set when the turn was rebuilt from a stored thread rather than just run,
   *  in which case retrieval evidence and the reasoning trace are not present. */
  restored?: boolean
}

export interface Turn {
  index: number
  query: string
  resolved_query: string
  answer: string
  citations: Citation[]
  confidence: number
  elapsed_ms: number
  created_at: string
}

export interface SessionSummary {
  id: string
  title: string
  video_ids: string[]
  turn_count: number
  created_at: string
  updated_at: string
}

export interface SessionDetail {
  session: SessionSummary
  turns: Turn[]
}

export interface GraphTopology {
  entry: string
  nodes: { id: string; label: string }[]
  edges: { from: string; to: string; kind: 'static' | 'conditional'; when?: string }[]
}

export interface Health {
  status: 'ok' | 'degraded'
  version: string
  env: string
  encoders: { text: any; image: any; notes: Record<string, string> }
  vector_store: Record<string, any>
  llm: { chain: string[]; any_available: boolean; providers: any[] }
  jobs: Record<string, any>
  lexical: Record<string, number>
  degraded: string[]
}
