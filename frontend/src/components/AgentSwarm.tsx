/**
 * Live view of the reasoning graph.
 *
 * The shape comes from `/api/system/graph` and node states from the event
 * stream, so this reflects what the server actually ran. Positions use
 * longest-path layering, which keeps every edge pointing forward.
 */

import { useMemo } from 'react'
import type { AgentEvent, GraphTopology } from '../lib/types'

type NodeState = 'idle' | 'running' | 'done' | 'error' | 'skipped'

const W = 560
const H = 176
const NODE_W = 104
const NODE_H = 40

export function AgentSwarm({
  topology, events, active,
}: { topology: GraphTopology | null; events: AgentEvent[]; active: boolean }) {
  const layout = useMemo(() => (topology ? computeLayout(topology) : null), [topology])

  const states = useMemo(() => {
    const map: Record<string, NodeState> = {}
    for (const ev of events) {
      if (ev.node === '__end__') continue
      if (ev.kind === 'start') map[ev.node] = 'running'
      else if (ev.kind === 'error') map[ev.node] = 'error'
      else if (ev.kind === 'result' || ev.kind === 'end') map[ev.node] = 'done'
    }
    return map
  }, [events])

  const timings = useMemo(() => {
    const map: Record<string, number> = {}
    for (const ev of events) {
      const ms = ev.data?.elapsed_ms
      if (typeof ms === 'number' && ev.node !== '__end__') map[ev.node] = ms
    }
    return map
  }, [events])

  const traversed = useMemo(() => {
    const done = new Set(Object.keys(states))
    return new Set(
      (topology?.edges ?? [])
        .filter((e) => done.has(e.from) && done.has(e.to))
        .map((e) => `${e.from}->${e.to}`),
    )
  }, [topology, states])

  if (!layout) {
    return <div className="skeleton h-[176px] rounded-[var(--radius-lg)]" />
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Agent reasoning graph">
      <defs>
        <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0 1.5 L7 4 L0 6.5 z" fill="rgba(255,255,255,0.28)" />
        </marker>
      </defs>

      {topology!.edges.map((e) => {
        const a = layout.nodes[e.from]
        const b = layout.nodes[e.to === '__end__' ? '__end__' : e.to]
        if (!a || !b) return null
        const isLive = traversed.has(`${e.from}->${e.to}`)
        const path = edgePath(a, b)
        return (
          <g key={`${e.from}-${e.to}-${e.when ?? ''}`}>
            <path
              d={path}
              fill="none"
              stroke={isLive ? 'rgba(255,255,255,0.42)' : 'rgba(255,255,255,0.11)'}
              strokeWidth={isLive ? 1.4 : 1}
              strokeDasharray={e.kind === 'conditional' ? '3 3' : undefined}
              markerEnd={isLive ? 'url(#arrow)' : undefined}
            />
          </g>
        )
      })}

      {Object.entries(layout.nodes).map(([id, pos]) => {
        if (id === '__end__') {
          return (
            <g key={id}>
              <circle cx={pos.x} cy={pos.y} r="6" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="1" />
              <circle cx={pos.x} cy={pos.y} r="2.5" fill={active ? 'rgba(255,255,255,0.25)' : 'var(--color-positive)'} />
            </g>
          )
        }
        const state = states[id] ?? 'idle'
        const label = topology!.nodes.find((n) => n.id === id)?.label ?? id
        // Node state is the only place the graph uses colour, and only three
        // states justify it: working, finished, failed.
        const color =
          state === 'running' ? '#f4f4f5'
          : state === 'done' ? '#93cc7e'
          : state === 'error' ? '#f07178'
          : '#303036'
        return (
          <g key={id}>
            <rect
              x={pos.x - NODE_W / 2}
              y={pos.y - NODE_H / 2}
              width={NODE_W}
              height={NODE_H}
              rx="7"
              fill={state === 'idle' ? '#0f0f11' : '#141417'}
              stroke={state === 'idle' ? '#232327' : color}
              strokeWidth="1"
              strokeOpacity={state === 'running' ? 1 : 0.55}
            >
              {state === 'running' && (
                <animate attributeName="stroke-opacity" values="0.3;1;0.3" dur="1.1s" repeatCount="indefinite" />
              )}
            </rect>
            <text
              x={pos.x} y={pos.y - 3} textAnchor="middle" fontSize="11" fontWeight="500"
              fill={state === 'idle' ? '#4a4a52' : '#f4f4f5'}
            >
              {label}
            </text>
            <text
              x={pos.x} y={pos.y + 10} textAnchor="middle" fontSize="9"
              fontFamily="JetBrains Mono, monospace"
              fill={state === 'idle' ? '#3a3a42' : color}
            >
              {state === 'running' ? 'working' : timings[id] !== undefined ? `${timings[id].toFixed(0)}ms` : state}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function computeLayout(topology: GraphTopology) {
  const ids = topology.nodes.map((n) => n.id)
  const all = [...ids, '__end__']
  const incoming: Record<string, string[]> = Object.fromEntries(all.map((id) => [id, []]))
  for (const e of topology.edges) incoming[e.to]?.push(e.from)

  // Longest-path layering, memoised, guarantees every edge points forward.
  const depth: Record<string, number> = {}
  const visiting = new Set<string>()
  const resolve = (id: string): number => {
    if (depth[id] !== undefined) return depth[id]
    if (visiting.has(id)) return 0
    visiting.add(id)
    const parents = incoming[id] ?? []
    depth[id] = parents.length ? Math.max(...parents.map(resolve)) + 1 : 0
    visiting.delete(id)
    return depth[id]
  }
  all.forEach(resolve)

  const layers: Record<number, string[]> = {}
  for (const id of all) (layers[depth[id]] ??= []).push(id)
  const maxDepth = Math.max(...Object.keys(layers).map(Number))
  const stepX = (W - NODE_W - 40) / Math.max(maxDepth, 1)

  const nodes: Record<string, { x: number; y: number }> = {}
  for (const [layerStr, members] of Object.entries(layers)) {
    const layer = Number(layerStr)
    const x = NODE_W / 2 + 14 + layer * stepX
    members.forEach((id, i) => {
      const span = H - 46
      const y = members.length === 1 ? H / 2 : 30 + (span / (members.length - 1)) * i
      nodes[id] = { x, y }
    })
  }
  return { nodes }
}

function edgePath(a: { x: number; y: number }, b: { x: number; y: number }) {
  const x1 = a.x + NODE_W / 2
  const x2 = b.x - NODE_W / 2
  const mid = (x1 + x2) / 2
  return `M ${x1} ${a.y} C ${mid} ${a.y}, ${mid} ${b.y}, ${x2} ${b.y}`
}
