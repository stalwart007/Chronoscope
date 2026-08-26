/**
 * Global keyboard registry.
 *
 * One listener and one table, so the help sheet is generated from the same
 * source that handles the keys and cannot drift out of sync.
 */

import { useEffect } from 'react'

export interface Shortcut {
  /** Lowercase `event.key`, optionally prefixed with `mod+` / `shift+`. */
  combo: string
  label: string
  group: 'Global' | 'Playback' | 'Search'
  run: (e: KeyboardEvent) => void
  /** Allow while a text field has focus (default false). */
  whileTyping?: boolean
}

const registry = new Map<string, Shortcut>()

function comboOf(e: KeyboardEvent): string {
  const mod = e.metaKey || e.ctrlKey ? 'mod+' : ''
  const shift = e.shiftKey && e.key.length > 1 ? 'shift+' : ''
  return `${mod}${shift}${e.key.toLowerCase()}`
}

function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  if (!el) return false
  return (
    el.tagName === 'INPUT' ||
    el.tagName === 'TEXTAREA' ||
    el.tagName === 'SELECT' ||
    el.isContentEditable
  )
}

function onKeyDown(e: KeyboardEvent): void {
  const shortcut = registry.get(comboOf(e))
  if (!shortcut) return
  if (isTyping(e.target) && !shortcut.whileTyping) return
  e.preventDefault()
  shortcut.run(e)
}

let listening = false

export function useShortcuts(shortcuts: Shortcut[]): void {
  useEffect(() => {
    for (const s of shortcuts) registry.set(s.combo, s)
    if (!listening) {
      window.addEventListener('keydown', onKeyDown)
      listening = true
    }
    return () => {
      for (const s of shortcuts) {
        if (registry.get(s.combo) === s) registry.delete(s.combo)
      }
    }
  }, [shortcuts])
}

export function registeredShortcuts(): Shortcut[] {
  return [...registry.values()]
}

const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)

/** Render a combo the way this platform's users expect to read it. */
export function prettyCombo(combo: string): string {
  return combo
    .replace('mod+', IS_MAC ? '⌘' : 'Ctrl ')
    .replace('shift+', IS_MAC ? '⇧' : 'Shift ')
    .replace('arrowright', '\u2192')
    .replace('arrowleft', '\u2190')
    .replace('arrowup', '\u2191')
    .replace('arrowdown', '\u2193')
    .replace(' ', 'Space')
    .replace(/^(\w)$/, (m) => m.toUpperCase())
}
