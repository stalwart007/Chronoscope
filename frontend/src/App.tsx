import { useCallback, useEffect, useMemo, useState } from 'react'
import { CommandPalette, ShortcutHelp, Toasts } from './components/Overlays'
import { Library } from './components/Library'
import { Studio, TopBar } from './components/Shell'
import { useShortcuts, type Shortcut } from './lib/shortcuts'
import { ErrorBoundary } from './components/ErrorBoundary'
import { useStore } from './lib/store'

export default function App() {
  const view = useStore((s) => s.view)
  const selectedId = useStore((s) => s.selectedId)
  const select = useStore((s) => s.select)
  const setPlaying = useStore((s) => s.setPlaying)
  const seek = useStore((s) => s.seek)

  const [palette, setPalette] = useState(false)
  const [help, setHelp] = useState(false)
  const [capabilities, setCapabilities] = useState(false)

  // Deep-linkable studio: /#/v/<id> survives reloads and can be shared.
  useEffect(() => {
    const apply = () => {
      const match = window.location.hash.match(/^#\/v\/([0-9a-f]{32})/)
      const id = match?.[1] ?? null
      if (id !== useStore.getState().selectedId) select(id)
    }
    apply()
    window.addEventListener('hashchange', apply)
    return () => window.removeEventListener('hashchange', apply)
  }, [select])

  useEffect(() => {
    const next = selectedId ? `#/v/${selectedId}` : '#/'
    if (window.location.hash !== next) window.history.replaceState(null, '', next)
  }, [selectedId])

  // The palette dispatches intents as events so it never has to import every
  // surface it can reach, keeps the command list flat and dependency-free.
  useEffect(() => {
    const openHelp = () => setHelp(true)
    const openCaps = () => setCapabilities(true)
    window.addEventListener('chronoscope:help', openHelp)
    window.addEventListener('chronoscope:capabilities', openCaps)
    return () => {
      window.removeEventListener('chronoscope:help', openHelp)
      window.removeEventListener('chronoscope:capabilities', openCaps)
    }
  }, [])

  const nudge = useCallback(
    (delta: number) => {
      const { currentTime, duration } = useStore.getState()
      seek(Math.max(0, Math.min(duration || 0, currentTime + delta)))
    },
    [seek],
  )

  const shortcuts = useMemo<Shortcut[]>(
    () => [
      { combo: 'mod+k', label: 'Open the command palette', group: 'Global', whileTyping: true,
        run: () => setPalette((p) => !p) },
      { combo: '?', label: 'Show this help', group: 'Global', run: () => setHelp((h) => !h) },
      { combo: 'escape', label: 'Close the open panel', group: 'Global', whileTyping: true,
        run: () => { setPalette(false); setHelp(false); setCapabilities(false) } },
      { combo: 'g', label: 'Back to the library', group: 'Global', run: () => select(null) },
      { combo: '/', label: 'Focus the question box', group: 'Search',
        run: () => document.querySelector<HTMLInputElement>('[data-ask-input]')?.focus() },
      { combo: 'f', label: 'Filter the transcript', group: 'Search',
        run: () => document.querySelector<HTMLInputElement>('[data-transcript-filter]')?.focus() },
      { combo: 'v', label: 'Show / hide the video', group: 'Playback',
        run: () => window.dispatchEvent(new CustomEvent('chronoscope:toggle-video')) },
      { combo: 'm', label: 'Mute / unmute', group: 'Playback',
        run: () => { const v = document.querySelector<HTMLVideoElement>('video'); if (v) v.muted = !v.muted } },
      { combo: ' ', label: 'Play / pause', group: 'Playback',
        run: () => setPlaying(!useStore.getState().playing) },
      { combo: 'arrowright', label: 'Forward 5 seconds', group: 'Playback', run: () => nudge(5) },
      { combo: 'arrowleft', label: 'Back 5 seconds', group: 'Playback', run: () => nudge(-5) },
      { combo: 'shift+arrowright', label: 'Forward 30 seconds', group: 'Playback', run: () => nudge(30) },
      { combo: 'shift+arrowleft', label: 'Back 30 seconds', group: 'Playback', run: () => nudge(-30) },
    ],
    [nudge, select, setPlaying],
  )
  useShortcuts(shortcuts)

  return (
    <div className="relative z-10 flex h-full flex-col">
      <TopBar
        capabilitiesOpen={capabilities}
        onToggleCapabilities={() => setCapabilities((c) => !c)}
        onOpenPalette={() => setPalette(true)}
        onOpenHelp={() => setHelp(true)}
      />
      <main className="flex min-h-0 flex-1 flex-col">
        <ErrorBoundary label={view === 'studio' ? 'The studio stopped responding' : 'The library stopped responding'}>
          {view === 'studio' ? <Studio /> : <Library />}
        </ErrorBoundary>
      </main>

      <CommandPalette open={palette} onClose={() => setPalette(false)} />
      <ShortcutHelp open={help} onClose={() => setHelp(false)} />
      <Toasts />
    </div>
  )
}
