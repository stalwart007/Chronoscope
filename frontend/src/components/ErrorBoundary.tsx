import { Component, type ErrorInfo, type ReactNode } from 'react'
import { Button, Icon } from './ui'

interface Props {
  children: ReactNode
  /** Shown instead of the default panel, e.g. to keep a sidebar usable. */
  label?: string
}

interface State {
  error: Error | null
}

/**
 * Catches render errors so one broken panel does not blank the whole app.
 *
 * Without this, any throw below the root unmounts the tree and leaves a white
 * screen with no way back. Recovery is local: reset the boundary and keep the
 * rest of the session, including the loaded video and any answer, intact.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('render error', error, info.componentStack)
  }

  private reset = () => this.setState({ error: null })

  render(): ReactNode {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div className="surface flex min-h-0 flex-1 flex-col items-center justify-center gap-3 p-8 text-center">
        <Icon.Warn className="text-[var(--color-caution)]" />
        <div className="text-[14px] font-medium">
          {this.props.label ?? 'This panel stopped responding'}
        </div>
        <p className="max-w-sm text-[12.5px] leading-relaxed text-[var(--color-fg-3)]">
          {error.message || 'An unexpected error occurred while rendering.'}
        </p>
        <div className="flex gap-1.5">
          <Button onClick={this.reset}>
            <Icon.Refresh /> Try again
          </Button>
          <Button variant="quiet" onClick={() => window.location.reload()}>
            Reload the app
          </Button>
        </div>
      </div>
    )
  }
}
