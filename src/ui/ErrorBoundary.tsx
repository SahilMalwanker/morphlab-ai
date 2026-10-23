import { Component, type ErrorInfo, type ReactNode } from 'react';

/** Shows a failure in one workspace as a message instead of a blank page. */
export class ErrorBoundary extends Component<{ label: string; children: ReactNode }, { error: Error | null }> {
  override state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error): { error: Error } {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.warn(`${this.props.label} failed`, error, info.componentStack);
  }

  override render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="viewport" role="alert" style={{ padding: 24 }}>
          <p className="note error">{this.props.label} stopped because of an error: {this.state.error.message}</p>
          <p className="note muted">Other views keep working; choose one in the navigation. Reload the page to try again.</p>
        </div>
      );
    }
    return this.props.children;
  }
}
