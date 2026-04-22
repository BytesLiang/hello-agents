export function StatusPill({ status }) {
  const label = status;
  return (
    <span className={`status-pill status-${status}`} role="status">
      {label}
    </span>
  );
}

export function MetricChip({ children }) {
  return <span className="metric-chip">{children}</span>;
}

export function LoadingIndicator({ text = "正在加载..." }) {
  return (
    <div className="loading-indicator" role="status" aria-live="polite">
      <span className="loading-spinner" aria-hidden="true" />
      <span>{text}</span>
    </div>
  );
}

export function EmptyState({ children }) {
  return (
    <div className="empty-state" role="status">
      {children}
    </div>
  );
}

export function ErrorMessage({ children }) {
  if (!children) return null;
  return (
    <p className="message error-message" role="alert">
      {children}
    </p>
  );
}

export function SuccessMessage({ children }) {
  if (!children) return null;
  return (
    <p className="message success-message" role="status" aria-live="polite">
      {children}
    </p>
  );
}
