import { useEffect, useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";

import {
  probeDesktopBackend,
  startDesktopBackend,
  stopDesktopBackend,
  waitForDesktopBackend,
} from "./desktopBackend";
import { isTauriRuntime } from "./runtime";
import { KnowledgeBaseDetailPage } from "./routes/KnowledgeBaseDetailPage";
import { KnowledgeBaseListPage } from "./routes/KnowledgeBaseListPage";

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<KnowledgeBaseListPage />} />
      <Route
        path="/knowledge-bases/:kbId"
        element={<KnowledgeBaseDetailPage />}
      />
    </Routes>
  );
}

function DesktopBootstrapScreen({ error }) {
  return (
    <main className="desktop-shell">
      <section className="desktop-panel">
        <p className="eyebrow">Desktop Runtime</p>
        <h1>Preparing the local Knowledge QA workspace.</h1>
        <p className="hero-copy">
          The desktop shell is checking or starting the local Python API before
          the workspace opens.
        </p>
        {error ? (
          <p className="message error-message">{error}</p>
        ) : (
          <p className="message muted-message">
            Starting the local API on <code>127.0.0.1:8000</code>.
          </p>
        )}
      </section>
    </main>
  );
}

export default function App() {
  const [ready, setReady] = useState(() => !isTauriRuntime());
  const [error, setError] = useState("");

  useEffect(() => {
    if (!isTauriRuntime()) {
      return undefined;
    }

    let cancelled = false;
    let shouldStop = false;

    async function bootstrapDesktopBackend() {
      try {
        const probe = await probeDesktopBackend();
        if (!probe.reachable) {
          const status = await startDesktopBackend();
          shouldStop = Boolean(status?.started_by_app);
        }
        await waitForDesktopBackend();
        if (!cancelled) {
          setReady(true);
        }
      } catch (startupError) {
        if (shouldStop) {
          void stopDesktopBackend();
          shouldStop = false;
        }
        if (!cancelled) {
          setError(
            startupError instanceof Error
              ? startupError.message
              : "Failed to prepare the desktop backend."
          );
        }
      }
    }

    void bootstrapDesktopBackend();

    return () => {
      cancelled = true;
      if (shouldStop) {
        void stopDesktopBackend();
      }
    };
  }, []);

  if (!ready) {
    return <DesktopBootstrapScreen error={error} />;
  }

  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
