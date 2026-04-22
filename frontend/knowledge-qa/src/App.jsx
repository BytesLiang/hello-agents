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
    <main className="desktop-shell" role="alert" aria-live="assertive">
      <section className="desktop-panel">
        <p className="eyebrow">桌面运行时</p>
        <h1>正在准备本地知识库问答工作区。</h1>
        <p className="hero-copy">
          桌面端正在检查或启动本地 Python API，工作区即将打开。
        </p>
        {error ? (
          <p className="message error-message">{error}</p>
        ) : (
          <p className="message muted-message">
            正在启动本地 API 服务 <code>127.0.0.1:8000</code>。
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
              : "桌面端后端启动失败。"
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
      <a href="#main-content" className="skip-link">
        跳转到主要内容
      </a>
      <AppRoutes />
    </BrowserRouter>
  );
}
