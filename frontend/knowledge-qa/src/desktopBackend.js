import { invoke } from "@tauri-apps/api/core";

import { isTauriRuntime, resolveApiBaseUrl } from "./runtime";

const STARTUP_TIMEOUT_MS = 30_000;
const POLL_INTERVAL_MS = 400;

async function fetchHealth() {
  const response = await fetch(`${resolveApiBaseUrl()}/api/health`);
  if (!response.ok) {
    throw new Error(`Backend health check failed with status ${response.status}.`);
  }
  return response.json();
}

export async function probeDesktopBackend() {
  if (!isTauriRuntime()) {
    return { reachable: true, managed: false };
  }

  try {
    await fetchHealth();
    return { reachable: true, managed: false };
  } catch {
    return { reachable: false, managed: false };
  }
}

export async function startDesktopBackend() {
  if (!isTauriRuntime()) {
    return { reachable: true, managed: false };
  }

  return invoke("start_backend");
}

export async function stopDesktopBackend() {
  if (!isTauriRuntime()) {
    return;
  }

  await invoke("stop_backend");
}

export async function waitForDesktopBackend() {
  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      await fetchHealth();
      return;
    } catch {
      await new Promise((resolve) => {
        window.setTimeout(resolve, POLL_INTERVAL_MS);
      });
    }
  }
  throw new Error(
    "本地 Python API 未能在规定时间内就绪。请检查 .hello_agents/logs/knowledge_qa_desktop_backend.log 日志，并确认 Python 环境、Qdrant 服务和 .env 配置是否正确。"
  );
}
