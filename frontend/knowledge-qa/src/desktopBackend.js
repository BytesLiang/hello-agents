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
    "The local Python API did not become ready in time. Check .hello_agents/logs/knowledge_qa_desktop_backend.log, then verify your Python environment, Qdrant, and .env configuration."
  );
}
