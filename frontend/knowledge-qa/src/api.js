function buildErrorMessage(payload, fallback) {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const { detail } = payload;
    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }
  }
  return fallback;
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    throw new Error(
      buildErrorMessage(payload, `Request failed with status ${response.status}.`)
    );
  }

  return payload;
}

export function listKnowledgeBases() {
  return requestJson("/api/knowledge-bases");
}

export function getKnowledgeBase(kbId) {
  return requestJson(`/api/knowledge-bases/${kbId}`);
}

export function createKnowledgeBase(payload) {
  return requestJson("/api/knowledge-bases", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function uploadKnowledgeBase({ name, description, files }) {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("description", description);
  for (const file of files) {
    formData.append("files", file);
  }

  const response = await fetch("/api/knowledge-bases/upload", {
    method: "POST",
    body: formData,
  });

  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    throw new Error(
      buildErrorMessage(payload, `Request failed with status ${response.status}.`)
    );
  }

  return payload;
}

export function askKnowledgeBase(kbId, question) {
  return requestJson(`/api/knowledge-bases/${kbId}/ask`, {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export function listTraces(limit = 10) {
  return requestJson(`/api/traces?limit=${limit}`);
}
