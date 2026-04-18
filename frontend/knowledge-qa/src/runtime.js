export function isTauriRuntime() {
  if (typeof window === "undefined") {
    return false;
  }

  return "__TAURI_INTERNALS__" in window || "__TAURI__" in window;
}

export function resolveApiBaseUrl() {
  const configuredBaseUrl = import.meta.env.VITE_KNOWLEDGE_QA_API_BASE_URL?.trim();
  if (configuredBaseUrl) {
    return configuredBaseUrl.replace(/\/$/, "");
  }

  if (isTauriRuntime()) {
    return "http://127.0.0.1:8000";
  }

  return "";
}

export async function pickDesktopDocuments() {
  const pluginName = "@tauri-apps/plugin-dialog";
  const { open } = await import(/* @vite-ignore */ pluginName);
  const selected = await open({
    multiple: true,
    directory: false,
    filters: [
      {
        name: "Documents",
        extensions: [
          "md",
          "markdown",
          "txt",
          "rst",
          "json",
          "yaml",
          "yml",
          "html",
          "htm",
          "pdf",
        ],
      },
    ],
  });

  if (selected === null) {
    return [];
  }
  if (Array.isArray(selected)) {
    return selected.filter((item) => typeof item === "string");
  }
  return typeof selected === "string" ? [selected] : [];
}
