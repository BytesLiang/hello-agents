export const STATUS_LABELS = {
  indexing: "索引中",
  ready: "就绪",
  failed: "失败",
};

export function formatDate(value) {
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function formatStatus(status) {
  return STATUS_LABELS[status] || status;
}
