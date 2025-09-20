// Formatting helpers shared across views

export const formatFileSize = (bytes: number) => {
  if (!bytes || bytes < 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  if (idx === 0) return `${bytes} ${units[idx]}`;
  const value = bytes / Math.pow(1024, idx);
  return `${value.toFixed(1)} ${units[idx]}`;
};

export const formatIsoDate = (iso?: string) => {
  if (!iso) return "unknown date";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString();
};

export const fileExtension = (filename: string) => {
  const parts = filename.split(".");
  if (parts.length < 2) return "FILE";
  return parts.pop()?.toUpperCase() || "FILE";
};

export const formatDuration = (ms: number) => {
  const s = Math.floor(ms / 1000);
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
};
