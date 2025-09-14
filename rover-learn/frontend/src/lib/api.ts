const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:4000";

export async function apiGet(path: string) {
  const res = await fetch(`${BASE_URL}${path}`, { mode: "cors", cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return res.json();
}

export async function apiPost(path: string, data?: any) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    mode: "cors",
    body: data !== undefined ? JSON.stringify(data) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path} -> ${res.status}`);
  return res.json();
}

export function connectWs(sessionId: string) {
  const wsBase = (BASE_URL.startsWith("https://") ? "wss://" : "ws://") + "localhost:4000";
  return new WebSocket(`${wsBase}/realtime?sessionId=${sessionId}`);
}

export async function startSession(title?: string) {
  return apiPost("/sessions/start", { title: title || null });
}

export async function exportSession(id: string) {
  return apiPost(`/export/${id}`);
}

export async function getGlossary() {
  return apiGet(`/glossary`);
}
