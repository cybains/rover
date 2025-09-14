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

export async function startCapture(
  sessionId: string,
  source: "auto" | "mic" | "loopback" = "auto"
) {
  return apiPost("/capture/start", { sessionId, source });
}

export async function stopCapture(sessionId: string) {
  return apiPost("/capture/stop", { sessionId });
}

export async function exportSession(id: string) {
  return apiPost(`/export/${id}`);
}

export async function getGlossary() {
  return apiGet(`/glossary`);
}

export async function recomputeQA(id: string) {
  return apiPost(`/sessions/${id}/qa/recompute`);
}

export async function getHighlights(id: string) {
  return apiGet(`/sessions/${id}/highlights`);
}

export async function bookmarkSegment(id: string) {
  return apiPost(`/segments/${id}/bookmark`);
}

export async function unbookmarkSegment(id: string) {
  const res = await fetch(`${BASE_URL}/segments/${id}/bookmark`, {
    method: "DELETE",
    mode: "cors",
  });
  if (!res.ok) throw new Error(`DELETE /segments/${id}/bookmark -> ${res.status}`);
  return res.json();
}

export async function renameSpeaker(
  sessionId: string,
  fromName: string,
  toName: string
) {
  return apiPost(`/sessions/${sessionId}/speakers/rename`, {
    from: fromName,
    to: toName,
  });
}

export async function generate(
  kind: string,
  sessionId: string,
  paraIds?: string[],
  options?: any
) {
  return apiPost(`/generate/${kind}`, { sessionId, paraIds, options });
}

export async function getGenerations(sessionId: string, type: string) {
  const q = type ? `?type=${type}` : "";
  return apiGet(`/sessions/${sessionId}/generations${q}`);
}

export async function getSettings() {
  return apiGet(`/settings`);
}

export async function updateSettings(data: any) {
  return apiPost(`/settings`, data);
}

export async function getStorageStatus() {
  return apiGet(`/storage/status`);
}

export async function purgeSessions(before: string) {
  return apiPost(`/sessions/purge`, { before });
}

export async function getSessionStatus(id: string) {
  return apiGet(`/status/${id}`);
}
