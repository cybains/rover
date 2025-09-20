const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:4000";

const ASR_BASE_URL =
  process.env.NEXT_PUBLIC_ASR_BASE?.replace(/\/$/, "") || "http://127.0.0.1:5001";

const WS_BASE = (() => {
  try {
    const url = new URL(BASE_URL);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return `${url.protocol}//${url.host}`;
  } catch (err) {
    console.warn("Unable to derive WS base from", BASE_URL, err);
    return BASE_URL.startsWith("https://") ? "wss://localhost:4000" : "ws://localhost:4000";
  }
})();

export async function apiGet(path: string) {
  const res = await fetch(`${BASE_URL}${path}`, { mode: "cors", cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return res.json();
}

export async function apiPost(path: string, data?: any) {
  const body = data !== undefined ? JSON.stringify(data) : undefined;
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    mode: "cors",
    body,
  });
  if (!res.ok) throw new Error(`POST ${path} -> ${res.status}`);
  return res.json();
}

export function connectWs(sessionId: string) {
  return new WebSocket(`${WS_BASE}/realtime?sessionId=${sessionId}`);
}

export async function listSessions() {
  return apiGet(`/sessions`);
}

export async function getSession(sessionId: string) {
  return apiGet(`/sessions/${sessionId}`);
}

export async function startSession(options?: { title?: string | null; docIds?: string[] }) {
  const payload: Record<string, unknown> = {};
  if (options?.title !== undefined) {
    payload.title = options.title ?? null;
  } else {
    payload.title = null;
  }
  if (options?.docIds?.length) {
    payload.docIds = options.docIds;
  } else {
    payload.docIds = [];
  }
  return apiPost(`/sessions/start`, payload);
}

export async function stopSession(sessionId: string) {
  return apiPost(`/sessions/stop`, { sessionId });
}

export async function deleteSession(sessionId: string) {
  const res = await fetch(`${BASE_URL}/sessions/${sessionId}`, {
    method: "DELETE",
    mode: "cors",
  });
  if (!res.ok) throw new Error(`DELETE /sessions/${sessionId} -> ${res.status}`);
  return res.json();
}

export async function startCapture(
  sessionId: string,
  source: "auto" | "mic" | "loopback" = "auto"
) {
  return apiPost(`/capture/start`, { sessionId, source });
}

export async function stopCapture(sessionId: string) {
  return apiPost(`/capture/stop`, { sessionId });
}

export async function linkSessionDocument(sessionId: string, documentId: string) {
  return apiPost(`/sessions/${sessionId}/documents/link`, { documentId });
}

export async function unlinkSessionDocument(sessionId: string, documentId: string) {
  return apiPost(`/sessions/${sessionId}/documents/unlink`, { documentId });
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

export async function listDocuments() {
  return apiGet(`/documents`);
}

export async function uploadDocument(options: {
  file: File;
  sessionIds?: string[];
  tags?: string[];
  notes?: string;
}) {
  const form = new FormData();
  form.append("file", options.file);
  if (options.sessionIds?.length) {
    form.append("sessionIds", JSON.stringify(options.sessionIds));
  }
  if (options.tags?.length) {
    form.append("tags", JSON.stringify(options.tags));
  }
  if (options.notes) {
    form.append("notes", options.notes);
  }
  const res = await fetch(`${BASE_URL}/documents/upload`, {
    method: "POST",
    mode: "cors",
    body: form,
  });
  if (!res.ok) throw new Error(`POST /documents/upload -> ${res.status}`);
  return res.json();
}

export async function deleteDocument(documentId: string) {
  const res = await fetch(`${BASE_URL}/documents/${documentId}`, {
    method: "DELETE",
    mode: "cors",
  });
  if (!res.ok) throw new Error(`DELETE /documents/${documentId} -> ${res.status}`);
  return res.json();
}
export async function getAsrHealth() {
  const res = await fetch(`${ASR_BASE_URL}/health`, { mode: 'cors', cache: 'no-store' });
  if (!res.ok) throw new Error(`GET /health -> ${res.status}`);
  return res.json();
}

export async function startAsrSession() {
  const res = await fetch(`${ASR_BASE_URL}/session/start`, {
    method: 'POST',
    mode: 'cors',
  });
  if (!res.ok) throw new Error(`POST /session/start -> ${res.status}`);
  return res.json();
}

export async function stopAsrSession(sessionId: string) {
  const res = await fetch(`${ASR_BASE_URL}/session/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    mode: 'cors',
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(`POST /session/stop -> ${res.status}`);
  return res.json();
}

export function streamAsrSession(sessionId: string) {
  return new EventSource(`${ASR_BASE_URL}/session/${sessionId}/stream`);
}
