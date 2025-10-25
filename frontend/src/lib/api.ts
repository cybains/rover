const API = "http://localhost:4000";

export async function apiGet(path: string) {
  const res = await fetch(API + path);
  return res.json();
}

export async function apiPost(path: string, body: any) {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

export function openRealtime(sessionId: string) {
  return new WebSocket(`ws://localhost:4000/realtime?sessionId=${sessionId}`);
}
