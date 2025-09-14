const BASE_URL = 'http://localhost:4000';

export async function apiGet(path: string) {
  const res = await fetch(`${BASE_URL}${path}`);
  return res.json();
}

export async function apiPost(path: string, data: any) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return res.json();
}

export function connectWs(sessionId: string) {
  return new WebSocket(`ws://localhost:4000/realtime?sessionId=${sessionId}`);
}
