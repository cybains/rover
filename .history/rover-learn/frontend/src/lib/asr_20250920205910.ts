// Minimal client for your FastAPI ASR (SSE)
export type AsrEvent = {
  session_id: string;
  chunk_index: number;
  start_sec: number;
  end_sec: number;
  text: string;
  words: { start: number | null; end: number | null; word: string }[];
  final: boolean;
};

const BASE = process.env.NEXT_PUBLIC_ASR_BASE ?? "http://127.0.0.1:5001";

export async function asrHealth(): Promise<boolean> {
  try { const r = await fetch(`${BASE}/health`, { cache: "no-store" }); return r.ok; }
  catch { return false; }
}

export async function asrStart(): Promise<string> {
  const r = await fetch(`${BASE}/session/start`, { method: "POST" });
  if (!r.ok) throw new Error(`ASR start failed: ${r.status}`);
  const j = await r.json();
  return j.session_id as string;
}

export async function asrStop(id: string): Promise<void> {
  await fetch(`${BASE}/session/stop`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: id }),
  }).catch(() => {});
}

export function asrOpenStream(
  id: string,
  onEvent: (ev: AsrEvent) => void,
  onClose?: () => void,
  onError?: (e: any) => void
): EventSource {
  const es = new EventSource(`${BASE}/session/${id}/stream`);
  es.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data) as AsrEvent;
      onEvent(data);
      if (data.final) { es.close(); onClose?.(); }
    } catch (e) { onError?.(e); }
  };
  es.onerror = (e) => { onError?.(e); /* keep open; server controls close */ };
  return es;
}
