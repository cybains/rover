// src/hooks/useSessionRealtime.ts
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const BACKEND_BASE = process.env.NEXT_PUBLIC_BACKEND_BASE || "http://localhost:4000";

type WsPartialMsg = {
  kind: "partial";
  paraId?: string;
  lang?: string;
  textSrcPartial?: string;
  textEnPartial?: string;
  speaker?: string;
};

type WsFinalMsg = {
  _id: string;
  sessionId: string;
  idxStart: number;
  idxEnd: number;
  tStart: number;
  tEnd: number;
  lang?: string;
  speaker?: string;
  textSrc: string;
  textEn: string;
  isQuestion?: boolean;
  bookmark?: boolean;
  glossaryHits?: any[];
};

type WsHeartbeat = { type: "heartbeat"; ts: string };

type UiSeg = {
  id: number;
  time: string;
  text: string;
  speaker: string;
  q: boolean;
  _tStart: number;
  _tEnd: number;
  _idx: number;
};

const secToClock = (sec: number) => {
  const s = Math.max(0, Math.floor(sec));
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
};

export function useSessionRealtime() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [live, setLive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [latency, setLatency] = useState(0);

  const [leftSegs, setLeftSegs] = useState<UiSeg[]>([]);
  const [rightSegs, setRightSegs] = useState<UiSeg[]>([]);
  const [sourceChoice, setSourceChoice] = useState<"mic" | "system">("mic");

  const wsRef = useRef<WebSocket | null>(null);
  const nextIdRef = useRef(0);
  const lastBeatRef = useRef<number>(Date.now());
  const lastLangRef = useRef<string>("");

  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => {
      setLatency(Math.min(12000, Date.now() - lastBeatRef.current));
    }, 300);
    return () => clearInterval(t);
  }, [live]);

  const pushUi = useCallback(
    (side: "left" | "right", seg: UiSeg) => {
      if (paused) return;
      if (side === "left") setLeftSegs((p) => [...p, seg]);
      else setRightSegs((p) => [...p, seg]);
    },
    [paused]
  );

  const handlePartial = useCallback(
    (msg: WsPartialMsg) => {
      const lang = (msg.lang || "").toLowerCase();
      if (lang) lastLangRef.current = lang;

      const textSrc = (msg.textSrcPartial || "").trim();
      const textEn = (msg.textEnPartial || "").trim();
      if (!textSrc && !textEn) return;

      const id = ++nextIdRef.current;
      const base: UiSeg = {
        id,
        time: secToClock(0),
        text: "",
        speaker: "Speaker",
        q: false,
        _tStart: 0,
        _tEnd: 0,
        _idx: -1,
      };

      if (textSrc) pushUi("left", { ...base, text: textSrc });
      if (textEn) pushUi("right", { ...base, text: textEn, speaker: "EN (auto)" });
    },
    [pushUi]
  );

  const handleFinal = useCallback(
    (msg: WsFinalMsg) => {
      const lang = (msg.lang || lastLangRef.current || "").toLowerCase();
      if (lang) lastLangRef.current = lang;

      const leftText = (msg.textSrc || "").trim();
      const rightText = (msg.textEn || "").trim();
      const tStart = msg.tStart ?? 0;

      const left: UiSeg = {
        id: ++nextIdRef.current,
        time: secToClock(Math.round(tStart)),
        text: leftText,
        speaker: "Speaker",
        q: !!msg.isQuestion || /\?$/.test(leftText),
        _tStart: msg.tStart ?? 0,
        _tEnd: msg.tEnd ?? (msg.tStart ?? 0),
        _idx: msg.idxStart ?? 0,
      };
      const right: UiSeg = {
        id: ++nextIdRef.current,
        time: secToClock(Math.round(tStart)),
        text: rightText,
        speaker: lang === "en" ? "DE (auto)" : "EN (auto)",
        q: false,
        _tStart: msg.tStart ?? 0,
        _tEnd: msg.tEnd ?? (msg.tStart ?? 0),
        _idx: msg.idxStart ?? 0,
      };

      if (leftText) pushUi("left", left);
      if (rightText) pushUi("right", right);
    },
    [pushUi]
  );

  const connectWs = useCallback(
    (sid: string) => {
      const url = new URL("/realtime", BACKEND_BASE);
      url.protocol = url.protocol.replace("http", "ws");
      url.searchParams.set("sessionId", sid);

      wsRef.current?.close();
      const ws = new WebSocket(url.toString());
      wsRef.current = ws;

      ws.onopen = () => {
        lastBeatRef.current = Date.now();
      };
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data) as WsPartialMsg | WsFinalMsg | WsHeartbeat;
          if ((data as WsHeartbeat).type === "heartbeat") {
            lastBeatRef.current = Date.now();
            return;
          }
          if ((data as any).kind === "partial") {
            handlePartial(data as WsPartialMsg);
          } else if ((data as any)._id && (data as any).tStart !== undefined) {
            handleFinal(data as WsFinalMsg);
          }
        } catch {
          /* ignore */
        }
      };
    },
    [handleFinal, handlePartial]
  );

  const start = useCallback(
    async (opts?: { title?: string; docIds?: string[]; source?: "mic" | "system" }) => {
      setLeftSegs([]);
      setRightSegs([]);
      nextIdRef.current = 0;
      setPaused(false);
      setLive(true);

      const source = opts?.source ?? sourceChoice;
      setSourceChoice(source);

      const r = await fetch(`${BACKEND_BASE}/sessions/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: opts?.title || "Untitled Session",
          docIds: opts?.docIds || [],
        }),
      });
      const j = await r.json();
      const sid = j._id as string;
      setSessionId(sid);

      connectWs(sid);

      await fetch(`${BACKEND_BASE}/capture/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: sid, source }),
      });
    },
    [connectWs, sourceChoice]
  );

  const pauseResume = useCallback(() => {
    setPaused((p) => !p);
  }, []);

  const stop = useCallback(async () => {
    setLive(false);
    const sid = sessionId;
    wsRef.current?.close();
    wsRef.current = null;

    if (sid) {
      try {
        await fetch(`${BACKEND_BASE}/capture/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessionId: sid }),
        });
      } catch {}
      try {
        await fetch(`${BACKEND_BASE}/sessions/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessionId: sid }),
        });
      } catch {}
    }
    setSessionId(null);
  }, [sessionId]);

  const addBookmark = useCallback(async (_atSec: number) => {
    /* placeholder if you wire bookmark-by-time later */
  }, []);

  const left = useMemo(() => leftSegs, [leftSegs]);
  const right = useMemo(() => rightSegs, [rightSegs]);

  return {
    sessionId,
    live,
    paused,
    latency,
    left,
    right,
    start,
    pauseResume,
    stop,
    sourceChoice,
    setSourceChoice,
    addBookmark,
  };
}
