"use client";
import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Cpu,
  FileText,
  Mic,
  Radio,
  Link2,
  Play,
  Pause,
  Square,
  Bookmark,
  Highlighter,
  ListChecks,
  Clock,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { DocMeta, SimpleSession, TranscriptSegment } from "@/app/types";

/* ------------------------- Inlined ASR mini-client ------------------------- */

type AsrEvent = {
  session_id: string;
  chunk_index: number;
  start_sec: number;
  end_sec: number;
  text: string;
  words: { start: number | null; end: number | null; word: string }[];
  final: boolean;
};

const ASR_BASE =
  (process.env.NEXT_PUBLIC_ASR_BASE as string | undefined) ?? "http://127.0.0.1:5001";

async function asrHealth(): Promise<boolean> {
  try {
    const r = await fetch(`${ASR_BASE}/health`, { cache: "no-store" });
    return r.ok;
  } catch {
    return false;
  }
}

async function asrStart(): Promise<string> {
  const r = await fetch(`${ASR_BASE}/session/start`, { method: "POST" });
  if (!r.ok) throw new Error(`ASR start failed: ${r.status}`);
  const j = await r.json();
  return j.session_id as string;
}

async function asrStop(id: string): Promise<void> {
  try {
    await fetch(`${ASR_BASE}/session/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: id }),
    });
  } catch {
    // ignore
  }
}

function asrOpenStream(
  id: string,
  onEvent: (ev: AsrEvent) => void,
  onClose?: () => void,
  onError?: (e: any) => void
): EventSource {
  const es = new EventSource(`${ASR_BASE}/session/${id}/stream`);
  // expose for quick debugging in DevTools
  (window as any).__dbg_es = es;

  es.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data) as AsrEvent;
      onEvent(data);
      if (data.final) {
        es.close();
        onClose?.();
      }
    } catch (e) {
      onError?.(e);
    }
  };
  es.onerror = (e) => {
    onError?.(e);
    // keep open; server signals final; we close there
  };
  return es;
}

/* ------------------------------ UI subcomponents ------------------------------ */

function Pane({ title, items }: { title: string; items: TranscriptSegment[] }) {
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [items]);

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-lg font-semibold">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div ref={listRef} className="h[58vh] md:h-[58vh] overflow-y-auto pr-2 space-y-3">
          {items.map((seg) => (
            <motion.div
              key={seg.id}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              className="group"
            >
              <div className="flex items-start gap-3">
                <Badge variant="secondary" className="shrink-0">
                  {seg.time}
                </Badge>
                <div className="flex-1">
                  <div className="text-sm text-muted-foreground">{seg.speaker}</div>
                  <p
                    className={`leading-relaxed ${
                      seg.q ? "border-l-2 pl-3 border-amber-400" : ""
                    }`}
                  >
                    {seg.text}
                  </p>
                </div>
                <div className="opacity-0 group-hover:opacity-100 transition">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="iconLg"
                          className="rounded-full border border-amber-200 text-amber-600 hover:bg-amber-50 hover:text-amber-700"
                        >
                          <Bookmark className="h-9 w-9" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Bookmark</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

type DocViewerProps = {
  doc: DocMeta;
  onClose: () => void;
  fileExtension: (filename: string) => string;
  formatFileSize: (bytes: number) => string;
  formatIsoDate: (iso?: string) => string;
};

function DocViewer({
  doc,
  onClose,
  fileExtension,
  formatFileSize,
  formatIsoDate,
}: DocViewerProps) {
  const [page, setPage] = useState<number>(1);

  return (
    <Card className="mb-3">
      <CardHeader className="py-3 flex flex-row items-center justify-between">
        <CardTitle className="text-base">{doc.filename}</CardTitle>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            {"<"}
          </Button>
          <span className="text-sm">Page {page}</span>
          <Button size="sm" variant="outline" onClick={() => setPage((p) => p + 1)}>
            {">"}
          </Button>
          <Button size="sm" variant="ghost" onClick={onClose}>
            Close
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-xs text-muted-foreground">
          {fileExtension(doc.filename)} | {formatFileSize(doc.size)} | Uploaded{" "}
          {formatIsoDate(doc.uploadedAt)}
        </div>
        {doc.notes && <div className="text-sm text-muted-foreground">{doc.notes}</div>}
        <div className="h-[35vh] overflow-x-auto border rounded grid place-items-center text-sm text-muted-foreground">
          (PDF preview mock)
        </div>
      </CardContent>
    </Card>
  );
}

/* ---------------------------------- Main ---------------------------------- */

type SessionViewProps = {
  session: SimpleSession | null;
  docs: DocMeta[];
  openDocId: string | null;
  onDocOpenChange: (id: string | null) => void;
  live: boolean;
  paused: boolean;
  latency: number;
  onStart: () => void;
  onPauseResume: () => void;
  onStop: () => void;
  onLinkDocs: () => void;
  formatDuration: (ms: number) => string;
  formatFileSize: (bytes: number) => string;
  formatIsoDate: (iso?: string) => string;
  fileExtension: (filename: string) => string;
};

export default function SessionView({
  session,
  docs,
  openDocId,
  onDocOpenChange,
  live,
  paused,
  latency,
  onStart,
  onPauseResume,
  onStop,
  onLinkDocs,
  formatDuration,
  formatFileSize,
  formatIsoDate,
  fileExtension,
}: SessionViewProps) {
  const activeDoc = openDocId ? docs.find((doc) => doc.id === openDocId) : null;

  // ASR state
  const [asrId, setAsrId] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const [liveTranscript, setLiveTranscript] = useState<TranscriptSegment[]>([]);
  const segCounterRef = useRef(0);

  const secToClock = (sec: number) => {
    const s = Math.max(0, Math.floor(sec));
    const hh = String(Math.floor(s / 3600)).padStart(2, "0");
    const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  };

  const subscribeAsr = (id: string) => {
    setLiveTranscript([]);
    segCounterRef.current = 0;
    esRef.current?.close();
    const es = asrOpenStream(
      id,
      (ev: AsrEvent) => {
        // DO NOT DROP EMPTY TEXT — show a visible heartbeat so the UI advances.
        const visible = ev.text && ev.text.trim().length > 0 ? ev.text : "…";
        setLiveTranscript((prev) => [
          ...prev,
          {
            id: ++segCounterRef.current,
            time: secToClock(Math.round(ev.start_sec)),
            text: visible,
            speaker: "ASR",
            q: /\?$/.test((ev.text || "").trim()),
          },
        ]);
      },
      () => {
        esRef.current?.close();
        esRef.current = null;
      },
      (e) => {
        console.error("ASR stream error", e);
      }
    );
    esRef.current = es;
  };

  const handleStart = async () => {
    // your app’s start flow
    onStart();

    // then start ASR + subscribe
    try {
      const ok = await asrHealth();
      if (!ok) console.warn("ASR /health not OK, attempting start anyway…");
      const id = await asrStart();
      setAsrId(id);
      subscribeAsr(id);
    } catch (e) {
      console.error("Failed to start ASR:", e);
    }
  };

  const handleStop = async () => {
    try {
      if (asrId) await asrStop(asrId);
    } catch {
      // ignore
    }
    esRef.current?.close();
    esRef.current = null;
    setAsrId(null);

    // your app’s stop flow
    onStop();
  };

  // cleanup on unmount
  useEffect(() => {
    return () => {
      esRef.current?.close();
    };
  }, []);

  // simple placeholder translation (until MT is wired on the client)
  const sampleTranslation: TranscriptSegment[] = liveTranscript.length
    ? liveTranscript.map((s) => ({
        ...s,
        speaker: "MT",
        // If source looked like a placeholder, mirror it; otherwise mark as "→ EN" mock
        text: s.text === "…" ? "…" : s.text + "  (→ EN mock)",
      }))
    : [
        {
          id: 1,
          time: "00:00:00",
          text: "Waiting for audio…",
          speaker: "MT",
          q: false,
        },
      ];

  return (
    <>
      <div className="border-b">
        <div className="max-w-7xl mx-auto grid grid-cols-12 items-center gap-3 px-4 py-3">
          <div className="col-span-12 md:col-span-6 flex items-center gap-2 flex-wrap">
            <Badge variant="secondary" className="gap-2">
              <Cpu className="h-3 w-3" /> Local {session ? `• ${session.title}` : ""}
            </Badge>
            {session?.docIds.map((id) => {
              const doc = docs.find((item) => item.id === id);
              if (!doc) return null;
              return (
                <Button
                  key={id}
                  size="sm"
                  variant="outline"
                  onClick={() => onDocOpenChange(id)}
                >
                  <FileText className="h-3 w-3 mr-1" /> {doc.filename}
                </Button>
              );
            })}
          </div>
          <div className="col-span-12 md:col-span-3 flex items-center gap-2 mt-2 md:mt-0">
            <Select defaultValue="mic">
              <SelectTrigger className="w-full md:w-40">
                <SelectValue placeholder="Source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="mic">
                  <div className="flex items-center gap-2">
                    <Mic className="h-4 w-4" /> Microphone
                  </div>
                </SelectItem>
                <SelectItem value="loopback">
                  <div className="flex items-center gap-2">
                    <Radio className="h-4 w-4" /> System Audio
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="col-span-12 md:col-span-3 flex flex-wrap items-center justify-start md:justify-end gap-2 md:gap-3 mt-2 md:mt-0">
            <Button
              variant="outline"
              size="iconLg"
              onClick={onLinkDocs}
              aria-label="Link documents"
              className="rounded-full border border-blue-200 text-blue-600 hover:bg-blue-50 hover:text-blue-700"
            >
              <Link2 className="h-5 w-5" />
            </Button>
            {!session && (
              <Button onClick={handleStart}>
                <Play className="mr-2 h-4 w-4" /> Start
              </Button>
            )}
            {session && (
              <>
                <Button variant="outline" onClick={onPauseResume}>
                  {live && !paused ? (
                    <>
                      <Pause className="mr-2 h-4 w-4" /> Pause
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" /> Resume
                    </>
                  )}
                </Button>
                <Button variant="destructive" onClick={handleStop}>
                  <Square className="mr-2 h-4 w-4" /> Stop
                </Button>
              </>
            )}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="iconLg"
                    aria-label="Bookmark moment"
                    className="rounded-full border border-amber-200 text-amber-600 hover:bg-amber-50 hover:text-amber-700"
                  >
                    <Bookmark className="h-9 w-9" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Bookmark (B)</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      </div>

      {activeDoc && (
        <div className="max-w-7xl mx-auto px-4 mt-3">
          <DocViewer
            doc={activeDoc}
            onClose={() => onDocOpenChange(null)}
            fileExtension={fileExtension}
            formatFileSize={formatFileSize}
            formatIsoDate={formatIsoDate}
          />
        </div>
      )}

      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-4 px-4 py-4">
        <div>
          <Pane
            title={`Transcript (source)${session ? ` - ${session.title}` : ""}`}
            items={
              liveTranscript.length > 0
                ? liveTranscript
                : [
                    {
                      id: 0,
                      time: "00:00:00",
                      text: "Waiting for audio…",
                      speaker: "ASR",
                      q: false,
                    },
                  ]
            }
          />
        </div>
        <div>
          <Pane title="Translation (auto)" items={sampleTranslation} />
        </div>
      </div>

      <div className="border-t">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-3">
          <Clock className="h-4 w-4" />
          <div className="w-full">
            <Progress value={live ? Math.min(100, latency / 12) : 0} />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>{session ? formatDuration(session.accumMs ?? 0) : "00:00:00"}</span>
              <span>{session ? new Date(session.createdAt).toLocaleTimeString() : ""}</span>
            </div>
          </div>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="iconLg"
                  className="rounded-full border border-sky-200 text-sky-600 hover:bg-sky-50 hover:text-sky-700"
                >
                  <Highlighter className="h-9 w-9" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Auto-highlight questions</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="iconLg"
                  className="rounded-full border border-emerald-200 text-emerald-600 hover:bg-emerald-50 hover:text-emerald-700"
                >
                  <ListChecks className="h-9 w-9" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Enforce glossary</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
    </>
  );
}
