"use client";
import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Cpu, FileText, Mic, Radio, Link2, Play, Pause, Square, Bookmark,
  Highlighter, ListChecks, Clock,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Tooltip, TooltipContent, TooltipProvider, TooltipTrigger,
} from "@/components/ui/tooltip";
import type { DocMeta, SimpleSession, TranscriptSegment } from "@/app/types";
import { useSessionRealtime } from "@/hooks/useSessionRealtime";

/* ------------------------------ UI subcomponents ------------------------------ */

function Pane({ title, items }: { title: string; items: TranscriptSegment[] }) {
  const listRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [items]);
  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-lg font-semibold">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div ref={listRef} className="h[58vh] md:h-[58vh] overflow-y-auto pr-2 space-y-3">
          {items.map((seg) => (
            <motion.div key={seg.id} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className="group">
              <div className="flex items-start gap-3">
                <Badge variant="secondary" className="shrink-0">{seg.time}</Badge>
                <div className="flex-1">
                  <div className="text-sm text-muted-foreground">{seg.speaker}</div>
                  <p className={`leading-relaxed ${seg.q ? "border-l-2 pl-3 border-amber-400" : ""}`}>{seg.text}</p>
                </div>
                <div className="opacity-0 group-hover:opacity-100 transition">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button variant="ghost" size="iconLg" className="rounded-full border border-amber-200 text-amber-600 hover:bg-amber-50 hover:text-amber-700">
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

/* ---------------------------------- Main ---------------------------------- */

type SessionViewProps = {
  // keep props for compatibility (session/docs if you load them elsewhere)
  session?: SimpleSession | null;
  docs?: DocMeta[];
  openDocId?: string | null;
  onDocOpenChange?: (id: string | null) => void;
};

export default function SessionView(props: SessionViewProps) {
  const {
    live, paused, latency,
    left, right,
    start, pauseResume, stop,
    sourceChoice, setSourceChoice,
  } = useSessionRealtime();

  // mock session header (you can replace with real data loader)
  const session: SimpleSession | null = props.session ?? {
    id: "live",
    title: "Live Meeting",
    createdAt: new Date().toISOString(),
    accumMs: 0,
    docIds: [],
  };

  const docs = props.docs ?? [];
  const openDocId = props.openDocId ?? null;

  const formatDuration = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const hh = String(Math.floor(s / 3600)).padStart(2, "0");
    const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  };
  const formatIsoDate = (iso?: string) => (iso ? new Date(iso).toLocaleString() : "");
  const fileExtension = (f: string) => f.split(".").pop() ?? "";
  const formatFileSize = (n: number) => `${(n / 1024).toFixed(1)} KB`;

  const leftItems: TranscriptSegment[] = left.map((s) => ({
    id: s.id, time: s.time, text: s.text, speaker: "Source", q: s.q,
  }));
  const rightItems: TranscriptSegment[] = right.map((s) => ({
    id: s.id, time: s.time, text: s.text, speaker: "Translation", q: false,
  }));

  const activeDoc = openDocId ? docs.find((d) => d.id === openDocId) : null;

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
                <Button key={id} size="sm" variant="outline" onClick={() => props.onDocOpenChange?.(id)}>
                  <FileText className="h-3 w-3 mr-1" /> {doc.filename}
                </Button>
              );
            })}
          </div>

          <div className="col-span-12 md:col-span-3 flex items-center gap-2 mt-2 md:mt-0">
            <Select
              value={sourceChoice}
              onValueChange={(v: "mic" | "system") => setSourceChoice(v)}
            >
              <SelectTrigger className="w-full md:w-40">
                <SelectValue placeholder="Source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="mic">
                  <div className="flex items-center gap-2"><Mic className="h-4 w-4" /> Microphone</div>
                </SelectItem>
                <SelectItem value="system">
                  <div className="flex items-center gap-2"><Radio className="h-4 w-4" /> System Audio</div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="col-span-12 md:col-span-3 flex flex-wrap items-center justify-start md:justify-end gap-2 md:gap-3 mt-2 md:mt-0">
            <Button
              variant="outline"
              size="iconLg"
              onClick={() => {}}
              aria-label="Link documents"
              className="rounded-full border border-blue-200 text-blue-600 hover:bg-blue-50 hover:text-blue-700"
            >
              <Link2 className="h-5 w-5" />
            </Button>
            {!live && (
              <Button onClick={() => start({ source: sourceChoice })}>
                <Play className="mr-2 h-4 w-4" /> Start
              </Button>
            )}
            {live && (
              <>
                <Button variant="outline" onClick={pauseResume}>
                  {!paused ? (<><Pause className="mr-2 h-4 w-4" /> Pause</>) : (<><Play className="mr-2 h-4 w-4" /> Resume</>)}
                </Button>
                <Button variant="destructive" onClick={stop}>
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
          <Card className="mb-3">
            <CardHeader className="py-3 flex flex-row items-center justify-between">
              <CardTitle className="text-base">{activeDoc.filename}</CardTitle>
              <div className="flex items-center gap-2">
                <Button size="sm" variant="outline">{"<"}</Button>
                <span className="text-sm">Page 1</span>
                <Button size="sm" variant="outline">{">"}</Button>
                <Button size="sm" variant="ghost" onClick={() => props.onDocOpenChange?.(null)}>Close</Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="text-xs text-muted-foreground">
                {activeDoc.filename.split(".").pop()} | {formatFileSize(activeDoc.size)} | Uploaded {formatIsoDate(activeDoc.uploadedAt)}
              </div>
              {activeDoc.notes && <div className="text-sm text-muted-foreground">{activeDoc.notes}</div>}
              <div className="h-[35vh] overflow-x-auto border rounded grid place-items-center text-sm text-muted-foreground">
                (PDF preview mock)
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-4 px-4 py-4">
        <div>
          <Pane title={`Transcript (source)`} items={leftItems.length ? leftItems : [{
            id: 0, time: "00:00:00", text: "Waiting for audio…", speaker: "ASR", q: false,
          }]} />
        </div>
        <div>
          <Pane title="Translation (auto)" items={rightItems.length ? rightItems : [{
            id: 0, time: "00:00:00", text: "Waiting for audio…", speaker: "MT", q: false,
          }]} />
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
                <Button variant="ghost" size="iconLg" className="rounded-full border border-sky-200 text-sky-600 hover:bg-sky-50 hover:text-sky-700">
                  <Highlighter className="h-9 w-9" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Auto-highlight questions</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="iconLg" className="rounded-full border border-emerald-200 text-emerald-600 hover:bg-emerald-50 hover:text-emerald-700">
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
