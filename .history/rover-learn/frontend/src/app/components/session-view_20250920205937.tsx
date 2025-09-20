"use client";
import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Cpu, FileText, Mic, Radio, Link2, Play, Pause, Square,
  Bookmark, Highlighter, ListChecks, Clock,
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
import { asrHealth, asrStart, asrStop, asrOpenStream, type AsrEvent } from "@/app/lib/asr";

// --- existing sample translation kept for now
const sampleTranslation: TranscriptSegment[] = [
  { id: 1, time: "00:00:02", text: "Good morning, everyone. Today we will talk about linear regression.", speaker: "Dr. Müller", q: false },
  { id: 2, time: "00:00:08", text: "What are the basic assumptions of this model?", speaker: "Student", q: true },
  { id: 3, time: "00:00:12", text: "The assumptions include linearity, independence, homoskedasticity, and normal distribution of errors.", speaker: "Dr. Müller", q: false },
];

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
        <div ref={listRef} className="h-[58vh] overflow-y-auto pr-2 space-y-3">
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

function DocViewer({ doc, onClose, fileExtension, formatFileSize, formatIsoDate }: DocViewerProps) {
  const [page, setPage] = useState<number>(1);
  return (
    <Card className="mb-3">
      <CardHeader className="py-3 flex flex-row items-center justify-between">
        <CardTitle className="text-base">{doc.filename}</CardTitle>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => setPage((p) => Math.max(1, p - 1))}>{"<"}</Button>
          <span className="text-sm">Page {page}</span>
          <Button size="sm" variant="outline" onClick={() => setPage((p) => p + 1)}>{">"}</Button>
          <Button size="sm" variant="ghost" onClick={onClose}>Close</Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-xs text-muted-foreground">
          {fileExtension(doc.filename)} | {formatFileSize(doc.size)} | Uploaded {formatIsoDate(doc.uploadedAt)}
        </div>
        {doc.notes && <div className="text-sm text-muted-foreground">{doc.notes}</div>}
        <div className="h-[35vh] overflow-x-auto border rounded grid place-items-center text-sm text-muted-foreground">
          (PDF preview mock)
        </div>
      </CardContent>
    </Card>
  );
}

export default function SessionView({
  session, docs, openDocId, onDocOpenChange,
  live, paused, latency, onStart, onPauseResume, onStop,
  onLinkDocs, formatDuration, formatFileSize, formatIsoDate, fileExtension,
}: SessionViewProps) {
  const activeDoc = openDocId ? docs.find((doc) => doc.id === openDocId) :
