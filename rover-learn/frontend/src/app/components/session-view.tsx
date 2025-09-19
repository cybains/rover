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

const sampleTranscript: TranscriptSegment[] = [
  {
    id: 1,
    time: "00:00:02",
    text: "Guten Morgen, alle zusammen. Heute sprechen wir über lineare Regression.",
    speaker: "Dr. Müller",
    q: false,
  },
  {
    id: 2,
    time: "00:00:08",
    text: "Was sind die Grundannahmen dieses Modells?",
    speaker: "Student",
    q: true,
  },
  {
    id: 3,
    time: "00:00:12",
    text: "Die Annahmen umfassen Linearität, Unabhängigkeit, Homoskedastizität und Normalverteilung der Fehler.",
    speaker: "Dr. Müller",
    q: false,
  },
];

const sampleTranslation: TranscriptSegment[] = [
  {
    id: 1,
    time: "00:00:02",
    text: "Good morning, everyone. Today we will talk about linear regression.",
    speaker: "Dr. Müller",
    q: false,
  },
  {
    id: 2,
    time: "00:00:08",
    text: "What are the basic assumptions of this model?",
    speaker: "Student",
    q: true,
  },
  {
    id: 3,
    time: "00:00:12",
    text: "The assumptions include linearity, independence, homoskedasticity, and normal distribution of errors.",
    speaker: "Dr. Müller",
    q: false,
  },
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
        <div ref={listRef} className="h-[58vh] overflow-y-auto pr-2 space-y-3">
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

function DocViewer({ doc, onClose, fileExtension, formatFileSize, formatIsoDate }: DocViewerProps) {
  const [page, setPage] = useState<number>(1);

  return (
    <Card className="mb-3">
      <CardHeader className="py-3 flex flex-row items-center justify-between">
        <CardTitle className="text-base">{doc.filename}</CardTitle>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => setPage((p) => Math.max(1, p - 1))}>
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
                <Button key={id} size="sm" variant="outline" onClick={() => onDocOpenChange(id)}>
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
              <Button onClick={onStart}>
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
                <Button variant="destructive" onClick={onStop}>
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

      <div className="max-w-7xl mx-auto grid grid-cols-2 gap-4 px-4 py-4">
        <div>
          <Pane title={`Transcript (DE)${session ? ` - ${session.title}` : ""}`} items={sampleTranscript} />
        </div>
        <div>
          <Pane title="Translation (EN)" items={sampleTranslation} />
        </div>
      </div>

      <div className="border-t">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-3">
          <Clock className="h-4 w-4" />
          <div className="w-full">
            <Progress value={live ? Math.min(100, latency / 12) : 0} />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>{session ? formatDuration(session.accumMs) : "00:00:00"}</span>
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
