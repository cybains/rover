use client;
import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import {
  Mic,
  Radio,
  Play,
  Pause,
  Square,
  Bookmark,
  Settings,
  Download,
  Cpu,
  Timer,
  Clock,
  FolderDown,
  Highlighter,
  ListChecks,
  Plus,
  Search,
  FileText,
  Link2,
  History,
  Trash2,
  Sparkles,
  Wrench,
  BookOpen,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Checkbox } from "@/components/ui/checkbox";

type TranscriptSegment = { id: number; time: string; text: string; speaker: string; q: boolean };

type DocMeta = { id: string; title: string; type: string; size: string; addedAt: string };

type ActiveView = "session" | "sessions" | "docs" | "summaries" | "flashcards" | "quizzes" | "explain" | "bookmarks" | "settings" | "exports" | "developer";

type SimpleSession = { id: string; title: string; subject?: string; course?: string; language?: string; tags?: string; docIds: string[]; finished: boolean; createdAt: string; accumMs: number };

const sampleTranscript: TranscriptSegment[] = [
  { id: 1, time: "00:00:02", text: "Guten Morgen, alle zusammen. Heute sprechen wir über lineare Regression.", speaker: "Dr. Müller", q: false },
  { id: 2, time: "00:00:08", text: "Was sind die Grundannahmen dieses Modells?", speaker: "Student", q: true },
  { id: 3, time: "00:00:12", text: "Die Annahmen umfassen Linearität, Unabhängigkeit, Homoskedastizität und Normalverteilung der Fehler.", speaker: "Dr. Müller", q: false },
];

const sampleTranslation: TranscriptSegment[] = [
  { id: 1, time: "00:00:02", text: "Good morning, everyone. Today we will talk about linear regression.", speaker: "Dr. Müller", q: false },
  { id: 2, time: "00:00:08", text: "What are the basic assumptions of this model?", speaker: "Student", q: true },
  { id: 3, time: "00:00:12", text: "The assumptions include linearity, independence, homoskedasticity, and normal distribution of errors.", speaker: "Dr. Müller", q: false },
];

const initialDocs: DocMeta[] = [
  { id: "d1", title: "Linear Regression – Lecture Notes.pdf", type: "pdf", size: "1.2 MB", addedAt: "2025-09-10" },
  { id: "d2", title: "Statistics Glossary.md", type: "md", size: "24 KB", addedAt: "2025-09-12" },
  { id: "d3", title: "Machine Learning Syllabus.pdf", type: "pdf", size: "890 KB", addedAt: "2025-09-14" },
];

function LatencyPill({ ms }: { ms: number }) {
  const color = ms < 400 ? "bg-emerald-500" : ms < 800 ? "bg-amber-500" : "bg-rose-500";
  return (
    <div className={`flex items-center gap-2 rounded-full px-3 py-1 text-white ${color}`}>
      <Timer className="h-4 w-4" />
      <span className="text-sm">{ms} ms</span>
    </div>
  );
}

function StatusPill({ live }: { live: boolean }) {
  return (
    <Badge variant={live ? "default" : "secondary"} className="rounded-full px-3 py-1">
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${live ? "bg-emerald-500" : "bg-gray-400"}`} />
        {live ? "Live" : "Idle"}
      </div>
    </Badge>
  );
}

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
                        <Button variant="ghost" size="icon"><Bookmark className="h-4 w-4" /></Button>
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

const fmtDuration = (ms: number) => {
  const s = Math.floor(ms / 1000);
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
};

function DocViewer({ doc, onClose }: { doc: DocMeta; onClose: () => void }) {
  const [page, setPage] = useState<number>(1);
  return (
    <Card className="mb-3">
      <CardHeader className="py-3 flex flex-row items-center justify-between">
        <CardTitle className="text-base">{doc.title}</CardTitle>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => setPage((p) => Math.max(1, p - 1))}>◀</Button>
          <span className="text-sm">Page {page}</span>
          <Button size="sm" variant="outline" onClick={() => setPage((p) => p + 1)}>▶</Button>
          <Button size="sm" variant="ghost" onClick={onClose}>Close</Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[35vh] overflow-x-auto border rounded grid place-items-center text-sm text-muted-foreground">(PDF preview mock)</div>
      </CardContent>
    </Card>
  );
}

export default function LearningAppUI() {
  const [live, setLive] = useState<boolean>(false);
  const [paused, setPaused] = useState<boolean>(false);
  const [latency, setLatency] = useState<number>(320);
  const [menuOpen, setMenuOpen] = useState<boolean>(false);
  const [activeView, setActiveView] = useState<ActiveView>("session");
  const [firstRun, setFirstRun] = useState<boolean>(() => (typeof window !== "undefined" ? localStorage.getItem("ll_first_run_seen") !== "true" : true));
  const [session, setSession] = useState<SimpleSession | null>(() => {
    try {
      if (typeof window === "undefined") return null;
      const saved = localStorage.getItem("currentSession");
      return saved ? (JSON.parse(saved) as SimpleSession) : null;
    } catch {
      return null;
    }
  });
  const [sessions, setSessions] = useState<SimpleSession[]>(() => {
    try {
      if (typeof window === "undefined") return [];
      const saved = localStorage.getItem("sessions");
      return saved ? (JSON.parse(saved) as SimpleSession[]) : [];
    } catch {
      return [];
    }
  });
  const [sessionInitOpen, setSessionInitOpen] = useState<boolean>(false);
  const [docs, setDocs] = useState<DocMeta[]>(initialDocs);
  const [docSearch, setDocSearch] = useState<string>("");
  const [docPickerOpen, setDocPickerOpen] = useState<boolean>(false);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [openDocId, setOpenDocId] = useState<string | null>(null);

  useEffect(() => {
    const id = setInterval(() => setLatency((l) => Math.max(120, Math.min(1200, Math.round(l + (Math.random() * 80 - 40))))), 1200);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    try {
      if (session) localStorage.setItem("currentSession", JSON.stringify(session));
      else localStorage.removeItem("currentSession");
    } catch {}
  }, [session]);
  useEffect(() => {
    try {
      localStorage.setItem("sessions", JSON.stringify(sessions));
    } catch {}
  }, [sessions]);
  useEffect(() => {
    try {
      localStorage.setItem("ll_docs", JSON.stringify(docs));
    } catch {}
  }, [docs]);
  useEffect(() => {
    try {
      const storedDocs = localStorage.getItem("ll_docs");
      if (storedDocs) setDocs(JSON.parse(storedDocs) as DocMeta[]);
    } catch {}
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem("ll_first_run_seen", String(!firstRun));
    } catch {}
  }, [firstRun]);

  const MenuButton = ({ label, icon: Icon, value }: { label: string; icon: LucideIcon; value: ActiveView }) => (
    <Button
      variant={value === activeView ? "secondary" : "ghost"}
      className="w-full justify-start"
      onClick={() => {
        setActiveView(value);
        setMenuOpen(false);
        setFirstRun(false);
      }}
    >
      <Icon className="mr-2 h-4 w-4" /> {label}
    </Button>
  );

  const filteredDocs = docs.filter((d) => d.title.toLowerCase().includes(docSearch.toLowerCase()));

  const toggleDocSelect = (id: string) => {
    setSelectedDocIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  const linkSelectedToSession = () => {
    if (!session) return;
    const merged = Array.from(new Set([...(session.docIds || []), ...selectedDocIds]));
    setSession({ ...session, docIds: merged });
    setOpenDocId(merged[merged.length - 1] || null);
    setSelectedDocIds([]);
    setDocPickerOpen(false);
  };

  const onClickStart = () => {
    if (!session) {
      setSessionInitOpen(true);
      return;
    }
    setLive(true);
    setPaused(false);
  };

  const onClickPauseOrResume = () => {
    if (!session) return;
    if (live && !paused) {
      setPaused(true);
      setLive(false);
    } else {
      setPaused(false);
      setLive(true);
    }
  };

  const onClickStop = () => {
    if (!session) return;
    const finished: SimpleSession = { ...session, finished: true, createdAt: session.createdAt || new Date().toISOString() };
    setSessions((prev) => [finished, ...prev]);
    setSession(null);
    setLive(false);
    setPaused(false);
    setOpenDocId(null);
  };

  const deleteSession = (id: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
  };
  const deleteDoc = (id: string) => {
    setDocs((prev) => prev.filter((d) => d.id !== id));
    setSession((s) => (s ? { ...s, docIds: (s.docIds || []).filter((x) => x !== id) } : s));
    if (openDocId === id) setOpenDocId(null);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {firstRun && (
        <div className="fixed inset-0 z-40 grid place-items-center bg-gradient-to-b from-background to-muted">
          <div className="w-full max-w-3xl text-center p-8">
            <div className="mx-auto h-14 w-14 rounded-2xl bg-foreground text-background grid place-items-center text-2xl font-bold">Λ</div>
            <div className="mt-3 text-3xl font-semibold">Lab</div>
            <div className="text-sm text-muted-foreground mb-6">Personal build</div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <Button className="h-20" onClick={() => { setActiveView("session"); setFirstRun(false); }}><Play className="mr-2"/>Start a Session</Button>
              <Button variant="secondary" className="h-20" onClick={() => { setActiveView("sessions"); setFirstRun(false); }}><History className="mr-2"/>Sessions</Button>
              <Button variant="secondary" className="h-20" onClick={() => { setActiveView("docs"); setFirstRun(false); }}><FileText className="mr-2"/>Docs</Button>
              <Button variant="outline" className="h-20" onClick={() => { setActiveView("summaries"); setFirstRun(false); }}><FileText className="mr-2"/>Summaries</Button>
              <Button variant="outline" className="h-20" onClick={() => { setActiveView("flashcards"); setFirstRun(false); }}><BookOpen className="mr-2"/>Flashcards</Button>
              <Button variant="outline" className="h-20" onClick={() => { setActiveView("quizzes"); setFirstRun(false); }}><ListChecks className="mr-2"/>Quizzes</Button>
            </div>
            <Button variant="ghost" className="mt-6" onClick={() => setFirstRun(false)}>Enter app</Button>
          </div>
        </div>
      )}

      <div className="sticky top-0 z-20 backdrop-blur supports-[backdrop-filter]:bg-background/70 border-b">
        <div className="max-w-7xl mx-auto flex items-center justify-between px-4 py-3">
          <button className="flex items-center gap-3 group cursor-pointer" onClick={() => setMenuOpen(true)}>
            <div className="h-8 w-8 rounded-xl bg-foreground/90 text-background grid place-items-center font-bold group-hover:opacity-90">Λ</div>
            <div>
              <div className="font-semibold leading-tight group-hover:underline">Lab</div>
              <div className="text-xs text-muted-foreground">Personal build</div>
            </div>
            <Badge variant="outline" className="ml-2">v0.6 UI</Badge>
          </button>

          <div className="flex items-center gap-3">
            <LatencyPill ms={latency} />
            {session && (
              <div className="flex items-center gap-2">
                <Badge variant="secondary">{session.title}</Badge>
                {session.docIds?.map((id) => {
                  const d = docs.find((x) => x.id === id);
                  if (!d) return null;
                  return (
                    <Badge key={id} variant="outline" className="gap-1 cursor-pointer" onClick={() => setOpenDocId(id)}>
                      <FileText className="h-3 w-3" /> {d.title}
                    </Badge>
                  );
                })}
              </div>
            )}
            <StatusPill live={live} />
          </div>
        </div>
      </div>

      <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
        <SheetContent side="left" className="w-[300px]">
          <SheetHeader>
            <SheetTitle>Main Menu</SheetTitle>
          </SheetHeader>
          <div className="mt-4 space-y-2">
            <MenuButton label="Start a Session" icon={Play as LucideIcon} value="session" />
            <MenuButton label="Sessions" icon={History as LucideIcon} value="sessions" />
            <MenuButton label="Docs" icon={FileText as LucideIcon} value="docs" />
            <MenuButton label="Summaries" icon={FileText as LucideIcon} value="summaries" />
            <MenuButton label="Flashcards" icon={BookOpen as LucideIcon} value="flashcards" />
            <MenuButton label="Quizzes" icon={ListChecks as LucideIcon} value="quizzes" />
            <MenuButton label="Explain" icon={Sparkles as LucideIcon} value="explain" />
            <MenuButton label="Bookmarks" icon={Bookmark as LucideIcon} value="bookmarks" />
            <MenuButton label="Settings" icon={Settings as LucideIcon} value="settings" />
            <MenuButton label="Exports" icon={Download as LucideIcon} value="exports" />
            <MenuButton label="Developer" icon={Wrench as LucideIcon} value="developer" />
          </div>
        </SheetContent>
      </Sheet>

      <Dialog open={sessionInitOpen} onOpenChange={setSessionInitOpen}>
        <DialogContent className="sm:max-w-[560px]">
          <DialogHeader>
            <DialogTitle>New Session</DialogTitle>
          </DialogHeader>
          <div className="grid gap-3 py-2">
            <div>
              <div className="text-sm font-medium mb-1">Session Title *</div>
              <Input placeholder="e.g., Linear Regression – Week 3" id="session-title" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="text-sm font-medium mb-1">Subject</div>
                <Input placeholder="e.g., Statistics" id="session-subject" />
              </div>
              <div>
                <div className="text-sm font-medium mb-1">Course</div>
                <Input placeholder="e.g., ML101" id="session-course" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="text-sm font-medium mb-1">Language</div>
                <Input placeholder="e.g., German" id="session-language" />
              </div>
              <div>
                <div className="text-sm font-medium mb-1">Tags</div>
                <Input placeholder="comma, separated, tags" id="session-tags" />
              </div>
            </div>
            <div className="pt-1">
              <Button variant="outline" onClick={() => setDocPickerOpen(true)}>
                <Link2 className="mr-2 h-4 w-4" /> Link docs now
              </Button>
            </div>
          </div>
          <DialogFooter>
            <Button onClick={() => {
              const title = (document.getElementById("session-title") as HTMLInputElement | null)?.value?.trim();
              if (!title) return;
              const subject = (document.getElementById("session-subject") as HTMLInputElement | null)?.value?.trim();
              const course = (document.getElementById("session-course") as HTMLInputElement | null)?.value?.trim();
              const language = (document.getElementById("session-language") as HTMLInputElement | null)?.value?.trim();
              const tags = (document.getElementById("session-tags") as HTMLInputElement | null)?.value?.trim();
              const createdAt = new Date().toISOString();
              const newSess: SimpleSession = { id: crypto.randomUUID(), title, subject, course, language, tags, docIds: selectedDocIds, finished: false, createdAt, accumMs: 0 };
              setSession(newSess);
              setLive(true);
              setPaused(false);
              setSessionInitOpen(false);
              setSelectedDocIds([]);
              setFirstRun(false);
            }}>Start Session</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={docPickerOpen} onOpenChange={setDocPickerOpen}>
        <DialogContent className="sm:max-w-[680px]">
          <DialogHeader>
            <DialogTitle>Select documents to link</DialogTitle>
          </DialogHeader>
          <div className="flex items-center gap-2 mb-3">
            <div className="relative w-full">
              <Search className="h-4 w-4 absolute left-2 top-2.5 opacity-60" />
              <Input className="pl-8" placeholder="Search docs…" value={docSearch} onChange={(e) => setDocSearch(e.target.value)} />
            </div>
            <Button variant="secondary" onClick={() => setDocs((prev) => [...prev, { id: crypto.randomUUID(), title: `New Doc ${prev.length + 1}.pdf`, type: "pdf", size: "500 KB", addedAt: new Date().toISOString().slice(0, 10) }])}>
              <Plus className="mr-2 h-4 w-4" /> Add File
            </Button>
          </div>
          <div className="max-h-[46vh] overflow-y-auto space-y-2 pr-1">
            {filteredDocs.map((doc) => (
              <div key={doc.id} className="flex items-center justify-between border rounded-lg p-3">
                <div className="flex items-center gap-3">
                  <Badge variant="secondary" className="gap-1"><FileText className="h-3 w-3" /> {doc.type.toUpperCase()}</Badge>
                  <div>
                    <div className="font-medium text-sm">{doc.title}</div>
                    <div className="text-xs text-muted-foreground">{doc.size} · added {doc.addedAt}</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Checkbox id={`sel-${doc.id}`} checked={selectedDocIds.includes(doc.id)} onCheckedChange={() => toggleDocSelect(doc.id)} />
                  <label htmlFor={`sel-${doc.id}`} className="text-sm">Select</label>
                </div>
              </div>
            ))}
            {filteredDocs.length === 0 && (<div className="text-sm text-muted-foreground p-6 text-center">No documents match your search.</div>)}
          </div>
          <DialogFooter>
            <Button disabled={!session} onClick={linkSelectedToSession}>
              <Link2 className="mr-2 h-4 w-4" /> Link to current session
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {activeView === "session" && (
        <>
          <div className="border-b">
            <div className="max-w-7xl mx-auto grid grid-cols-12 items-center gap-3 px-4 py-3">
              <div className="col-span-12 md:col-span-7 flex items-center gap-2 flex-wrap">
                <Badge variant="secondary" className="gap-2"><Cpu className="h-3 w-3" /> Local {session ? `· ${session.title}` : ""}</Badge>
                {session?.docIds.map((id) => {
                  const d = docs.find((x) => x.id === id);
                  if (!d) return null;
                  return (
                    <Button key={id} size="sm" variant="outline" onClick={() => setOpenDocId(id)}>
                      <FileText className="h-3 w-3 mr-1" /> {d.title}
                    </Button>
                  );
                })}
              </div>
              <div className="col-span-6 md:col-span-3 flex items-center gap-2">
                <Select defaultValue="mic">
                  <SelectTrigger className="w-40"><SelectValue placeholder="Source" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mic"><div className="flex items-center gap-2"><Mic className="h-4 w-4" /> Microphone</div></SelectItem>
                    <SelectItem value="loopback"><div className="flex items-center gap-2"><Radio className="h-4 w-4" /> System Audio</div></SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="col-span-6 md:col-span-2 flex items-center justify-end gap-2">
                <Button variant="outline" onClick={() => setDocPickerOpen(true)} aria-label="Link documents"><Link2 className="h-4 w-4" /></Button>
                <Button onClick={onClickStart}><Play className="mr-2 h-4 w-4" /> Start</Button>
                {session && (<>
                  <Button variant="outline" onClick={onClickPauseOrResume}>{(live && !paused) ? (<><Pause className="mr-2 h-4 w-4" /> Pause</>) : (<><Play className="mr-2 h-4 w-4" /> Resume</>)}</Button>
                  <Button variant="destructive" onClick={onClickStop}><Square className="mr-2 h-4 w-4" /> Stop</Button>
                </>)}
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="outline" size="icon" aria-label="Bookmark moment"><Bookmark className="h-4 w-4" /></Button>
                    </TooltipTrigger>
                    <TooltipContent>Bookmark (B)</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            </div>
          </div>

          {openDocId && (
            <div className="max-w-7xl mx-auto px-4 mt-3">
              {(() => {
                const d = docs.find((x) => x.id === openDocId);
                if (!d) return null;
                return <DocViewer doc={d} onClose={() => setOpenDocId(null)} />;
              })()}
            </div>
          )}

          <div className="max-w-7xl mx-auto grid grid-cols-2 gap-4 px-4 py-4">
            <div><Pane title={`Transcript (DE)${session ? ` — ${session.title}` : ""}`} items={sampleTranscript} /></div>
            <div><Pane title="Translation (EN)" items={sampleTranslation} /></div>
          </div>

          <div className="border-t">
            <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-3">
              <Clock className="h-4 w-4" />
              <div className="w-full">
                <Progress value={live ? Math.min(100, latency / 12) : 0} />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>{session ? fmtDuration(session.accumMs) : "00:00:00"}</span>
                  <span>{session ? new Date(session.createdAt).toLocaleTimeString() : ""}</span>
                </div>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild><Button variant="ghost" size="icon"><Highlighter className="h-4 w-4" /></Button></TooltipTrigger>
                  <TooltipContent>Auto-highlight questions</TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild><Button variant="ghost" size="icon"><ListChecks className="h-4 w-4" /></Button></TooltipTrigger>
                  <TooltipContent>Enforce glossary</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
        </>
      )}

      {activeView === "sessions" && (
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between mb-4">
            <div className="text-2xl font-semibold">Sessions</div>
            <div className="text-sm text-muted-foreground">{sessions.length} total</div>
          </div>
          {sessions.length === 0 ? (
            <Card><CardContent className="py-8 text-center text-sm text-muted-foreground">No finished sessions yet. Start one from the menu.</CardContent></Card>
          ) : (
            <div className="space-y-2">
              {sessions.map((s) => (
                <Card key={s.id}>
                  <CardContent className="py-4 flex items-center justify-between">
                    <div>
                      <div className="font-medium">{s.title}</div>
                      <div className="text-xs text-muted-foreground">{new Date(s.createdAt).toLocaleString()} · {fmtDuration(s.accumMs)} · {s.docIds?.length || 0} doc(s) · Locked</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">Finished</Badge>
                      <Button size="sm" variant="outline" onClick={() => alert("Open session viewer (todo)")}>View</Button>
                      <Button size="icon" variant="ghost" aria-label="Delete session" onClick={() => deleteSession(s.id)}><Trash2 className="h-4 w-4" /></Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {activeView === "docs" && (
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between mb-3">
            <div className="text-2xl font-semibold">Docs</div>
            <div className="flex items-center gap-2">
              <Button variant="secondary" onClick={() => setDocPickerOpen(true)}><Link2 className="mr-2 h-4 w-4" /> Link to Session</Button>
              <Button onClick={() => setDocs((prev) => [...prev, { id: crypto.randomUUID(), title: `New Doc ${prev.length + 1}.pdf`, type: "pdf", size: "500 KB", addedAt: new Date().toISOString().slice(0, 10) }])}><Plus className="mr-2 h-4 w-4" /> Add File</Button>
            </div>
          </div>
          <div className="flex items-center gap-2 mb-3">
            <div className="relative w-full">
              <Search className="h-4 w-4 absolute left-2 top-2.5 opacity-60" />
              <Input className="pl-8" placeholder="Search docs…" value={docSearch} onChange={(e) => setDocSearch(e.target.value)} />
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {filteredDocs.map((doc) => (
              <Card key={doc.id} className="overflow-hidden">
                <CardHeader className="py-3"><CardTitle className="text-base flex items-center gap-2"><FileText className="h-4 w-4" /> {doc.title}</CardTitle></CardHeader>
                <CardContent className="text-sm text-muted-foreground">
                  <div className="flex items-center justify-between"><div>{doc.type.toUpperCase()} · {doc.size}</div><div>Added {doc.addedAt}</div></div>
                  <div className="mt-3 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Checkbox id={`pick-${doc.id}`} checked={selectedDocIds.includes(doc.id)} onCheckedChange={() => toggleDocSelect(doc.id)} />
                      <label htmlFor={`pick-${doc.id}`}>Select</label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button size="sm" variant="outline" onClick={() => setOpenDocId(doc.id)}>Preview</Button>
                      <Button size="icon" variant="ghost" aria-label="Delete doc" onClick={() => deleteDoc(doc.id)}><Trash2 className="h-4 w-4" /></Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          {filteredDocs.length === 0 && (<div className="text-sm text-muted-foreground p-6 text-center">No documents match your search.</div>)}
        </div>
      )}

      {activeView === "summaries" && (
        <div className="max-w-5xl mx-auto px-4 py-10">
          <div className="text-2xl font-semibold mb-4">Summaries</div>
          <Card>
            <CardContent className="py-6 space-y-4">
              <div className="text-sm text-muted-foreground">Modes: Short / Medium / Long. Export: PDF / Markdown.</div>
              <div className="flex items-center gap-2">
                <Select defaultValue="medium">
                  <SelectTrigger className="w-40"><SelectValue placeholder="Mode" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="short">Short</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="long">Long</SelectItem>
                  </SelectContent>
                </Select>
                <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> Export PDF</Button>
                <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> Export Markdown</Button>
              </div>
              <div className="p-3 border rounded text-sm text-muted-foreground">(Preview will appear here)</div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === "flashcards" && (
        <div className="max-w-5xl mx-auto px-4 py-10">
          <div className="text-2xl font-semibold mb-4">Flashcards</div>
          <Card>
            <CardContent className="py-6 space-y-4">
              <div className="text-sm text-muted-foreground">Auto-generate Q/A from session. Review mode like Anki.</div>
              <div className="flex items-center gap-2"><Button>Generate</Button><Button variant="secondary">Start Review</Button></div>
              <div className="p-3 border rounded text-sm text-muted-foreground">(Generated cards will appear here)</div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === "quizzes" && (
        <div className="max-w-5xl mx-auto px-4 py-10">
          <div className="text-2xl font-semibold mb-4">Quizzes</div>
          <Card>
            <CardContent className="py-6 space-y-4">
              <div className="text-sm text-muted-foreground">Multiple-choice or open-ended. Difficulty selector.</div>
              <div className="flex items-center gap-2">
                <Select defaultValue="mcq">
                  <SelectTrigger className="w-44"><SelectValue placeholder="Type" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mcq">Multiple-choice</SelectItem>
                    <SelectItem value="open">Open-ended</SelectItem>
                  </SelectContent>
                </Select>
                <Select defaultValue="med">
                  <SelectTrigger className="w-36"><SelectValue placeholder="Difficulty" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="easy">Easy</SelectItem>
                    <SelectItem value="med">Medium</SelectItem>
                    <SelectItem value="hard">Hard</SelectItem>
                  </SelectContent>
                </Select>
                <Button>Generate</Button>
              </div>
              <div className="p-3 border rounded text-sm text-muted-foreground">(Quiz will appear here)</div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === "explain" && (
        <div className="max-w-5xl mx-auto px-4 py-10">
          <div className="text-2xl font-semibold mb-4">Explain</div>
          <Card>
            <CardContent className="py-6 space-y-4">
              <div className="text-sm text-muted-foreground">Levels: Simple, Medium, Expert. Rewrites selected passage accordingly.</div>
              <div className="flex items-center gap-2">
                <Select defaultValue="simple">
                  <SelectTrigger className="w-40"><SelectValue placeholder="Level" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="simple">Simple</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="expert">Expert</SelectItem>
                  </SelectContent>
                </Select>
                <Button variant="secondary"><Sparkles className="mr-2 h-4 w-4" />Rewrite</Button>
              </div>
              <div className="p-3 border rounded text-sm text-muted-foreground">(Explanation will appear here)</div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === "bookmarks" && (
        <div className="max-w-5xl mx-auto px-4 py-16">
          <div className="text-2xl font-semibold mb-4">Bookmarks</div>
          <Card><CardContent className="py-6 text-sm text-muted-foreground">Your saved moments will appear here.</CardContent></Card>
        </div>
      )}

      {activeView === "settings" && (
        <div className="max-w-5xl mx-auto px-4 py-16">
          <div className="text-2xl font-semibold mb-4">Settings</div>
          <Card>
            <CardContent className="grid grid-cols-2 gap-4 py-6">
              <div><div className="text-sm font-medium mb-1">Chunk Size (ms)</div><Input placeholder="500" defaultValue={500} /></div>
              <div><div className="text-sm font-medium mb-1">Silence Gate (ms)</div><Input placeholder="120" defaultValue={120} /></div>
              <div className="col-span-2 flex items-center justify-between border rounded-lg p-3">
                <div><div className="text-sm font-medium">Force Input Language</div><div className="text-xs text-muted-foreground">Avoid auto-detect drift</div></div>
                <Switch />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === "exports" && (
        <div className="max-w-5xl mx-auto px-4 py-16">
          <div className="text-2xl font-semibold mb-4">Exports</div>
          <div className="grid grid-cols-2 gap-3">
            <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> JSONL</Button>
            <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> VTT / SRT</Button>
            <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> Transcripts</Button>
            <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> Study Pack</Button>
          </div>
          <div className="text-xs text-muted-foreground mt-3">Exports are local; nothing leaves your machine.</div>
        </div>
      )}

      {activeView === "developer" && (
        <div className="max-w-5xl mx-auto px-4 py-16">
          <div className="text-2xl font-semibold mb-4">Developer</div>
          <Card>
            <CardContent className="py-6 text-sm text-muted-foreground space-y-2">
              <div>Build: Local-only UI shell</div>
              <div>Planned: wire ASR/MT, PDF renderer, session viewer, hotkeys</div>
              <div>Env checks, logs, and preflight</div>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="border-t bg-muted/30">
        <div className="max-w-7xl mx-auto px-4 py-3 text-xs text-muted-foreground flex items-center justify-between">
          <div className="flex items-center gap-3"><span className="font-medium">Hotkeys:</span><span>Start/Resume ⌘/Ctrl+K</span><span>Bookmark B</span><span>Open Menu ⌘/Ctrl+M</span></div>
          <div>Built for: You · 100% local · Vienna</div>
        </div>
      </div>
    </div>
  );
}

if (typeof window !== "undefined") {
  try {
    const a = Array.from(new Set(["a", "b", "b", "c"]));
    console.assert(JSON.stringify(a) === JSON.stringify(["a", "b", "c"]));
    console.assert(typeof BookOpen !== "undefined");
    console.assert("grid grid-cols-2".includes("grid-cols-2"));
  } catch {}
}
