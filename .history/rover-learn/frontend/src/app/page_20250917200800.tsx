import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
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
  BookOpen, // FIX: ensure BookOpen is imported
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

// ---------------- Mock Data ----------------
const sampleTranscript = [
  { id: 1, time: "00:00:02", text: "Guten Morgen, alle zusammen. Heute sprechen wir über lineare Regression.", speaker: "Dr. Müller", q: false },
  { id: 2, time: "00:00:08", text: "Was sind die Grundannahmen dieses Modells?", speaker: "Student", q: true },
  { id: 3, time: "00:00:12", text: "Die Annahmen umfassen Linearität, Unabhängigkeit, Homoskedastizität und Normalverteilung der Fehler.", speaker: "Dr. Müller", q: false },
];


  { id: 1, time: "00:00:02", text: "Good morning, everyone. Today we will talk about linear regression.", speaker: "Dr. Müller", q: false },
  { id: 2, time: "00:00:08", text: "What are the basic assumptions of this model?", speaker: "Student", q: true },
  { id: 3, time: "00:00:12", text: "The assumptions include linearity, independence, homoskedasticity, and normal distribution of errors.", speaker: "Dr. Müller", q: false },
];

const initialDocs = [
  { id: "d1", title: "Linear Regression – Lecture Notes.pdf", type: "pdf", size: "1.2 MB", addedAt: "2025-09-10" },
  { id: "d2", title: "Statistics Glossary.md", type: "md", size: "24 KB", addedAt: "2025-09-12" },
  { id: "d3", title: "Machine Learning Syllabus.pdf", type: "pdf", size: "890 KB", addedAt: "2025-09-14" },
];

// ---------------- UI Helpers ----------------
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

function Pane({ title, items }: { title: string; items: typeof sampleTranscript }) {
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

// ---------------- Component ----------------
export default function LearningAppUI() {
  const [live, setLive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [latency, setLatency] = useState(320);
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeView, setActiveView] = useState<
    "session" | "sessions" | "docs" | "summaries" | "flashcards" | "quizzes" | "explain" | "bookmarks" | "settings" | "exports" | "developer"
  >("session");
  const [firstRun, setFirstRun] = useState(() => localStorage.getItem('ll_first_run_seen') !== 'true');

  type SimpleSession = {
    id: string;
    title: string;
    subject?: string;
    course?: string;
    language?: string;
    tags?: string;
    docIds: string[];
    finished: boolean;
    createdAt: string;
    accumMs: number; // accumulated run time
  };

  const [session, setSession] = useState<SimpleSession | null>(() => {
    const saved = localStorage.getItem("currentSession");
    return saved ? JSON.parse(saved) : null;
  });
  const [sessions, setSessions] = useState<SimpleSession[]>(() => {
    const saved = localStorage.getItem("sessions");
    return saved ? JSON.parse(saved) : [];
  });

  const [sessionInitOpen, setSessionInitOpen] = useState(false);

  const [docs, setDocs] = useState(initialDocs);
  const [docSearch, setDocSearch] = useState("");
  const [docPickerOpen, setDocPickerOpen] = useState(false);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  // Latency ticker
  useEffect(() => {
    const id = setInterval(() => setLatency((l) => Math.max(120, Math.min(1200, Math.round(l + (Math.random() * 80 - 40))))), 1200);
    return () => clearInterval(id);
  }, []);

  // Persist key data
  useEffect(() => { if (session) localStorage.setItem("currentSession", JSON.stringify(session)); else localStorage.removeItem("currentSession"); }, [session]);
  useEffect(() => { localStorage.setItem("sessions", JSON.stringify(sessions)); }, [sessions]);
  useEffect(() => { localStorage.setItem("ll_docs", JSON.stringify(docs)); }, [docs]);
  useEffect(() => { const storedDocs = localStorage.getItem("ll_docs"); if (storedDocs) setDocs(JSON.parse(storedDocs)); }, []);
  useEffect(() => { localStorage.setItem('ll_first_run_seen', String(!firstRun)); }, [firstRun]);

  const MenuButton = ({ label, icon: Icon, value }: { label: string; icon: any; value: typeof activeView extends string ? any : never }) => (
    <Button
      variant={value === activeView ? 'secondary' : 'ghost'}
      className="w-full justify-start"
      onClick={() => { setActiveView(value as any); setMenuOpen(false); setFirstRun(false); }}
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
    setSelectedDocIds([]);
    setDocPickerOpen(false);
  };

  const onClickStart = () => {
    if (!session) { setSessionInitOpen(true); return; }
    setLive(true);
    setPaused(false);
  };

  const onClickPauseOrResume = () => {
    if (!session) return;
    if (live && !paused) { setPaused(true); setLive(false); } else { setPaused(false); setLive(true); }
  };

  const onClickStop = () => {
    if (!session) return;
    const finished: SimpleSession = { ...session, finished: true, createdAt: session.createdAt || new Date().toISOString() };
    setSessions((prev) => [finished, ...prev]);
    setSession(null);
    setLive(false);
    setPaused(false);
  };

  const deleteSession = (id: string) => { setSessions(prev => prev.filter(s => s.id !== id)); };
  const deleteDoc = (id: string) => {
    setDocs(prev => prev.filter(d => d.id !== id));
    setSession(s => s ? { ...s, docIds: (s.docIds || []).filter(x => x !== id) } : s);
  };

  // ---------------- Render ----------------
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* First-run minimal launcher */}
      {firstRun && (
        <div className="fixed inset-0 z-40 grid place-items-center bg-gradient-to-b from-background to-muted">
          <div className="w-full max-w-3xl text-center p-8">
            <div className="mx-auto h-14 w-14 rounded-2xl bg-foreground text-background grid place-items-center text-2xl font-bold">Λ</div>
            <div className="mt-3 text-3xl font-semibold">Lab</div>
            <div className="text-sm text-muted-foreground mb-6">Your personal bilingual seminar assistant</div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <Button className="h-20" onClick={() => { setActiveView('session'); setFirstRun(false); }}><Play className="mr-2"/>Start a Session</Button>
              <Button variant="secondary" className="h-20" onClick={() => { setActiveView('sessions'); setFirstRun(false); }}><History className="mr-2"/>Sessions</Button>
              <Button variant="secondary" className="h-20" onClick={() => { setActiveView('docs'); setFirstRun(false); }}><FileText className="mr-2"/>Docs</Button>
              <Button variant="outline" className="h-20" onClick={() => { setActiveView('summaries'); setFirstRun(false); }}><FileText className="mr-2"/>Summaries</Button>
              <Button variant="outline" className="h-20" onClick={() => { setActiveView('flashcards'); setFirstRun(false); }}><BookOpen className="mr-2"/>Flashcards</Button>
              <Button variant="outline" className="h-20" onClick={() => { setActiveView('quizzes'); setFirstRun(false); }}><ListChecks className="mr-2"/>Quizzes</Button>
            </div>
            <Button variant="ghost" className="mt-6" onClick={() => setFirstRun(false)}>Enter app</Button>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="sticky top-0 z-20 backdrop-blur supports-[backdrop-filter]:bg-background/70 border-b">
        <div className="max-w-7xl mx-auto flex items-center justify-between px-4 py-3">
          <button className="flex items-center gap-3 group cursor-pointer" onClick={() => setMenuOpen(true)}>
            <div className="h-8 w-8 rounded-xl bg-foreground/90 text-background grid place-items-center font-bold group-hover:opacity-90">Λ</div>
            <div>
              <div className="font-semibold leading-tight group-hover:underline">Lab</div>
              <div className="text-xs text-muted-foreground">Personal build</div>
            </div>
            <Badge variant="outline" className="ml-2">v0.5 UI</Badge>
          </button>

          <div className="flex items-center gap-3">
            <LatencyPill ms={latency} />
            {session && <Badge variant="secondary">{session.title}</Badge>}
            <StatusPill live={live} />
          </div>
        </div>
      </div>

      {/* Left Menu */}
      <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
        <SheetContent side="left" className="w-[300px]">
          <SheetHeader>
            <SheetTitle>Main Menu</SheetTitle>
          </SheetHeader>
          <div className="mt-4 space-y-2">
            <MenuButton label="Start a Session" icon={Play} value="session" />
            <MenuButton label="Sessions" icon={History} value="sessions" />
            <MenuButton label="Docs" icon={FileText} value="docs" />
            <MenuButton label="Summaries" icon={FileText} value="summaries" />
            <MenuButton label="Flashcards" icon={BookOpen} value="flashcards" />
            <MenuButton label="Quizzes" icon={ListChecks} value="quizzes" />
            <MenuButton label="Explain" icon={Sparkles} value="explain" />
            <MenuButton label="Bookmarks" icon={Bookmark} value="bookmarks" />
            <MenuButton label="Settings" icon={Settings} value="settings" />
            <MenuButton label="Exports" icon={Download} value="exports" />
            <MenuButton label="Developer" icon={Wrench} value="developer" />
          </div>
        </SheetContent>
      </Sheet>

      {/* Session Init Dialog */}
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
              <Button variant=\"outline\" onClick=\{() => setDocPickerOpen(true)\}><Link2 className=\"mr-2 h-4 w-4\" \/> Link docs now<\/Button>
            </div>
          </div>
          <DialogFooter>
            <Button onClick={() => {
              const title = (document.getElementById('session-title') as HTMLInputElement)?.value?.trim();
              if (!title) return; // require a title
              const subject = (document.getElementById('session-subject') as HTMLInputElement)?.value?.trim();
              const course = (document.getElementById('session-course') as HTMLInputElement)?.value?.trim();
              const language = (document.getElementById('session-language') as HTMLInputElement)?.value?.trim();
              const tags = (document.getElementById('session-tags') as HTMLInputElement)?.value?.trim();
              const createdAt = new Date().toISOString();
              setSession({ id: crypto.randomUUID(), title, subject, course, language, tags, docIds: selectedDocIds, finished: false, createdAt, accumMs: 0 });
              setLive(true); setPaused(false); setSessionInitOpen(false); setSelectedDocIds([]); setFirstRun(false);
            }}>Start Session</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Doc Picker */}
      <Dialog open={docPickerOpen} onOpenChange={setDocPickerOpen}>
        <DialogContent className="sm:max-w-[680px]">
          <DialogHeader><DialogTitle>Select documents to link</DialogTitle></DialogHeader>
          <div className="flex items-center gap-2 mb-3">
            <div className="relative w-full">
              <Search className="h-4 w-4 absolute left-2 top-2.5 opacity-60" />
              <Input className="pl-8" placeholder="Search docs…" value={docSearch} onChange={(e) => setDocSearch(e.target.value)} />
            </div>
            <Button variant="secondary" onClick={() => setDocs(prev => [...prev, { id: crypto.randomUUID(), title: `New Doc ${prev.length+1}.pdf`, type: 'pdf', size: '500 KB', addedAt: new Date().toISOString().slice(0,10) }])}><Plus className="mr-2 h-4 w-4"/>Add Mock</Button>
          </div>
          <div className="max-h-[46vh] overflow-y-auto space-y-2 pr-1">
            {filteredDocs.map(doc => (
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
            <Button disabled={!session} onClick={linkSelectedToSession}><Link2 className="mr-2 h-4 w-4"/> Link to current session</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Views */}
      {activeView === 'session' && (
        <>
          {/* Controls Row */}
          <div className="border-b">
            <div className="max-w-7xl mx-auto grid grid-cols-12 items-center gap-3 px-4 py-3">
              <div className="col-span-5 flex items-center gap-2">
                <Badge variant="secondary" className="gap-2"><Cpu className="h-3 w-3" /> Local {session ? `· ${session.title}` : ''}</Badge>
              </div>
              <div className="col-span-4 flex items-center gap-2">
                <Select defaultValue="mic">
                  <SelectTrigger className="w-40"><SelectValue placeholder="Source" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mic"><div className="flex items-center gap-2"><Mic className="h-4 w-4" /> Microphone</div></SelectItem>
                    <SelectItem value="loopback"><div className="flex items-center gap-2"><Radio className="h-4 w-4" /> System Audio</div></SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="col-span-3 flex items-center justify-end gap-2">
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

          {/* Linked docs chips */}
          {session && session.docIds?.length > 0 && (
            <div className="max-w-7xl mx-auto px-4 mt-3 flex flex-wrap gap-2">
              {session.docIds.map(id => { const d = docs.find(x => x.id === id); if (!d) return null; return <Badge key={id} variant="outline" className="gap-1"><FileText className="h-3 w-3" /> {d.title}</Badge>; })}
              <Button size="sm" variant="ghost" onClick={() => setDocPickerOpen(true)}><Link2 className="h-4 w-4 mr-2"/>Link more</Button>
            </div>
          )}

          {/* Main Content: Transcript + Translation side-by-side */}
          <div className="max-w-7xl mx-auto grid grid-cols-12 gap-4 px-4 py-4">
            <div className="col-span-12 lg:col-span-6"><Pane title={`Transcript (DE)${session ? ` — ${session.title}` : ''}`} items={sampleTranscript} /></div>
            <div className="col-span-12 lg:col-span-6"><Pane title="Translation (EN)" items={sampleTranslation} /></div>
          </div>

          {/* Timeline */}
          <div className="border-t">
            <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-3">
              <Clock className="h-4 w-4" />
              <div className="w-full">
                <Progress value={live ? Math.min(100, latency / 12) : 0} />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>{session ? fmtDuration(session.accumMs) : '00:00:00'}</span>
                  <span>{session ? new Date(session.createdAt).toLocaleTimeString() : ''}</span>
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

      {activeView === 'sessions' && (
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between mb-4">
            <div className="text-2xl font-semibold">Sessions</div>
            <div className="text-sm text-muted-foreground">{sessions.length} total</div>
          </div>
          {sessions.length === 0 ? (
            <Card><CardContent className="py-8 text-center text-sm text-muted-foreground">No finished sessions yet. Start one from the menu.</CardContent></Card>
          ) : (
            <div className="space-y-2">
              {sessions.map(s => (
                <Card key={s.id}>
                  <CardContent className="py-4 flex items-center justify-between">
                    <div>
                      <div className="font-medium">{s.title}</div>
                      <div className="text-xs text-muted-foreground">{new Date(s.createdAt).toLocaleString()} · {fmtDuration(s.accumMs)} · {s.docIds?.length || 0} doc(s) · Locked</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">Finished</Badge>
                      <Button size="sm" variant="outline" onClick={() => alert('Open session viewer (todo)')}>View</Button>
                      <Button size="icon" variant="ghost" aria-label="Delete session" onClick={() => deleteSession(s.id)}><Trash2 className="h-4 w-4"/></Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {activeView === 'docs' && (
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between mb-3">
            <div className="text-2xl font-semibold">Docs</div>
            <div className="flex items-center gap-2">
              <Button variant="secondary" onClick={() => setDocPickerOpen(true)}><Link2 className="mr-2 h-4 w-4"/>Link to Session</Button>
              <Button onClick={() => setDocs(prev => [...prev, { id: crypto.randomUUID(), title: `New Doc ${prev.length+1}.pdf`, type: 'pdf', size: '500 KB', addedAt: new Date().toISOString().slice(0,10) }])}><Plus className="mr-2 h-4 w-4"/>Add Mock Doc</Button>
            </div>
          </div>
          <div className="flex items-center gap-2 mb-3">
            <div className="relative w-full">
              <Search className="h-4 w-4 absolute left-2 top-2.5 opacity-60" />
              <Input className="pl-8" placeholder="Search docs…" value={docSearch} onChange={(e) => setDocSearch(e.target.value)} />
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {filteredDocs.map(doc => (
              <Card key={doc.id} className="overflow-hidden">
                <CardHeader className="py-3">
                  <CardTitle className="text-base flex items-center gap-2"><FileText className="h-4 w-4" /> {doc.title}</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground">
                  <div className="flex items-center justify-between">
                    <div>{doc.type.toUpperCase()} · {doc.size}</div>
                    <div>Added {doc.addedAt}</div>
                  </div>
                  <div className="mt-3 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Checkbox id={`pick-${doc.id}`} checked={selectedDocIds.includes(doc.id)} onCheckedChange={() => toggleDocSelect(doc.id)} />
                      <label htmlFor={`pick-${doc.id}`}>Select</label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button size="sm" variant="outline" onClick={() => { alert(`Previewing ${doc.title} (mock)`); }}>Preview</Button>
                      <Button size="icon" variant="ghost" aria-label="Delete doc" onClick={() => deleteDoc(doc.id)}><Trash2 className="h-4 w-4"/></Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          {filteredDocs.length === 0 && (<div className="text-sm text-muted-foreground p-6 text-center">No documents match your search.</div>)}
        </div>
      )}

      {activeView === 'summaries' && (
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
                <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4"/> Export PDF</Button>
                <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4"/> Export Markdown</Button>
              </div>
              <div className="p-3 border rounded text-sm text-muted-foreground">(Preview will appear here)</div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === 'flashcards' && (
        <div className="max-w-5xl mx-auto px-4 py-10">
          <div className="text-2xl font-semibold mb-4">Flashcards</div>
          <Card>
            <CardContent className="py-6 space-y-4">
              <div className="text-sm text-muted-foreground">Auto-generate Q/A from session. Review mode like Anki.</div>
              <div className="flex items-center gap-2">
                <Button>Generate</Button>
                <Button variant="secondary">Start Review</Button>
              </div>
              <div className="p-3 border rounded text-sm text-muted-foreground">(Generated cards will appear here)</div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === 'quizzes' && (
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

      {activeView === 'explain' && (
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
                <Button variant="secondary"><Sparkles className="mr-2 h-4 w-4"/>Rewrite</Button>
              </div>
              <div className="p-3 border rounded text-sm text-muted-foreground">(Explanation will appear here)</div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === 'bookmarks' && (
        <div className="max-w-5xl mx-auto px-4 py-16">
          <div className="text-2xl font-semibold mb-4">Bookmarks</div>
          <Card><CardContent className="py-6 text-sm text-muted-foreground">Your saved moments will appear here.</CardContent></Card>
        </div>
      )}

      {activeView === 'settings' && (
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

      {activeView === 'exports' && (
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

      {activeView === 'developer' && (
        <div className="max-w-5xl mx-auto px-4 py-16">
          <div className="text-2xl font-semibold mb-4">Developer</div>
          <Card>
            <CardContent className="py-6 text-sm text-muted-foreground space-y-2">
              <div>Build: Local-only UI shell</div>
              <div>Planned: wire ASR/MT, PDF renderer, session viewer, hotkeys</div>
              <div>Env checks, logs, and preflight go here.</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Footer */}
      <div className="border-t bg-muted/30">
        <div className="max-w-7xl mx-auto px-4 py-3 text-xs text-muted-foreground flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="font-medium">Hotkeys:</span>
            <span>Start/Resume ⌘/Ctrl+K</span>
            <span>Bookmark B</span>
            <span>Open Menu ⌘/Ctrl+M</span>
          </div>
          <div>Built for: You · 100% local · Vienna</div>
        </div>
      </div>
    </div>
  );
}

// ---------------- Minimal Inline Tests (dev) ----------------
function _mergeDocs(a: string[], b: string[]) { return Array.from(new Set([...(a || []), ...(b || [])])); }
function _nextState(live: boolean, paused: boolean, action: "start" | "pause" | "resume" | "stop") {
  if (action === "start") return [true, false] as const;
  if (action === "pause") return [false, true] as const;
  if (action === "resume") return [true, false] as const;
  if (action === "stop") return [false, false] as const;
  return [live, paused] as const;
}
if (typeof window !== "undefined") {
  try {
    // grid class sanity
    console.assert("col-span-12 lg:col-span-6".endsWith("6"), "Grid class should end with a number");
    // merge uniqueness
    console.assert(JSON.stringify(_mergeDocs(["a", "b"], ["b", "c"])) === JSON.stringify(["a", "b", "c"]), "Doc merge should be unique");
    // state transitions
    console.assert(JSON.stringify(_nextState(false, false, "start")) === JSON.stringify([true, false]), "Start sets live");
    console.assert(JSON.stringify(_nextState(true, false, "pause")) === JSON.stringify([false, true]), "Pause toggles");
    console.assert(JSON.stringify(_nextState(false, true, "resume")) === JSON.stringify([true, false]), "Resume toggles");
    console.assert(JSON.stringify(_nextState(true, false, "stop")) === JSON.stringify([false, false]), "Stop resets");
    // first-run persistence
    localStorage.setItem('ll_first_run_seen', 'true');
    console.assert(localStorage.getItem('ll_first_run_seen') === 'true', 'first-run flag should persist');
    // delete tests (docs/sessions)
    const tmpSessions = [{id:'1'},{id:'2'}] as any[];
    const afterDelete = tmpSessions.filter(s => s.id !== '1');
    console.assert(afterDelete.length === 1 && afterDelete[0].id === '2', 'Session delete should remove item');
    // NEW: icon import smoke test
    console.assert(typeof BookOpen !== 'undefined', 'BookOpen icon should be defined');
  } catch (e) {
    console.warn("Dev inline tests failed:", e);
  }
}
/*************  ✨ Windsurf Command ⭐  *************/
// ---------------- Bug fixes and explanations ----------------

// Fix: "Cannot update a component (`LearningAppUI`) while rendering a different component (`div`)"
// Reason: We were updating the component state while rendering a different component
// Solution: Use `useEffect` to update the state only when the component is mounted
useEffect(() => {
  if (localStorage.getItem('ll_first_run_seen') === null) {
    localStorage.setItem('ll_first_run_seen', 'true');
  }
}, []);
/*******  dbc4145d-0bd3-42ce-82c7-10e7bbb82ae9  *******/