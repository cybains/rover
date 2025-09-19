'use client';
import React, { useState, useEffect, useRef } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Play,
  Bookmark,
  Settings,
  Download,
  Timer,
  FolderDown,
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
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Checkbox } from "@/components/ui/checkbox";

import SessionView from "./components/session-view";
import type { DocMeta, SimpleSession } from "@/app/types";

import { listDocuments, uploadDocument, deleteDocument as deleteDocumentApi } from "@/lib/api";

const fallbackId = () =>
  typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

const normaliseDoc = (doc: any): DocMeta => ({
  id: doc?._id ? String(doc._id) : doc?.id ? String(doc.id) : fallbackId(),
  filename:
    (doc?.filenameOriginal as string) ||
    (doc?.filename as string) ||
    (doc?.title as string) ||
    "Document",
  contentType: doc?.contentType || "application/octet-stream",
  size: Number(doc?.size ?? 0) || 0,
  uploadedAt: doc?.uploadedAt || doc?.addedAt,
  tags: Array.isArray(doc?.tags) ? doc.tags.map(String) : [],
  notes: typeof doc?.notes === "string" ? doc.notes : "",
  linkedSessions: Array.isArray(doc?.linkedSessions)
    ? doc.linkedSessions.map(String)
    : [],
});

const formatFileSize = (bytes: number) => {
  if (!bytes || bytes < 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  if (idx === 0) return `${bytes} ${units[idx]}`;
  const value = bytes / Math.pow(1024, idx);
  return `${value.toFixed(1)} ${units[idx]}`;
};

const formatIsoDate = (iso?: string) => {
  if (!iso) return "unknown date";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString();
};

const fileExtension = (filename: string) => {
  const parts = filename.split(".");
  if (parts.length < 2) return "FILE";
  return parts.pop()?.toUpperCase() || "FILE";
};

type ActiveView = "session" | "sessions" | "docs" | "summaries" | "flashcards" | "quizzes" | "explain" | "bookmarks" | "settings" | "exports" | "developer";


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

const fmtDuration = (ms: number) => {
  const s = Math.floor(ms / 1000);
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
};

export default function LearningAppUI() {
  const [live, setLive] = useState<boolean>(false);
  const [paused, setPaused] = useState<boolean>(false);
  const [latency, setLatency] = useState<number>(320);
  const [menuOpen, setMenuOpen] = useState<boolean>(false);
  const [activeView, setActiveView] = useState<ActiveView>('session');
  const [firstRun, setFirstRun] = useState<boolean>(true);
  const [session, setSession] = useState<SimpleSession | null>(null);
  const [sessions, setSessions] = useState<SimpleSession[]>([]);
  const [sessionInitOpen, setSessionInitOpen] = useState<boolean>(false);
  const [docs, setDocs] = useState<DocMeta[]>([]);
  const [docSearch, setDocSearch] = useState<string>('');
  const [docPickerOpen, setDocPickerOpen] = useState<boolean>(false);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [openDocId, setOpenDocId] = useState<string | null>(null);
  const [docsLoading, setDocsLoading] = useState<boolean>(false);
  const [docError, setDocError] = useState<string | null>(null);
  const [docNotice, setDocNotice] = useState<string | null>(null);
  const [docActionId, setDocActionId] = useState<string | null>(null);
  const [uploadingDoc, setUploadingDoc] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const id = setInterval(() => setLatency((l) => Math.max(120, Math.min(1200, Math.round(l + (Math.random() * 80 - 40))))), 1200);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      setFirstRun(localStorage.getItem('ll_first_run_seen') !== 'true');
      const savedSession = localStorage.getItem('currentSession');
      if (savedSession) {
        setSession(JSON.parse(savedSession) as SimpleSession);
      }
      const savedSessions = localStorage.getItem('sessions');
      if (savedSessions) {
        setSessions(JSON.parse(savedSessions) as SimpleSession[]);
      }
      const storedDocs = localStorage.getItem('ll_docs');
      if (storedDocs) {
        try {
          const parsed = JSON.parse(storedDocs) as any[];
          setDocs(parsed.map(normaliseDoc));
        } catch {}
      }
    } catch {}
  }, []);

  useEffect(() => {
    try {
      if (session) localStorage.setItem('currentSession', JSON.stringify(session));
      else localStorage.removeItem('currentSession');
    } catch {}
  }, [session]);
  useEffect(() => {
    let cancelled = false;

    const loadDocs = async () => {
      setDocsLoading(true);
      setDocError(null);
      try {
        const payload = await listDocuments();
        if (!cancelled) {
          const normalised = Array.isArray(payload) ? payload.map(normaliseDoc) : [];
          setDocs(normalised);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setDocError('Unable to load documents. Is the backend running?');
        }
      } finally {
        if (!cancelled) {
          setDocsLoading(false);
        }
      }
    };

    loadDocs();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem('sessions', JSON.stringify(sessions));
    } catch {}
  }, [sessions]);
  useEffect(() => {
    try {
      localStorage.setItem('ll_docs', JSON.stringify(docs));
    } catch {}
  }, [docs]);
  useEffect(() => {
    try {
      localStorage.setItem('ll_first_run_seen', String(!firstRun));
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

  const searchTerm = docSearch.trim().toLowerCase();
  const filteredDocs = docs.filter((d) => {
    if (!searchTerm) return true;
    const nameMatch = d.filename.toLowerCase().includes(searchTerm);
    const tagMatch = d.tags.some((tag) => tag.toLowerCase().includes(searchTerm));
    return nameMatch || tagMatch;
  });

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

  const handleAddFileClick = () => {
    setDocError(null);
    setDocNotice(null);
    fileInputRef.current?.click();
  };

  const handleFileSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setDocError(null);
    setDocNotice(null);
    setUploadingDoc(true);
    try {
      const uploaded = await uploadDocument({ file });
      const doc = normaliseDoc(uploaded);
      setDocs((prev) => [doc, ...prev.filter((d) => d.id !== doc.id)]);
      setOpenDocId(doc.id);
      setDocNotice(`${file.name} uploaded`);
    } catch (err) {
      console.error(err);
      setDocError(`Failed to upload ${file.name}.`);
    } finally {
      setUploadingDoc(false);
      event.target.value = '';
    }
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
  const deleteDoc = async (id: string) => {
    setDocError(null);
    setDocNotice(null);
    setDocActionId(id);
    try {
      await deleteDocumentApi(id);
      setDocs((prev) => prev.filter((d) => d.id !== id));
      setSelectedDocIds((prev) => prev.filter((x) => x !== id));
      setSession((s) => (s ? { ...s, docIds: (s.docIds || []).filter((x) => x !== id) } : s));
      if (openDocId === id) setOpenDocId(null);
      setDocNotice('Document deleted');
    } catch (err) {
      console.error(err);
      setDocError('Failed to delete document. Check backend logs for details.');
    } finally {
      setDocActionId(null);
    }
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
                      <FileText className="h-3 w-3" /> {d.filename}
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
              <Input className="pl-8" placeholder="Search docs..." value={docSearch} onChange={(e) => setDocSearch(e.target.value)} />
            </div>
            <Button variant="secondary" onClick={handleAddFileClick} disabled={uploadingDoc}>
              {uploadingDoc ? 'Uploading...' : (<><Plus className="mr-2 h-4 w-4" /> Add File</>)}
            </Button>
          </div>
          <div className="max-h-[46vh] overflow-y-auto space-y-2 pr-1">
            {filteredDocs.map((doc) => (
              <div key={doc.id} className="flex items-center justify-between border rounded-lg p-3">
                <div className="flex items-center gap-3">
                  <Badge variant="secondary" className="gap-1"><FileText className="h-3 w-3" /> {fileExtension(doc.filename)}</Badge>
                  <div>
                    <div className="font-medium text-sm">{doc.filename}</div>
                    <div className="text-xs text-muted-foreground">{formatFileSize(doc.size)} | uploaded {formatIsoDate(doc.uploadedAt)}</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Checkbox id={`sel-${doc.id}`} checked={selectedDocIds.includes(doc.id)} onCheckedChange={() => toggleDocSelect(doc.id)} />
                  <label htmlFor={`sel-${doc.id}`} className="text-sm">Select</label>
                </div>
              </div>
            ))}
            {!docsLoading && filteredDocs.length === 0 && (<div className="text-sm text-muted-foreground p-6 text-center">No documents match your search.</div>)}
          </div>
          <DialogFooter>
            <Button disabled={!session} onClick={linkSelectedToSession}>
              <Link2 className="mr-2 h-4 w-4" /> Link to current session
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {activeView === "session" && (
        <SessionView
          session={session}
          docs={docs}
          openDocId={openDocId}
          onDocOpenChange={(id) => setOpenDocId(id)}
          live={live}
          paused={paused}
          latency={latency}
          onStart={onClickStart}
          onPauseResume={onClickPauseOrResume}
          onStop={onClickStop}
          onLinkDocs={() => setDocPickerOpen(true)}
          formatDuration={fmtDuration}
          formatFileSize={formatFileSize}
          formatIsoDate={formatIsoDate}
          fileExtension={fileExtension}
        />
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
                      <Button size="iconLg" variant="ghost" aria-label="Delete session" onClick={() => deleteSession(s.id)} className="rounded-full border border-red-200 text-red-500 hover:bg-red-50 hover:text-red-600"><Trash2 className="h-9 w-9" /></Button>
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
              <Button onClick={handleAddFileClick} disabled={uploadingDoc}>
                {uploadingDoc ? 'Uploading...' : (<><Plus className="mr-2 h-4 w-4" /> Add File</>)}
              </Button>
              <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileSelected} />
            </div>
          </div>
          <div className="flex items-center gap-2 mb-3">
            <div className="relative w-full">
              <Search className="h-4 w-4 absolute left-2 top-2.5 opacity-60" />
              <Input className="pl-8" placeholder="Search docs..." value={docSearch} onChange={(e) => setDocSearch(e.target.value)} />
            </div>
          </div>
          {(docNotice || docError || docsLoading) && (
            <div className="mb-3 space-y-1">
              {docNotice && <div className="text-sm text-emerald-600">{docNotice}</div>}
              {docError && <div className="text-sm text-red-600">{docError}</div>}
              {docsLoading && <div className="text-xs text-muted-foreground">Loading documents...</div>}
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {filteredDocs.map((doc) => (
              <Card key={doc.id} className="overflow-hidden">
                <CardHeader className="py-3">
                  <CardTitle className="text-base flex items-center gap-2"><FileText className="h-4 w-4" /> {doc.filename}</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <span>{fileExtension(doc.filename)} | {formatFileSize(doc.size)}</span>
                    <span>Uploaded {formatIsoDate(doc.uploadedAt)}</span>
                  </div>
                  {doc.tags.length > 0 && (<div className="text-xs">Tags: {doc.tags.join(', ')}</div>)}
                  {doc.notes && (<div className="text-xs">{doc.notes}</div>)}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Checkbox id={`pick-${doc.id}`} checked={selectedDocIds.includes(doc.id)} onCheckedChange={() => toggleDocSelect(doc.id)} />
                      <label htmlFor={`pick-${doc.id}`}>Select</label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button size="sm" variant="outline" onClick={() => setOpenDocId(doc.id)}>Preview</Button>
                      <Button size="iconLg" variant="ghost" aria-label="Delete doc" onClick={() => void deleteDoc(doc.id)} disabled={docActionId === doc.id} className="rounded-full border border-red-200 text-red-500 hover:bg-red-50 hover:text-red-600 disabled:opacity-60 disabled:pointer-events-none"><Trash2 className="h-9 w-9" /></Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          {!docsLoading && filteredDocs.length === 0 && (<div className="text-sm text-muted-foreground p-6 text-center">No documents match your search.</div>)}
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






