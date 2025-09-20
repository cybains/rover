"use client";
import React, { useEffect, useRef, useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Play, History, FileText, ListChecks, Sparkles, Bookmark,
  Settings, Download, Wrench, FolderDown, BookOpen, Search, Plus, Trash2, Link2
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

import AppHeader from "@/app/components/header";
import MainMenuSheet from "@/app/components/main-menu-sheet";
import FirstRunOverlay from "@/app/components/first-run-overlay";
import FooterHotkeys from "@/app/components/footer-hotkeys";
import SessionsListView from "@/app/components/sessions-list-view";
import DocsView from "@/app/components/docs-view";
import NewSessionDialog from "@/app/components/dialogs/new-session-dialog";
import DocPickerDialog from "@/app/components/dialogs/doc-picker-dialog";

import SessionView from "./components/session-view";
import type { DocMeta, SimpleSession } from "@/app/types";
import { normaliseDoc, normaliseSession } from "@/app/lib/normalize";
import { fileExtension, formatDuration as fmtDuration, formatFileSize, formatIsoDate } from "@/app/lib/format";

import {
  listDocuments, uploadDocument, deleteDocument as deleteDocumentApi,
  startSession as startSessionApi, listSessions as listSessionsApi,
  stopSession as stopSessionApi, linkSessionDocument, deleteSession as deleteSessionApi
} from "@/lib/api";

type ActiveView =
  | "session" | "sessions" | "docs" | "summaries" | "flashcards" | "quizzes"
  | "explain" | "bookmarks" | "settings" | "exports" | "developer";

export default function LearningAppUI() {
  // UI state
  const [latency, setLatency] = useState<number>(320);
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeView, setActiveView] = useState<ActiveView>("session");
  const [firstRun, setFirstRun] = useState(true);

  // Session state
  const [live, setLive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [session, setSession] = useState<SimpleSession | null>(null);
  const [sessions, setSessions] = useState<SimpleSession[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState<string | null>(null);

  // Docs state
  const [docs, setDocs] = useState<DocMeta[]>([]);
  const [docSearch, setDocSearch] = useState("");
  const [openDocId, setOpenDocId] = useState<string | null>(null);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docError, setDocError] = useState<string | null>(null);
  const [docNotice, setDocNotice] = useState<string | null>(null);
  const [docActionId, setDocActionId] = useState<string | null>(null);
  const [uploadingDoc, setUploadingDoc] = useState(false);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  // Dialogs
  const [sessionInitOpen, setSessionInitOpen] = useState(false);
  const [docPickerOpen, setDocPickerOpen] = useState(false);
  const [docPickerMode, setDocPickerMode] = useState<"new-session" | "link-to-current">("new-session");
  const [linkingDocs, setLinkingDocs] = useState(false);
  const [sessionBusy, setSessionBusy] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);

  // New session form (lives in parent to preserve your existing flow)
  const [sessionTitle, setSessionTitle] = useState("");
  const [sessionSubject, setSessionSubject] = useState("");
  const [sessionCourse, setSessionCourse] = useState("");
  const [sessionLanguage, setSessionLanguage] = useState("");
  const [sessionTags, setSessionTags] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Latency ticker (visual only)
  useEffect(() => {
    const id = setInterval(
      () => setLatency((l) => Math.max(120, Math.min(1200, Math.round(l + (Math.random() * 80 - 40))))),
      1200
    );
    return () => clearInterval(id);
  }, []);

  // Load sessions
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setSessionsLoading(true);
      setSessionsError(null);
      try {
        const payload = await listSessionsApi();
        if (cancelled) return;
        const items = Array.isArray(payload) ? payload.map(normaliseSession) : [];
        setSessions(items);
         catch (err) {
        console.error(err);
        if (!cancelled) setSessionsError("Unable to load sessions. Is the backend running?");
      } finally {
        if (!cancelled) setSessionsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Load docs
  useEffect(() => {
    let cancelled = false;
    (async () => {
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
        if (!cancelled) setDocError("Unable to load documents. Is the backend running?");
      } finally {
        if (!cancelled) setDocsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Session elapsed timer (simple createdAt-based as in your original)
  const sessionCreatedAt = session?.createdAt;
  useEffect(() => {
    if (!sessionCreatedAt || !live) return;
    if (Number.isNaN(new Date(sessionCreatedAt).getTime())) return;
    const update = () => {
      setSession((prev) => {
        if (!prev) return prev;
        const base = new Date(prev.createdAt).getTime();
        if (Number.isNaN(base)) return prev;
        return { ...prev, accumMs: Date.now() - base };
      });
    };
    update();
    const id = window.setInterval(update, 1000);
    return () => window.clearInterval(id);
  }, [live, sessionCreatedAt]);

  // Helpers
  const toggleDocSelect = (id: string) => {
    setSelectedDocIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
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
      event.target.value = "";
    }
  };

  // Session actions
  const onClickStart = () => {
    if (!session || session.status !== "live") {
      setSessionError(null);
      setDocPickerMode("new-session");
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

  const onClickStop = async () => {
    if (!session) return;
    try {
      await stopSessionApi(session.id);
    } catch (err) {
      console.error(err);
    }
    setSessions((prev) =>
      prev.map((s) =>
        s.id === session.id ? { ...s, status: "stopped", updatedAt: new Date().toISOString() } : s
      )
    );
    setSession(null);
    setLive(false);
    setPaused(false);
    setOpenDocId(null);
  };

  const handleCreateSession = async () => {
    const title = sessionTitle.trim();
    if (!title) {
      setSessionError("Please enter a session title.");
      return;
    }
    setSessionBusy(true);
    setSessionError(null);
    try {
      const response = await startSessionApi({ title, docIds: selectedDocIds });
      const base = normaliseSession(response);
      const metadata = {
        subject: sessionSubject.trim() || undefined,
        course: sessionCourse.trim() || undefined,
        language: sessionLanguage.trim() || undefined,
        tags: sessionTags.trim() || undefined,
      };
      const hasMeta = Object.values(metadata).some(Boolean);
      const nextSession: SimpleSession = { ...base, metadata: hasMeta ? metadata : undefined, accumMs: 0 };

      setSession(nextSession);
      setSessions((prev) => [nextSession, ...prev.filter((s) => s.id !== nextSession.id)]);
      setLive(true);
      setPaused(false);
      setFirstRun(false);
      setSessionInitOpen(false);

      // reset form + selection
      setSessionTitle(""); setSessionSubject(""); setSessionCourse(""); setSessionLanguage(""); setSessionTags("");
      setSelectedDocIds([]);

      const docIds = nextSession.docIds;
      setOpenDocId(docIds[docIds.length - 1] ?? null);

      // optionally hydrate docs returned from backend
      if (response && Array.isArray((response as any).documents)) {
        const docsFromResponse = (response as any).documents.map(normaliseDoc);
        setDocs((prev) => {
          const next = [...prev];
          for (const doc of docsFromResponse) {
            const idx = next.findIndex((d) => d.id === doc.id);
            if (idx >= 0) next[idx] = doc;
            else next.unshift(doc);
          }
          return next;
        });
      }
    } catch (err) {
      console.error(err);
      setSessionError("Unable to start session. Please ensure the backend is running.");
    } finally {
      setSessionBusy(false);
    }
  };

  const linkSelectedToSession = async (ids: string[]) => {
    if (!session) return;
    const toLink = ids.filter((id) => !session.docIds.includes(id));
    if (toLink.length === 0) return;

    setLinkingDocs(true);
    setDocError(null);
    try {
      await Promise.all(toLink.map((id) => linkSessionDocument(session.id, id)));
      setSession((prev) => {
        if (!prev) return prev;
        const updatedDocIds = Array.from(new Set([...prev.docIds, ...toLink]));
        return { ...prev, docIds: updatedDocIds };
      });
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== session.id) return s;
          const updatedDocIds = Array.from(new Set([...s.docIds, ...toLink]));
          const docDelta = updatedDocIds.length - s.docIds.length;
          return {
            ...s,
            docIds: updatedDocIds,
            documentsCount:
              typeof s.documentsCount === "number"
                ? s.documentsCount + docDelta
                : updatedDocIds.length || undefined,
          };
        })
      );
      setDocNotice(`${toLink.length} document${toLink.length === 1 ? "" : "s"} linked`);
      setOpenDocId(toLink[toLink.length - 1] ?? null);
    } catch (err) {
      console.error(err);
      setDocError("Failed to link selected documents. Check backend logs for details.");
    } finally {
      setLinkingDocs(false);
    }
  };

  const deleteSession = async (id: string) => {
    try {
      await deleteSessionApi(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (session?.id === id) {
        setSession(null); setLive(false); setPaused(false); setOpenDocId(null);
      }
    } catch (err) {
      console.error(err);
      setSessionsError("Failed to delete session. Check backend logs for details.");
    }
  };

  const deleteDoc = async (id: string) => {
    setDocError(null); setDocNotice(null); setDocActionId(id);
    try {
      await deleteDocumentApi(id);
      setDocs((prev) => prev.filter((d) => d.id !== id));
      setSelectedDocIds((prev) => prev.filter((x) => x !== id));
      setSession((s) => (s ? { ...s, docIds: (s.docIds || []).filter((x) => x !== id) } : s));
      if (openDocId === id) setOpenDocId(null);
      setDocNotice("Document deleted");
    } catch (err) {
      console.error(err);
      setDocError("Failed to delete document. Check backend logs for details.");
    } finally {
      setDocActionId(null);
    }
  };

  // Derived list for DocsView search
  const filteredDocs = docs.filter((d) => {
    const term = docSearch.trim().toLowerCase();
    if (!term) return true;
    return d.filename.toLowerCase().includes(term) || d.tags.some((t) => t.toLowerCase().includes(term));
  });

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* First run overlay */}
      {firstRun && (
        <FirstRunOverlay
          onSelect={(v) => { setActiveView(v); setFirstRun(false); }}
          onClose={() => setFirstRun(false)}
        />
      )}

      {/* Header */}
      <AppHeader
        latency={latency}
        live={live}
        session={session}
        docs={docs}
        onOpenMenu={() => setMenuOpen(true)}
        onOpenDoc={(id) => setOpenDocId(id)}
        version="v0.6 UI"
      />

      {/* Main menu */}
      <MainMenuSheet
        open={menuOpen}
        onOpenChange={setMenuOpen}
        activeView={activeView}
        onSelect={(value) => { setActiveView(value); setMenuOpen(false); setFirstRun(false); }}
      />

      {/* New Session dialog */}
      <NewSessionDialog
        open={sessionInitOpen}
        busy={sessionBusy}
        selectedDocIds={selectedDocIds}
        onOpenChange={(open) => {
          setSessionInitOpen(open);
          if (!open) { setSessionError(null); setSessionBusy(false); }
        }}
        onCreate={() => handleCreateSession()}
        onOpenDocPicker={() => { setDocPickerMode("new-session"); setDocPickerOpen(true); }}
        error={sessionError}
      />

      {/* Doc Picker dialog (mode-aware) */}
      <DocPickerDialog
        open={docPickerOpen}
        mode={docPickerMode}
        docs={docs}
        initiallySelectedIds={selectedDocIds}
        uploading={uploadingDoc}
        linking={linkingDocs}
        onOpenChange={(open) => {
          setDocPickerOpen(open);
          if (!open) setLinkingDocs(false);
        }}
        onAddFileClick={handleAddFileClick}
        onConfirm={(ids) => {
          setSelectedDocIds(ids);
          if (docPickerMode === "link-to-current") {
            void linkSelectedToSession(ids);
          }
          setDocPickerOpen(false);
        }}
      />

      {/* Hidden file input for uploads */}
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleFileSelected}
        // optional: accept common study file types
        // accept=".pdf,.txt,.md,.doc,.docx,.ppt,.pptx"
      />

      {/* Views */}
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
          onLinkDocs={() => { setDocPickerMode(session ? "link-to-current" : "new-session"); setDocPickerOpen(true); }}
          formatDuration={fmtDuration}
          formatFileSize={formatFileSize}
          formatIsoDate={formatIsoDate}
          fileExtension={fileExtension}
        />
      )}

      {activeView === "sessions" && (
        <SessionsListView
          sessions={sessions}
          loading={sessionsLoading}
          error={sessionsError}
          onDelete={(id) => void deleteSession(id)}
          onOpen={(id) => {
            // (future) navigate or load by id; for now just focus the session if it matches
            const found = sessions.find((s) => s.id === id);
            if (found) { setSession(found); setActiveView("session"); }
          }}
        />
      )}

      {activeView === "docs" && (
        <DocsView
          docs={filteredDocs}
          search={docSearch}
          onSearchChange={setDocSearch}
          selectedDocIds={selectedDocIds}
          onToggleSelect={toggleDocSelect}
          uploading={uploadingDoc}
          notice={docNotice}
          error={docError}
          onUploadClick={handleAddFileClick}
          onPreview={(id) => setOpenDocId(id)}
          onDelete={(id) => void deleteDoc(id)}
          onLinkToSession={() => { setDocPickerMode(session ? "link-to-current" : "new-session"); setDocPickerOpen(true); }}
        />
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

      {/* Footer */}
      <FooterHotkeys />
    </div>
  );
}
