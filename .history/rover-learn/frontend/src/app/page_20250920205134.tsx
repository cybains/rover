"use client";
import React, { useEffect, useRef, useState } from "react";

import AppHeader from "@/app/components/header";
import MainMenuSheet from "@/app/components/main-menu-sheet";
import FirstRunOverlay from "@/app/components/first-run-overlay";
import FooterHotkeys from "@/app/components/footer-hotkeys";
import SessionsListView from "@/app/components/sessions-list-view";
import DocsView from "@/app/components/docs-view";
import NewSessionDialog from "@/app/components/dialogs/new-session-dialog";
import DocPickerDialog from "@/app/components/dialogs/doc-picker-dialog";

import SummariesView from "@/app/components/views/summaries-view";
import FlashcardsView from "@/app/components/views/flashcards-view";
import QuizzesView from "@/app/components/views/quizzes-view";
import ExplainView from "@/app/components/views/explain-view";
import BookmarksView from "@/app/components/views/bookmarks-view";
import SettingsView from "@/app/components/views/settings-view";
import ExportsView from "@/app/components/views/exports-view";

import SessionView from "./components/session-view";
import type { DocMeta, SimpleSession } from "@/app/types";
import { normaliseDoc, normaliseSession } from "@/app/lib/normalize";
import {
  fileExtension,
  formatDuration as fmtDuration,
  formatFileSize,
  formatIsoDate,
} from "@/app/lib/format";

import {
  listDocuments,
  uploadDocument,
  deleteDocument as deleteDocumentApi,
  startSession as startSessionApi,
  listSessions as listSessionsApi,
  stopSession as stopSessionApi,
  linkSessionDocument,
  deleteSession as deleteSessionApi,
} from "@/lib/api";

// Local type so there’s no cross-file type import to break
type NewSessionPayload = {
  title: string;
  subject?: string;
  course?: string;
  language?: string;
  tags?: string;
  selectedDocIds: string[];
};

type ActiveView =
  | "session"
  | "sessions"
  | "docs"
  | "summaries"
  | "flashcards"
  | "quizzes"
  | "explain"
  | "bookmarks"
  | "settings"
  | "exports"
  | "developer";

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
  const [uploadingDoc, setUploadingDoc] = useState(false);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  // Dialogs
  const [sessionInitOpen, setSessionInitOpen] = useState(false);
  const [docPickerOpen, setDocPickerOpen] = useState(false);
  const [docPickerMode, setDocPickerMode] =
    useState<"new-session" | "link-to-current">("new-session");
  const [linkingDocs, setLinkingDocs] = useState(false);
  const [sessionBusy, setSessionBusy] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Latency ticker (visual)
  useEffect(() => {
    const id = setInterval(
      () =>
        setLatency((l) =>
          Math.max(120, Math.min(1200, Math.round(l + (Math.random() * 80 - 40))))
        ),
      1200
    );
    return () => clearInterval(id);
  }, []);

  // Load sessions — IMPORTANT: do NOT auto-close the overlay
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
        // intentionally keep firstRun=true; overlay closes only by user action
      } catch (err) {
        console.error(err);
        if (!cancelled)
          setSessionsError("Unable to load sessions. Is the backend running?");
      } finally {
        if (!cancelled) setSessionsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Load documents
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setDocsLoading(true);
      setDocError(null);
      try {
        const payload = await listDocuments();
        if (!cancelled) {
          const normalised = Array.isArray(payload)
            ? payload.map(normaliseDoc)
            : [];
          setDocs(normalised);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled)
          setDocError("Unable to load documents. Is the backend running?");
      } finally {
        if (!cancelled) setDocsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Session elapsed timer
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
    setSelectedDocIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const handleAddFileClick = () => {
    setDocError(null);
    setDocNotice(null);
    fileInputRef.current?.click();
  };

  const handleFileSelected = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
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
        s.id === session.id
          ? { ...s, status: "stopped", updatedAt: new Date().toISOString() }
          : s
      )
    );
    setSession(null);
    setLive(false);
    setPaused(false);
    setOpenDocId(null);
  };

  // Create session — uses payload from dialog + blocks duplicate titles
  const handleCreateSession = async (p: NewSessionPayload) => {
    const title = (p.title || "").trim();

    if (!title) {
      setSessionError("Please enter a session title.");
      return;
    }

    const exists = sessions.some(
      (s) => s.title.trim().toLowerCase() === title.toLowerCase()
    );
    if (exists) {
      setSessionError(`A session named “${title}” already exists.`);
      return;
    }

    setSessionBusy(true);
    setSessionError(null);
    try {
      const response = await startSessionApi({
        title,
        docIds: p.selectedDocIds,
      });
      const base = normaliseSession(response);
      const metadata = {
        subject: p.subject?.trim() || undefined,
        course: p.course?.trim() || undefined,
        language: p.language?.trim() || undefined,
        tags: p.tags?.trim() || undefined,
      };
      const hasMeta = Object.values(metadata).some(Boolean);

      const nextSession: SimpleSession = {
        ...base,
        metadata: hasMeta ? metadata : undefined,
        accumMs: 0,
      };

      setSession(nextSession);
      setSessions((prev) => [
        nextSession,
        ...prev.filter((s) => s.id !== nextSession.id),
      ]);
      setLive(true);
      setPaused(false);
      setFirstRun(false);
      setSessionInitOpen(false);

      // reflect latest selection
      setSelectedDocIds([]);

      const docIds = nextSession.docIds;
      setOpenDocId(docIds[docIds.length - 1] ?? null);

      // hydrate docs from backend if provided
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
      setSessionError(
        "Unable to start session. Please ensure the backend is running."
      );
    } finally {
      setSessionBusy(false);
    }
  };

  // Link docs to current session
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
      setDocNotice(
        `${toLink.length} document${toLink.length === 1 ? "" : "s"} linked`
      );
      setOpenDocId(toLink[toLink.length - 1] ?? null);
    } catch (err) {
      console.error(err);
      setDocError("Failed to link selected documents. Check backend logs for details.");
    } finally {
      setLinkingDocs(false);
    }
  };

  // Delete actions
  const deleteSession = async (id: string) => {
    try {
      await deleteSessionApi(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (session?.id === id) {
        setSession(null);
        setLive(false);
        setPaused(false);
        setOpenDocId(null);
      }
    } catch (err) {
      console.error(err);
      setSessionsError("Failed to delete session. Check backend logs for details.");
    }
  };

  const deleteDoc = async (id: string) => {
    setDocError(null);
    setDocNotice(null);
    try {
      await deleteDocumentApi(id);
      setDocs((prev) => prev.filter((d) => d.id !== id));
      setSelectedDocIds((prev) => prev.filter((x) => x !== id));
      setSession((s) =>
        s ? { ...s, docIds: (s.docIds || []).filter((x) => x !== id) } : s
      );
      if (openDocId === id) setOpenDocId(null);
      setDocNotice("Document deleted");
    } catch (err) {
      console.error(err);
      setDocError("Failed to delete document. Check backend logs for details.");
    }
  };

  // Derived list for DocsView search
  const filteredDocs = docs.filter((d) => {
    const term = docSearch.trim().toLowerCase();
    if (!term) return true;
    return (
      d.filename.toLowerCase().includes(term) ||
      d.tags.some((t) => t.toLowerCase().includes(term))
    );
  });

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* First-run overlay — stays until user chooses */}
      {firstRun && (
        <FirstRunOverlay
          onSelect={(v) => {
            setActiveView(v);
            setFirstRun(false);
          }}
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
        onSelect={(value) => {
          setActiveView(value);
          setMenuOpen(false);
          setFirstRun(false);
        }}
      />

      {/* New Session dialog */}
      <NewSessionDialog
        open={sessionInitOpen}
        busy={sessionBusy}
        selectedDocIds={selectedDocIds}
        onOpenChange={(open) => {
          setSessionInitOpen(open);
          if (!open) {
            setSessionError(null);
            setSessionBusy(false);
          }
        }}
        onCreate={(payload) => handleCreateSession(payload)}
        onOpenDocPicker={() => {
          setDocPickerMode("new-session");
          setDocPickerOpen(true);
        }}
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
          onLinkDocs={() => {
            setDocPickerMode(session ? "link-to-current" : "new-session");
            setDocPickerOpen(true);
          }}
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
            const found = sessions.find((s) => s.id === id);
            if (found) {
              setSession(found);
              setActiveView("session");
            }
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
          onLinkToSession={() => {
            setDocPickerMode(session ? "link-to-current" : "new-session");
            setDocPickerOpen(true);
          }}
        />
      )}

      {activeView === "summaries" && <SummariesView />}
      {activeView === "flashcards" && <FlashcardsView />}
      {activeView === "quizzes" && <QuizzesView />}
      {activeView === "explain" && <ExplainView />}
      {activeView === "bookmarks" && <BookmarksView />}
      {activeView === "settings" && <SettingsView />}
      {activeView === "exports" && <ExportsView />}

      {/* Footer */}
      <FooterHotkeys />
    </div>
  );
}
