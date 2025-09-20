"use client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, History, FileText, ListChecks, BookOpen } from "lucide-react";

type ActiveView = "session" | "sessions" | "docs" | "summaries" | "flashcards" | "quizzes";
type Props = { onSelect: (v: ActiveView) => void; onClose: () => void; version?: string; };

export default function FirstRunOverlay({ onSelect, onClose, version = "v0.6 UI" }: Props) {
  return (
    <div className="fixed inset-0 z-40 grid place-items-center bg-gradient-to-b from-background to-muted">
      <div className="w-full max-w-3xl text-center p-8">
        <div className="mx-auto h-14 w-14 rounded-2xl bg-foreground text-background grid place-items-center text-2xl font-bold">Λ</div>
        <div className="mt-3 text-3xl font-semibold">Lab</div>
        <div className="text-sm text-muted-foreground mb-6">Personal build <Badge variant="outline" className="ml-2">{version}</Badge></div>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <Button className="h-20" onClick={() => onSelect("session")}><Play className="mr-2"/>Start a Session</Button>
          <Button variant="secondary" className="h-20" onClick={() => onSelect("sessions")}><History className="mr-2"/>Sessions</Button>
          <Button variant="secondary" className="h-20" onClick={() => onSelect("docs")}><FileText className="mr-2"/>Docs</Button>
          <Button variant="outline" className="h-20" onClick={() => onSelect("summaries")}><FileText className="mr-2"/>Summaries</Button>
          <Button variant="outline" className="h-20" onClick={() => onSelect("flashcards")}><BookOpen className="mr-2"/>Flashcards</Button>
          <Button variant="outline" className="h-20" onClick={() => onSelect("quizzes")}><ListChecks className="mr-2"/>Quizzes</Button>
        </div>

        <Button variant="ghost" className="mt-6" onClick={onClose}>Enter app</Button>
      </div>
    </div>
  );
}
