"use client";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import {
  Play, History, FileText, Sparkles, BookOpen, ListChecks,
  Bookmark, Settings, Download, Wrench
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

type ActiveView =
  | "session" | "sessions" | "docs" | "summaries" | "flashcards" | "quizzes"
  | "explain" | "bookmarks" | "settings" | "exports" | "developer";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  activeView: ActiveView;
  onSelect: (v: ActiveView) => void;
};

const MenuButton = ({ label, icon: Icon, value, active, onClick }:{
  label: string; icon: LucideIcon; value: ActiveView; active: boolean; onClick: () => void;
}) => (
  <Button
    variant={active ? "secondary" : "ghost"}
    className="w-full justify-start"
    onClick={onClick}
  >
    <Icon className="mr-2 h-4 w-4" /> {label}
  </Button>
);

export default function MainMenuSheet({ open, onOpenChange, activeView, onSelect }: Props) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="left" className="w-[300px]">
        <SheetHeader>
          <SheetTitle>Main Menu</SheetTitle>
        </SheetHeader>
        <div className="mt-4 space-y-2">
          <MenuButton label="Start a Session" icon={Play} value="session" active={activeView==="session"} onClick={() => onSelect("session")} />
          <MenuButton label="Sessions" icon={History} value="sessions" active={activeView==="sessions"} onClick={() => onSelect("sessions")} />
          <MenuButton label="Docs" icon={FileText} value="docs" active={activeView==="docs"} onClick={() => onSelect("docs")} />
          <MenuButton label="Summaries" icon={FileText} value="summaries" active={activeView==="summaries"} onClick={() => onSelect("summaries")} />
          <MenuButton label="Flashcards" icon={BookOpen} value="flashcards" active={activeView==="flashcards"} onClick={() => onSelect("flashcards")} />
          <MenuButton label="Quizzes" icon={ListChecks} value="quizzes" active={activeView==="quizzes"} onClick={() => onSelect("quizzes")} />
          <MenuButton label="Explain" icon={Sparkles} value="explain" active={activeView==="explain"} onClick={() => onSelect("explain")} />
          <MenuButton label="Bookmarks" icon={Bookmark} value="bookmarks" active={activeView==="bookmarks"} onClick={() => onSelect("bookmarks")} />
          <MenuButton label="Settings" icon={Settings} value="settings" active={activeView==="settings"} onClick={() => onSelect("settings")} />
          <MenuButton label="Exports" icon={Download} value="exports" active={activeView==="exports"} onClick={() => onSelect("exports")} />
          <MenuButton label="Developer" icon={Wrench} value="developer" active={activeView==="developer"} onClick={() => onSelect("developer")} />
        </div>
      </SheetContent>
    </Sheet>
  );
}
