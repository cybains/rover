"use client";
import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Link2 } from "lucide-react";

type NewSessionPayload = {
  title: string;
  subject?: string;
  course?: string;
  language?: string;
  tags?: string;
  selectedDocIds: string[];
};

type Props = {
  open: boolean;
  busy?: boolean;
  selectedDocIds: string[];
  onOpenChange: (open: boolean) => void;
  onCreate: (payload: NewSessionPayload) => void;
  onOpenDocPicker: () => void; // opens in "new-session" mode
  error?: string | null;
};

export default function NewSessionDialog({
  open, busy, selectedDocIds, onOpenChange, onCreate, onOpenDocPicker, error
}: Props) {
  const [title, setTitle] = useState("");
  const [subject, setSubject] = useState("");
  const [course, setCourse] = useState("");
  const [language, setLanguage] = useState("");
  const [tags, setTags] = useState("");

  useEffect(() => {
    if (!open) {
      setTitle(""); setSubject(""); setCourse(""); setLanguage(""); setTags("");
    }
  }, [open]);

  const submit = () => {
    if (!title.trim()) return;
    onCreate({
      title: title.trim(),
      subject: subject.trim() || undefined,
      course: course.trim() || undefined,
      language: language.trim() || undefined,
      tags: tags.trim() || undefined,
      selectedDocIds,
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>New Session</DialogTitle>
        </DialogHeader>

        <div className="grid gap-3 py-2">
          <div>
            <div className="text-sm font-medium mb-1">Session Title *</div>
            <Input
              placeholder="e.g., Linear Regression - Week 3"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              autoFocus
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-sm font-medium mb-1">Subject</div>
              <Input placeholder="e.g., Statistics" value={subject} onChange={(e) => setSubject(e.target.value)} />
            </div>
            <div>
              <div className="text-sm font-medium mb-1">Course</div>
              <Input placeholder="e.g., ML101" value={course} onChange={(e) => setCourse(e.target.value)} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-sm font-medium mb-1">Language</div>
              <Input placeholder="e.g., German" value={language} onChange={(e) => setLanguage(e.target.value)} />
            </div>
            <div>
              <div className="text-sm font-medium mb-1">Tags</div>
              <Input placeholder="comma, separated, tags" value={tags} onChange={(e) => setTags(e.target.value)} />
            </div>
          </div>

          <div className="pt-1 flex flex-col gap-2">
            <Button variant="outline" onClick={onOpenDocPicker} disabled={!!busy}>
              <Link2 className="mr-2 h-4 w-4" /> Select docs for this session
            </Button>
            {error && <div className="text-sm text-destructive">{error}</div>}
          </div>
        </div>

        <DialogFooter>
          <Button onClick={submit} disabled={!!busy}>{busy ? "Starting..." : "Start Session"}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
