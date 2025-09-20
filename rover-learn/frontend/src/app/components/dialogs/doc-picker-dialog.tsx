"use client";
import { useMemo, useState, useEffect } from "react";
import type { DocMeta } from "@/app/types";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { FileText, Link2, Plus, Search } from "lucide-react";
import { fileExtension, formatFileSize, formatIsoDate } from "@/app/lib/format";

type Mode = "new-session" | "link-to-current";

type Props = {
  open: boolean;
  mode: Mode;
  docs: DocMeta[];
  initiallySelectedIds?: string[];
  uploading?: boolean;
  linking?: boolean;
  onOpenChange: (open: boolean) => void;
  onAddFileClick: () => void;       // triggers hidden input in parent
  onConfirm: (selectedIds: string[]) => void;
};

export default function DocPickerDialog({
  open, mode, docs, initiallySelectedIds = [],
  uploading, linking, onOpenChange, onAddFileClick, onConfirm
}: Props) {
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string[]>(initiallySelectedIds);

  useEffect(() => {
    if (open) setSelected(initiallySelectedIds);
  }, [open, initiallySelectedIds]);

  const filteredDocs = useMemo(() => {
    const term = search.trim().toLowerCase();
    if (!term) return docs;
    return docs.filter((d) =>
      d.filename.toLowerCase().includes(term) ||
      d.tags.some((t) => t.toLowerCase().includes(term))
    );
  }, [docs, search]);

  const toggle = (id: string) => {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  const handleConfirm = () => onConfirm(selected);

  const cta =
    mode === "new-session"
      ? (linking ? "Applying..." : "Use selected in new session")
      : (linking ? "Linking..." : "Link to current session");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[680px]">
        <DialogHeader>
          <DialogTitle>{mode === "new-session" ? "Select documents for session" : "Select documents to link"}</DialogTitle>
        </DialogHeader>

        <div className="flex items-center gap-2 mb-3">
          <div className="relative w-full">
            <Search className="h-4 w-4 absolute left-2 top-2.5 opacity-60" />
            <Input className="pl-8" placeholder="Search docs..." value={search} onChange={(e) => setSearch(e.target.value)} />
          </div>
          <Button variant="secondary" onClick={onAddFileClick} disabled={uploading}>
            {uploading ? 'Uploading...' : (<><Plus className="mr-2 h-4 w-4" /> Add File</>)}
          </Button>
        </div>

        <div className="max-h-[46vh] overflow-y-auto space-y-2 pr-1">
          {filteredDocs.map((doc) => (
            <div key={doc.id} className="flex items-center justify-between border rounded-lg p-3">
              <div className="flex items-center gap-3">
                <Badge variant="secondary" className="gap-1">
                  <FileText className="h-3 w-3" /> {fileExtension(doc.filename)}
                </Badge>
                <div>
                  <div className="font-medium text-sm">{doc.filename}</div>
                  <div className="text-xs text-muted-foreground">
                    {formatFileSize(doc.size)} | uploaded {formatIsoDate(doc.uploadedAt)}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Checkbox id={`sel-${doc.id}`} checked={selected.includes(doc.id)} onCheckedChange={() => toggle(doc.id)} />
                <label htmlFor={`sel-${doc.id}`} className="text-sm">Select</label>
              </div>
            </div>
          ))}
          {!filteredDocs.length && (
            <div className="text-sm text-muted-foreground p-6 text-center">No documents match your search.</div>
          )}
        </div>

        <DialogFooter>
          <Button
            onClick={handleConfirm}
            disabled={linking || selected.length === 0}
          >
            <Link2 className="mr-2 h-4 w-4" /> {cta}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
