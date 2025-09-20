"use client";
import { useMemo } from "react";
import type { DocMeta } from "@/app/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { FileText, Link2, Plus, Trash2 } from "lucide-react";
import { fileExtension, formatFileSize, formatIsoDate } from "@/app/lib/format";

type Props = {
  docs: DocMeta[];
  search: string;
  onSearchChange: (v: string) => void;

  selectedDocIds: string[];
  onToggleSelect: (id: string) => void;

  uploading?: boolean;
  notice?: string | null;
  error?: string | null;

  onUploadClick: () => void;
  onPreview: (id: string) => void;
  onDelete: (id: string) => void;
  onLinkToSession: () => void;
};

export default function DocsView({
  docs, search, onSearchChange, selectedDocIds,
  onToggleSelect, uploading, notice, error, onUploadClick,
  onPreview, onDelete, onLinkToSession
}: Props) {
  const filteredDocs = useMemo(() => {
    const term = search.trim().toLowerCase();
    if (!term) return docs;
    return docs.filter(d => d.filename.toLowerCase().includes(term) || d.tags.some(t => t.toLowerCase().includes(term)));
  }, [docs, search]);

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-3">
        <div className="text-2xl font-semibold">Docs</div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={onLinkToSession}><Link2 className="mr-2 h-4 w-4" /> Link to Session</Button>
          <Button onClick={onUploadClick} disabled={uploading}>
            {uploading ? 'Uploading...' : (<><Plus className="mr-2 h-4 w-4" /> Add File</>)}
          </Button>
        </div>
      </div>

      <div className="flex items-center gap-2 mb-3">
        <Input
          className="w-full"
          placeholder="Search docs..."
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
        />
      </div>

      {(notice || error || uploading) && (
        <div className="mb-3 space-y-1">
          {notice && <div className="text-sm text-emerald-600">{notice}</div>}
          {error && <div className="text-sm text-red-600">{error}</div>}
          {uploading && <div className="text-xs text-muted-foreground">Uploading...</div>}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {filteredDocs.map((doc) => (
          <Card key={doc.id} className="overflow-hidden">
            <CardHeader className="py-3">
              <CardTitle className="text/base flex items-center gap-2">
                <FileText className="h-4 w-4" /> {doc.filename}
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span>{fileExtension(doc.filename)} | {formatFileSize(doc.size)}</span>
                <span>Uploaded {formatIsoDate(doc.uploadedAt)}</span>
              </div>
              {doc.tags.length > 0 && (<div className="text-xs">Tags: {doc.tags.join(", ")}</div>)}
              {doc.notes && (<div className="text-xs">{doc.notes}</div>)}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Checkbox id={`pick-${doc.id}`} checked={selectedDocIds.includes(doc.id)} onCheckedChange={() => onToggleSelect(doc.id)} />
                  <label htmlFor={`pick-${doc.id}`}>Select</label>
                </div>
                <div className="flex items-center gap-2">
                  <Button size="sm" variant="outline" onClick={() => onPreview(doc.id)}>Preview</Button>
                  <Button
                    size="iconLg"
                    variant="ghost"
                    aria-label="Delete doc"
                    onClick={() => onDelete(doc.id)}
                    className="rounded-full border border-red-200 text-red-500 hover:bg-red-50 hover:text-red-600"
                  >
                    <Trash2 className="h-9 w-9" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredDocs.length === 0 && (
        <div className="text-sm text-muted-foreground p-6 text-center">No documents match your search.</div>
      )}
    </div>
  );
}
