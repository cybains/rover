"use client";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FolderDown } from "lucide-react";

export default function ExportsView() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-16">
      <div className="text-2xl font-semibold mb-4">Exports</div>
      <div className="grid grid-cols-2 gap-3">
        <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> JSONL</Button>
        <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> VTT / SRT</Button>
        <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> Transcripts</Button>
        <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> Study Pack</Button>
      </div>
      <div className="text-xs text-muted-foreground mt-3">
        Exports are local; nothing leaves your machine.
      </div>
    </div>
  );
}
