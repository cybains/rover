"use client";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { FolderDown } from "lucide-react";

export default function SummariesView() {
  return (
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
  );
}
