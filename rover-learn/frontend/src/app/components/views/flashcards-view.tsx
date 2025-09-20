"use client";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function FlashcardsView() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      <div className="text-2xl font-semibold mb-4">Flashcards</div>
      <Card>
        <CardContent className="py-6 space-y-4">
          <div className="text-sm text-muted-foreground">Auto-generate Q/A from session. Review mode like Anki.</div>
          <div className="flex items-center gap-2">
            <Button>Generate</Button>
            <Button variant="secondary">Start Review</Button>
          </div>
          <div className="p-3 border rounded text-sm text-muted-foreground">(Generated cards will appear here)</div>
        </CardContent>
      </Card>
    </div>
  );
}
