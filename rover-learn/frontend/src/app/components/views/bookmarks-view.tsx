"use client";
import { Card, CardContent } from "@/components/ui/card";

export default function BookmarksView() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-16">
      <div className="text-2xl font-semibold mb-4">Bookmarks</div>
      <Card>
        <CardContent className="py-6 text-sm text-muted-foreground">
          Your saved moments will appear here.
        </CardContent>
      </Card>
    </div>
  );
}
