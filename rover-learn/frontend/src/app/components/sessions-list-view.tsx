"use client";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";
import type { SimpleSession } from "@/app/types";
import { formatDuration } from "@/app/lib/format";

type Props = {
  sessions: SimpleSession[];
  loading?: boolean;
  error?: string | null;
  onDelete: (id: string) => void;
  onOpen: (id: string) => void;
};

export default function SessionsListView({ sessions, loading, error, onDelete, onOpen }: Props) {
  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-4">
        <div className="text-2xl font-semibold">Sessions</div>
        <div className="text-sm text-muted-foreground">{loading ? "Loading..." : `${sessions.length} total`}</div>
      </div>

      {error && <div className="mb-3 text-sm text-destructive">{error}</div>}

      {loading ? (
        <Card><CardContent className="py-8 text-center text-sm text-muted-foreground">Loading your sessions...</CardContent></Card>
      ) : sessions.length === 0 ? (
        <Card><CardContent className="py-8 text-center text-sm text-muted-foreground">No sessions yet. Start one from the menu.</CardContent></Card>
      ) : (
        <div className="space-y-2">
          {sessions.map((s) => {
            const docCount = typeof s.documentsCount === "number" ? s.documentsCount : s.docIds?.length || 0;
            const statusLabel = s.status ? s.status.charAt(0).toUpperCase() + s.status.slice(1) : "Unknown";
            const badgeVariant = s.status === "live" ? "default" : "secondary";
            return (
              <Card key={s.id}>
                <CardContent className="py-4 flex items-center justify-between">
                  <div>
                    <div className="font-medium">{s.title}</div>
                    <div className="text-xs text-muted-foreground">
                      {new Date(s.createdAt).toLocaleString()} • {formatDuration(s.accumMs ?? 0)} • {docCount} doc(s)
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={badgeVariant}>{statusLabel}</Badge>
                    <Button size="sm" variant="outline" onClick={() => onOpen(s.id)}>View</Button>
                    <Button
                      size="iconLg"
                      variant="ghost"
                      aria-label="Delete session"
                      onClick={() => onDelete(s.id)}
                      className="rounded-full border border-red-200 text-red-500 hover:bg-red-50 hover:text-red-600"
                    >
                      <Trash2 className="h-9 w-9" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
