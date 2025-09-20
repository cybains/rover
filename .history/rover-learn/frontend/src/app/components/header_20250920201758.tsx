"use client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Timer, FileText } from "lucide-react";
import type { DocMeta, SimpleSession } from "@/app/types";

function LatencyPill({ ms }: { ms: number }) {
  const color = ms < 400 ? "bg-emerald-500" : ms < 800 ? "bg-amber-500" : "bg-rose-500";
  return (
    <div className={`flex items-center gap-2 rounded-full px-3 py-1 text-white ${color}`}>
      <Timer className="h-4 w-4" />
      <span className="text-sm">{ms} ms</span>
    </div>
  );
}

function StatusPill({ live }: { live: boolean }) {
  return (
    <Badge variant={live ? "default" : "secondary"} className="rounded-full px-3 py-1">
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${live ? "bg-emerald-500" : "bg-gray-400"}`} />
        {live ? "Live" : "Idle"}
      </div>
    </Badge>
  );
}

type HeaderProps = {
  latency: number;
  live: boolean;
  session: SimpleSession | null;
  docs: DocMeta[];
  onOpenMenu: () => void;
  onOpenDoc: (id: string) => void;
  version?: string;
};

export default function AppHeader({
  latency,
  live,
  session,
  docs,
  onOpenMenu,
  onOpenDoc,
  version = "v0.6 UI",
}: HeaderProps) {
  return (
    <div className="sticky top-0 z-20 backdrop-blur supports-[backdrop-filter]:bg-background/70 border-b">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-4 py-3">
        <button className="flex items-center gap-3 group cursor-pointer" onClick={onOpenMenu}>
          <div className="h-8 w-8 rounded-xl bg-foreground/90 text-background grid place-items-center font-bold group-hover:opacity-90">Λ</div>
          <div>
            <div className="font-semibold leading-tight group-hover:underline">Lab</div>
            <div className="text-xs text-muted-foreground">Personal build</div>
          </div>
          <Badge variant="outline" className="ml-2">{version}</Badge>
        </button>

        <div className="flex items-center gap-3">
          <LatencyPill ms={latency} />
          {session && (
            <div className="flex items-center gap-2">
              <Badge variant="secondary">{session.title}</Badge>
              {session.docIds?.map((id) => {
                const d = docs.find((x) => x.id === id);
                if (!d) return null;
                return (
                  <Badge
                    key={id}
                    variant="outline"
                    className="gap-1 cursor-pointer"
                    onClick={() => onOpenDoc(id)}
                  >
                    <FileText className="h-3 w-3" /> {d.filename}
                  </Badge>
                );
              })}
            </div>
          )}
          <StatusPill live={live} />
        </div>
      </div>
    </div>
  );
}
