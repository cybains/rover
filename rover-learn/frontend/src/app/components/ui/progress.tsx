import * as React from "react";
import { cn } from "@/lib/utils";

export function Progress({ value = 0, className }: { value?: number; className?: string }) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div className={cn("relative h-2 w-full overflow-hidden rounded bg-gray-200", className)}>
      <div className="h-full bg-black transition-all" style={{ width: `${clamped}%` }} />
    </div>
  );
}