import { useState } from "react";

export interface Segment {
  _id?: string;
  sessionId: string;
  idx: number;
  tStart: number;
  tEnd: number;
  lang: string;
  speaker: string;
  textSrc: string;
  textEn: string;
  partial: boolean;
  confidence?: number;
}

export function useLiveFeed() {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [atLive, setAtLive] = useState(true);
  const append = (segs: Segment[]) => setSegments((s) => [...s, ...segs]);
  const reset = () => setSegments([]);
  return { segments, append, reset, atLive, setAtLive };
}
