export type TranscriptSegment = {
  id: number | string;
  time: string;           // HH:MM:SS
  text: string;           // UI display text
  speaker: string;        // "Speaker"
  q: boolean;             // question highlight
};

export type DocMeta = {
  id: string;
  filename: string;
  size: number;
  uploadedAt?: string;
  notes?: string;
};

export type SimpleSession = {
  id: string;
  title: string;
  createdAt: string;
  accumMs?: number;
  docIds: string[];
};
