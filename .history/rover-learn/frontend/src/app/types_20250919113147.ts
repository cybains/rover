export type TranscriptSegment = {
  id: number;
  time: string;
  text: string;
  speaker: string;
  q: boolean;
};

export type DocMeta = {
  id: string;
  filename: string;
  contentType: string;
  size: number;
  uploadedAt?: string;
  tags: string[];
  notes: string;
  linkedSessions: string[];
};

export type SessionStatus = "live" | "stopped" | "archived" | "trashed" | string;

export type SessionMetadata = {
  subject?: string;
  course?: string;
  language?: string;
  tags?: string;
};

export type SimpleSession = {
  id: string;
  title: string;
  status: SessionStatus;
  createdAt: string;
  updatedAt?: string;
  docIds: string[];
  segmentsCount?: number;
  documentsCount?: number;
  accumMs?: number;
  metadata?: SessionMetadata;
};
