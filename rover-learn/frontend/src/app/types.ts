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

export type SimpleSession = {
  id: string;
  title: string;
  subject?: string;
  course?: string;
  language?: string;
  tags?: string;
  docIds: string[];
  finished: boolean;
  createdAt: string;
  accumMs: number;
};
