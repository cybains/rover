import type { DocMeta, SimpleSession } from "@/app/types";

const fallbackId = () =>
  typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

export const normaliseDoc = (doc: any): DocMeta => ({
  id: doc?._id ? String(doc._id) : doc?.id ? String(doc.id) : fallbackId(),
  filename:
    (doc?.filenameOriginal as string) ||
    (doc?.filename as string) ||
    (doc?.title as string) ||
    "Document",
  contentType: doc?.contentType || "application/octet-stream",
  size: Number(doc?.size ?? 0) || 0,
  uploadedAt: doc?.uploadedAt || doc?.addedAt,
  tags: Array.isArray(doc?.tags) ? doc.tags.map(String) : [],
  notes: typeof doc?.notes === "string" ? doc.notes : "",
  linkedSessions: Array.isArray(doc?.linkedSessions) ? doc.linkedSessions.map(String) : [],
});

export const normaliseSession = (raw: any): SimpleSession => {
  const docIds = Array.isArray(raw?.docIds) ? raw.docIds.map(String) : [];
  const metaSource = raw?.metadata && typeof raw.metadata === "object" ? raw.metadata : raw;
  const metadata = {
    subject: typeof metaSource?.subject === "string" && metaSource.subject.trim() ? metaSource.subject.trim() : undefined,
    course: typeof metaSource?.course === "string" && metaSource.course.trim() ? metaSource.course.trim() : undefined,
    language: typeof metaSource?.language === "string" && metaSource.language.trim() ? metaSource.language.trim() : undefined,
    tags: typeof metaSource?.tags === "string" && metaSource.tags.trim() ? metaSource.tags.trim() : undefined,
  };
  const hasMetadata = Object.values(metadata).some(Boolean);

  return {
    id: raw?._id ? String(raw._id) : raw?.id ? String(raw.id) : fallbackId(),
    title: typeof raw?.title === "string" && raw.title.trim() ? raw.title : "Untitled Session",
    status: typeof raw?.status === "string" ? raw.status : raw?.finished ? "stopped" : "live",
    createdAt: typeof raw?.createdAt === "string" && raw.createdAt ? raw.createdAt : new Date().toISOString(),
    updatedAt: typeof raw?.updatedAt === "string" && raw.updatedAt ? raw.updatedAt : undefined,
    docIds,
    segmentsCount: typeof raw?.segmentsCount === "number" ? raw.segmentsCount : undefined,
    documentsCount: typeof raw?.documentsCount === "number" ? raw.documentsCount : docIds.length || undefined,
    accumMs: typeof raw?.accumMs === "number" ? raw.accumMs : undefined,
    metadata: hasMetadata ? metadata : undefined,
  };
};
