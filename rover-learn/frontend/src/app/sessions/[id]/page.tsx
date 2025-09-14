'use client';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import {
  apiGet,
  exportSession,
  recomputeQA,
  bookmarkSegment,
  unbookmarkSegment,
  renameSpeaker,
  getGlossary,
} from '../../../lib/api';

interface Paragraph {
  _id: string;
  textSrc: string;
  textEn: string;
  idxStart: number;
  speaker: string;
  isQuestion?: boolean;
  qa?: { bestAnswerIdx?: number };
  bookmark?: boolean;
  glossaryHits?: string[];
}

export default function SessionDetail({ params }: { params: { id: string } }) {
  const { id } = params;
  const [session, setSession] = useState<
    { title: string; segments: Paragraph[]; segmentsCount: number } | null
  >(null);
  const [exportInfo, setExportInfo] = useState<
    { exportDir: string; files: string[] } | null
  >(null);
  const answerIdxsRef = useRef<Set<number>>(new Set());
  const [speakerMap, setSpeakerMap] = useState<Record<string, string>>({});
  const [glossary, setGlossary] = useState<Record<string, string>>({});

  useEffect(() => {
    apiGet(`/sessions/${id}`).then((data) => {
      setSession(data);
      setSpeakerMap(data.speakerMap || {});
    });
  }, [id]);

  useEffect(() => {
    getGlossary().then((list) => {
      const map: Record<string, string> = {};
      list.forEach((t: any) => {
        map[t.id] = t.en;
      });
      setGlossary(map);
    });
  }, []);

  useEffect(() => {
    if (!session) return;
    const set = new Set<number>();
    session.segments.forEach((s) => {
      if (s.isQuestion && s.qa && s.qa.bestAnswerIdx !== undefined) {
        set.add(s.qa.bestAnswerIdx);
      }
    });
    answerIdxsRef.current = set;
  }, [session]);

  const jumpToIdx = (idx: number) => {
    const el = document.getElementById(`seg-${idx}`);
    el && el.scrollIntoView({ behavior: 'smooth' });
    const en = document.getElementById(`seg-en-${idx}`);
    en && en.scrollIntoView({ behavior: 'smooth' });
  };

  const toggleBookmark = async (seg: Paragraph) => {
    if (seg.bookmark) await unbookmarkSegment(seg._id);
    else await bookmarkSegment(seg._id);
    const data = await apiGet(`/sessions/${id}`);
    setSession(data);
  };

  const doRecompute = async () => {
    await recomputeQA(id);
    const data = await apiGet(`/sessions/${id}`);
    setSession(data);
  };

  const doRename = async (name: string) => {
    const to = prompt('Rename speaker', speakerMap[name] || name);
    if (!to || to === name) return;
    await renameSpeaker(id, name, to);
    setSpeakerMap((prev) => ({ ...prev, [name]: to }));
  };

  const renderGloss = (text: string, hits?: string[]) => {
    if (!hits || !hits.length) return text;
    let html = text;
    hits.forEach((h) => {
      const term = glossary[h];
      if (!term) return;
      const re = new RegExp(`\\b${term}\\b`, 'g');
      html = html.replace(
        re,
        `<span style="text-decoration:underline dotted" title="${term}">${term}</span>`
      );
    });
    return <span dangerouslySetInnerHTML={{ __html: html }} />;
  };

  if (!session) return <div>Loading...</div>;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div>
            <h1>{session.title}</h1>
            <p>Segments ({session.segmentsCount})</p>
          </div>
          <button onClick={() => exportSession(id).then(setExportInfo)}>Export</button>
          <button onClick={doRecompute} style={{ marginLeft: '0.5rem' }}>
            Recompute Q/A
          </button>
          <Link href={`/sessions/${id}/highlights`} style={{ marginLeft: '0.5rem' }}>
            Highlights
          </Link>
        </div>
        {exportInfo && (
          <div style={{ background: '#e0ffe0', padding: '0.5rem', marginTop: '0.5rem' }}>
            <p>Exported to: {exportInfo.exportDir}</p>
            <ul>
              {exportInfo.files.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
        <div style={{ overflowY: 'auto', padding: '1rem', borderRight: '1px solid #ccc' }}>
          {session.segments.map((s) => {
            const style: any = {};
            if (answerIdxsRef.current.has(s.idxStart)) style.background = '#e0ffe0';
            if (s.isQuestion) style.background = '#e0f0ff';
            return (
              <p key={s._id} id={`seg-${s.idxStart}`} style={style}>
                <span
                  style={{ cursor: 'pointer', marginRight: '0.25rem' }}
                  onClick={() => toggleBookmark(s)}
                >
                  {s.bookmark ? '★' : '☆'}
                </span>
                <strong
                  style={{ cursor: 'pointer', marginRight: '0.25rem' }}
                  onClick={() => doRename(s.speaker)}
                >
                  {speakerMap[s.speaker] || s.speaker}:
                </strong>
                {s.textSrc}
                {s.isQuestion && s.qa?.bestAnswerIdx !== undefined && (
                  <button
                    onClick={() => jumpToIdx(s.qa!.bestAnswerIdx!)}
                    style={{ marginLeft: '0.5rem' }}
                  >
                    Find answer
                  </button>
                )}
              </p>
            );
          })}
        </div>
        <div style={{ overflowY: 'auto', padding: '1rem' }}>
          {session.segments.map((s) => {
            const style: any = {};
            if (answerIdxsRef.current.has(s.idxStart)) style.background = '#e0ffe0';
            if (s.isQuestion) style.background = '#e0f0ff';
            return (
              <p key={s._id} id={`seg-en-${s.idxStart}`} style={style}>
                <strong
                  style={{ cursor: 'pointer', marginRight: '0.25rem' }}
                  onClick={() => doRename(s.speaker)}
                >
                  {speakerMap[s.speaker] || s.speaker}:
                </strong>
                {renderGloss(s.textEn, s.glossaryHits)}
              </p>
            );
          })}
        </div>
      </div>
    </div>
  );
}
