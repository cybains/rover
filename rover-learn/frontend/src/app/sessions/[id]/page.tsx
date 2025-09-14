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
  generate,
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
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [gen, setGen] = useState<any | null>(null);
  const [genType, setGenType] = useState<string>('');

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

  const toggleSelect = (sid: string) => {
    setSelected((prev) => {
      const s = new Set(prev);
      if (s.has(sid)) s.delete(sid);
      else s.add(sid);
      return s;
    });
  };

  const doGenerate = async (type: string) => {
    const paraIds = Array.from(selected);
    const res = await generate(type, id, paraIds.length ? paraIds : undefined);
    setGen(res.output);
    setGenType(type);
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
          <Link href={`/sessions/${id}/summaries`} style={{ marginLeft: '0.5rem' }}>
            Summaries
          </Link>
          <Link href={`/sessions/${id}/flashcards`} style={{ marginLeft: '0.5rem' }}>
            Flashcards
          </Link>
          <select
            onChange={(e) => {
              const v = e.target.value;
              if (v) {
                doGenerate(v);
                e.target.value = '';
              }
            }}
            style={{ marginLeft: '0.5rem' }}
          >
            <option value="">Generate...</option>
            <option value="summary">Summary</option>
            <option value="flashcards">Flashcards</option>
            <option value="quiz">Quiz</option>
            <option value="explain">Explain</option>
          </select>
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
      <div
        style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: gen ? '1fr 1fr 1fr' : '1fr 1fr',
        }}
      >
        <div style={{ overflowY: 'auto', padding: '1rem', borderRight: '1px solid #ccc' }}>
          {session.segments.map((s) => {
            const style: any = {};
            if (answerIdxsRef.current.has(s.idxStart)) style.background = '#e0ffe0';
            if (s.isQuestion) style.background = '#e0f0ff';
            if (selected.has(s._id)) style.outline = '1px dashed #888';
            return (
              <p key={s._id} id={`seg-${s.idxStart}`} style={style}>
                <input
                  type="checkbox"
                  checked={selected.has(s._id)}
                  onChange={() => toggleSelect(s._id)}
                  style={{ marginRight: '0.25rem' }}
                />
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
            if (selected.has(s._id)) style.outline = '1px dashed #888';
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
        {gen && (
          <div style={{ overflowY: 'auto', padding: '1rem', borderLeft: '1px solid #ccc' }}>
            <h3>{genType}</h3>
            <pre>{JSON.stringify(gen, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
