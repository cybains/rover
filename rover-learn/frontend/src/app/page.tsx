'use client';
import { useEffect, useRef, useState } from 'react';
import {
  apiPost,
  connectWs,
  startSession,
  startCapture,
  stopCapture,
  bookmarkSegment,
  unbookmarkSegment,
  getGlossary,
  renameSpeaker as apiRenameSpeaker,
} from '../lib/api';

export default function LivePage() {
  const [session, setSession] = useState<any | null>(null);
  const [order, setOrder] = useState<string[]>([]);
  const [paras, setParas] = useState<Map<string, any>>(new Map());
  const [answerIdxs, setAnswerIdxs] = useState<Set<number>>(new Set());
  const wsRef = useRef<WebSocket | null>(null);
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const idxMapRef = useRef<Map<number, string>>(new Map());
  const [scrolled, setScrolled] = useState(false);
  const [title, setTitle] = useState('Untitled Session');
  const lastTsRef = useRef(Date.now());
  const [latency, setLatency] = useState(0);
  const [source, setSource] = useState<'auto' | 'mic' | 'loopback'>('auto');
  const [speakerMap, setSpeakerMap] = useState<Record<string, string>>({});
  const [glossary, setGlossary] = useState<Record<string, string>>({});

  useEffect(() => {
    const el = leftRef.current;
    if (!el) return;
    const onScroll = () => {
      setScrolled(el.scrollTop + el.clientHeight < el.scrollHeight);
    };
    el.addEventListener('scroll', onScroll);
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    const id = setInterval(() => {
      setLatency(Date.now() - lastTsRef.current);
    }, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    getGlossary().then((list) => {
      const map: Record<string, string> = {};
      list.forEach((t: any) => {
        map[t.id] = t.en;
      });
      setGlossary(map);
    });
  }, []);

  const start = async () => {
    const data = await startSession(title);
    setSession(data);
    setOrder([]);
    setParas(new Map());
    setSpeakerMap({});
    await startCapture(data._id, source);
    lastTsRef.current = Date.now();
    const ws = connectWs(data._id);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'speakerRename') {
        setSpeakerMap((prev) => ({ ...prev, [msg.from]: msg.to }));
        return;
      }
      const seg = msg;
      if (!seg || !seg.paraId) return;
      lastTsRef.current = Date.now();
      setParas((prev) => {
        const map = new Map(prev);
        const p = map.get(seg.paraId) || {};
        p.speaker = seg.speaker;
        if (seg.kind === 'partial') {
          p.srcPartial = seg.textSrcPartial;
          p.enPartial = seg.textEnPartial;
        } else if (seg.kind === 'final') {
          p.srcFinal = seg.textSrc;
          p.enFinal = seg.textEn;
          p.srcPartial = undefined;
          p.enPartial = undefined;
          p.idx = seg.idxStart;
          p.isQuestion = seg.isQuestion;
          p.qa = seg.qa;
          p.bookmark = seg.bookmark;
          p._id = seg._id;
          p.glossaryHits = seg.glossaryHits;
          idxMapRef.current.set(seg.idxStart, seg.paraId);
        }
        map.set(seg.paraId, p);
        const ans = new Set<number>();
        map.forEach((v) => {
          if (v.isQuestion && v.qa && v.qa.bestAnswerIdx !== undefined) {
            ans.add(v.qa.bestAnswerIdx);
          }
        });
        setAnswerIdxs(ans);
        return map;
      });
      setOrder((prev) => (prev.includes(seg.paraId) ? prev : [...prev, seg.paraId]));
    };
  };

  const stop = async () => {
    wsRef.current?.close();
    wsRef.current = null;
    if (session) {
      await stopCapture(session._id);
      await apiPost('/sessions/stop', { sessionId: session._id });
      setSession({ ...session, status: 'stopped' });
    }
  };

  useEffect(() => {
    leftRef.current && (leftRef.current.scrollTop = leftRef.current.scrollHeight);
    rightRef.current && (rightRef.current.scrollTop = rightRef.current.scrollHeight);
  }, [order, paras]);

  const jump = () => {
    leftRef.current && (leftRef.current.scrollTop = leftRef.current.scrollHeight);
    rightRef.current && (rightRef.current.scrollTop = rightRef.current.scrollHeight);
  };

  const jumpToIdx = (idx: number) => {
    const paraId = idxMapRef.current.get(idx);
    if (!paraId) return;
    const el = document.getElementById(paraId);
    el && el.scrollIntoView({ behavior: 'smooth' });
    const el2 = document.getElementById(`en-${paraId}`);
    el2 && el2.scrollIntoView({ behavior: 'smooth' });
  };

  const toggleBookmark = async (id: string, p: any) => {
    if (!p._id) return;
    if (p.bookmark) await unbookmarkSegment(p._id);
    else await bookmarkSegment(p._id);
    setParas((prev) => {
      const map = new Map(prev);
      const obj = map.get(id);
      if (obj) obj.bookmark = !p.bookmark;
      map.set(id, obj);
      return map;
    });
  };

  const renameSpeaker = async (name: string) => {
    if (!session) return;
    const to = prompt('Rename speaker', speakerMap[name] || name);
    if (!to || to === name) return;
    await apiRenameSpeaker(session._id, name, to);
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

  return (
    <div style={{ height: '100%', display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
      <div ref={leftRef} style={{ overflowY: 'auto', padding: '1rem', borderRight: '1px solid #ccc' }}>
        {order.map((id) => {
          const p = paras.get(id);
          if (!p) return null;
          const txt = p.srcFinal || p.srcPartial || '';
          const partial = !p.srcFinal;
          const style: any = { fontStyle: partial ? 'italic' : 'normal' };
          if (answerIdxs.has(p.idx)) style.background = '#e0ffe0';
          if (p.isQuestion) style.background = '#e0f0ff';
          return (
            <p key={id} id={id} style={style}>
              <span
                style={{ cursor: 'pointer', marginRight: '0.25rem' }}
                onClick={() => toggleBookmark(id, p)}
              >
                {p.bookmark ? '★' : '☆'}
              </span>
              <strong
                style={{ cursor: 'pointer', marginRight: '0.25rem' }}
                onClick={() => renameSpeaker(p.speaker || 'Speaker 1')}
              >
                {speakerMap[p.speaker || 'Speaker 1'] || p.speaker || 'Speaker 1'}:
              </strong>
              {txt}
              {p.isQuestion && p.qa?.bestAnswerIdx !== undefined && (
                <button
                  onClick={() => jumpToIdx(p.qa!.bestAnswerIdx!)}
                  style={{ marginLeft: '0.5rem' }}
                >
                  Find answer
                </button>
              )}
            </p>
          );
        })}
      </div>
      <div ref={rightRef} style={{ overflowY: 'auto', padding: '1rem' }}>
        {order.map((id) => {
          const p = paras.get(id);
          if (!p) return null;
          const txt = p.enFinal || p.enPartial || '';
          const partial = !p.enFinal;
          const style: any = { fontStyle: partial ? 'italic' : 'normal' };
          if (answerIdxs.has(p.idx)) style.background = '#e0ffe0';
          if (p.isQuestion) style.background = '#e0f0ff';
          return (
            <p key={id} id={`en-${id}`} style={style}>
              <strong
                style={{ cursor: 'pointer', marginRight: '0.25rem' }}
                onClick={() => renameSpeaker(p.speaker || 'Speaker 1')}
              >
                {speakerMap[p.speaker || 'Speaker 1'] || p.speaker || 'Speaker 1'}:
              </strong>
              {renderGloss(txt, p.glossaryHits)}
            </p>
          );
        })}
      </div>
        <div style={{ position: 'absolute', bottom: '1rem', left: '50%', transform: 'translateX(-50%)', textAlign: 'center' }}>
        <div style={{ marginBottom: '0.5rem' }}>
          <input value={title} onChange={(e) => setTitle(e.target.value)} />
        </div>
        <div style={{ marginBottom: '0.5rem' }}>
          <select value={source} onChange={(e) => setSource(e.target.value as any)}>
            <option value="auto">Auto (Mic→Loopback)</option>
            <option value="mic">Mic only</option>
            <option value="loopback">Loopback only</option>
          </select>
        </div>
        <div>
          {session && (
            <span style={{ marginRight: '0.5rem', color: session.status === 'live' ? 'green' : 'gray' }}>
              ● {session.status === 'live' ? 'Live' : 'Stopped'}
            </span>
          )}
          {!session || session.status !== 'live' ? (
            <button onClick={start}>Start</button>
          ) : (
            <button onClick={stop}>Stop</button>
          )}
          {scrolled && (
            <button onClick={jump} style={{ marginLeft: '1rem' }}>
              Jump to Live
            </button>
          )}
          <span
            style={{
              marginLeft: '1rem',
              background: '#eee',
              color: '#555',
              padding: '0.1rem 0.4rem',
              borderRadius: '0.5rem',
              fontSize: '0.75rem',
            }}
          >
            Audio: {source === 'auto' ? 'Auto' : source === 'mic' ? 'Mic' : 'Loopback'}
          </span>
          <span
            style={{
              marginLeft: '1rem',
              background: '#eee',
              color: '#555',
              padding: '0.1rem 0.4rem',
              borderRadius: '0.5rem',
              fontSize: '0.75rem',
            }}
          >
            MT: Marian (local)
          </span>
        </div>
      </div>
      <div style={{ position: 'absolute', top: '0.5rem', right: '0.5rem', color: latency > 2000 ? 'orange' : 'inherit' }}>
        {latency}ms
      </div>
    </div>
  );
}
