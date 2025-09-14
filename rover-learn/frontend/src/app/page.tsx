'use client';
import { useEffect, useRef, useState } from 'react';
import { apiPost, connectWs } from '../lib/api';

interface Segment {
  textSrc: string;
  textEn: string;
}

export default function LivePage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const el = leftRef.current;
    if (!el) return;
    const onScroll = () => {
      setScrolled(el.scrollTop + el.clientHeight < el.scrollHeight);
    };
    el.addEventListener('scroll', onScroll);
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  const start = async () => {
    const data = await apiPost('/sessions/start', {});
    setSessionId(data._id);
    setSegments([]);
    const ws = connectWs(data._id);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      const seg = JSON.parse(ev.data);
      setSegments((prev) => [...prev, seg]);
    };
  };

  const stop = async () => {
    wsRef.current?.close();
    wsRef.current = null;
    if (sessionId) {
      await apiPost('/sessions/stop', { sessionId });
    }
    setSessionId(null);
  };

  useEffect(() => {
    leftRef.current && (leftRef.current.scrollTop = leftRef.current.scrollHeight);
    rightRef.current && (rightRef.current.scrollTop = rightRef.current.scrollHeight);
  }, [segments]);

  const jump = () => {
    leftRef.current && (leftRef.current.scrollTop = leftRef.current.scrollHeight);
    rightRef.current && (rightRef.current.scrollTop = rightRef.current.scrollHeight);
  };

  return (
    <div style={{ height: '100%', display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
      <div ref={leftRef} style={{ overflowY: 'auto', padding: '1rem', borderRight: '1px solid #ccc' }}>
        {segments.map((s, i) => (
          <p key={i}>{s.textSrc}</p>
        ))}
      </div>
      <div ref={rightRef} style={{ overflowY: 'auto', padding: '1rem' }}>
        {segments.map((s, i) => (
          <p key={i}>{s.textEn}</p>
        ))}
      </div>
      <div style={{ position: 'absolute', bottom: '1rem', left: '50%', transform: 'translateX(-50%)' }}>
        {!sessionId ? (
          <button onClick={start}>Start</button>
        ) : (
          <button onClick={stop}>Stop</button>
        )}
        {scrolled && (
          <button onClick={jump} style={{ marginLeft: '1rem' }}>
            Jump to Live
          </button>
        )}
      </div>
    </div>
  );
}
