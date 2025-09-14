'use client';
import { useEffect, useRef, useState } from 'react';
import { apiPost, connectWs, startSession, startCapture, stopCapture } from '../lib/api';

interface Segment {
  textSrc: string;
  textEn: string;
}

export default function LivePage() {
  const [session, setSession] = useState<any | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const [scrolled, setScrolled] = useState(false);
  const [title, setTitle] = useState('Untitled Session');
  const lastTsRef = useRef(Date.now());
  const [latency, setLatency] = useState(0);
  const [source, setSource] = useState<'auto' | 'mic' | 'loopback'>('auto');

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

  const start = async () => {
    const data = await startSession(title);
    setSession(data);
    setSegments([]);
    await startCapture(data._id, source);
    lastTsRef.current = Date.now();
    const ws = connectWs(data._id);
    wsRef.current = ws;
    ws.onmessage = (ev) => {
      const seg = JSON.parse(ev.data);
      lastTsRef.current = Date.now();
      setSegments((prev) => [...prev, seg]);
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
