'use client';
import { useEffect, useRef, useState } from 'react';
import SplitPane from '../../components/SplitPane';
import { useLiveFeed, Segment } from '../../store/liveFeed';
import { apiGet, apiPost, openRealtime } from '../../lib/api';

export default function LivePage() {
  const { segments, append, reset, atLive, setAtLive } = useLiveFeed();
  const [session, setSession] = useState<any>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const lastSaved = useRef(0);
  const queryId = typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('session') : null;

  useEffect(() => {
    if (queryId) {
      apiGet('/sessions').then((list) => {
        const found = list.find((s: any) => s._id === queryId) || { _id: queryId, title: 'Session', status: 'stopped' };
        setSession(found);
        apiGet(`/sessions/${queryId}/segments`).then((segs) => {
          append(segs);
          lastSaved.current = segs.length;
        });
      });
      return;
    }
    apiGet('/sessions').then((list) => {
      const active = list.find((s: any) => s.status === 'active');
      if (active) {
        setSession(active);
        apiGet(`/sessions/${active._id}/segments`).then((segs) => {
          append(segs);
          lastSaved.current = segs.length;
        });
        connectWs(active._id);
      }
    });
  }, [queryId]);

  useEffect(() => {
    const t = setInterval(() => {
      if (session && wsRef.current) {
        const unsaved = segments.slice(lastSaved.current);
        if (unsaved.length) {
          apiPost('/segments', { sessionId: session._id, segments: unsaved });
          lastSaved.current = segments.length;
        }
      }
    }, 5000);
    return () => clearInterval(t);
  }, [segments, session]);

  const connectWs = (sid: string) => {
    const ws = openRealtime(sid);
    ws.onmessage = (ev) => {
      const segs: Segment[] = JSON.parse(ev.data);
      append(segs);
    };
    wsRef.current = ws;
  };

  const start = async () => {
    const s = await apiPost('/sessions/start', { title: 'Session' });
    setSession(s);
    reset();
    lastSaved.current = 0;
    connectWs(s._id);
  };

  const stop = async () => {
    if (!session) return;
    await apiPost('/sessions/stop', { id: session._id });
    wsRef.current && wsRef.current.close();
    setSession(null);
  };

  const renderSeg = (s: Segment, field: 'textSrc' | 'textEn') => {
    const txt = s[field];
    const q = txt.endsWith('?');
    return (
      <div key={s.idx} style={{ background: q ? '#e0f0ff' : undefined, marginBottom: 4 }}>
        {txt} {q && <span style={{ fontSize: 10, color: '#06f' }}>Find Answer</span>}
      </div>
    );
  };

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: 8, borderBottom: '1px solid #ccc' }}>
        <button onClick={start} disabled={!!session}>Start Session</button>
        <button onClick={stop} disabled={!session}>Stop</button>
        <span style={{ marginLeft: 8 }}>{session ? session.title : 'No session'}</span>
        <span style={{ marginLeft: 8, color: session ? 'green' : 'red' }}>
          {session ? 'live' : 'stopped'}
        </span>
      </div>
      <div style={{ flex: 1 }}>
        <SplitPane
          left={segments.map(s => renderSeg(s, 'textSrc'))}
          right={segments.map(s => renderSeg(s, 'textEn'))}
          atLive={atLive}
          onAtLiveChange={setAtLive}
        />
      </div>
    </div>
  );
}
