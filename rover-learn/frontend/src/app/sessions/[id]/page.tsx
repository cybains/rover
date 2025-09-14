'use client';
import { useEffect, useState } from 'react';
import { apiGet, exportSession } from '../../../lib/api';

interface Segment {
  textSrc: string;
  textEn: string;
}

export default function SessionDetail({ params }: { params: { id: string } }) {
  const { id } = params;
  const [session, setSession] = useState<
    { title: string; segments: Segment[]; segmentsCount: number } | null
  >(null);
  const [exportInfo, setExportInfo] = useState<
    { exportDir: string; files: string[] } | null
  >(null);

  useEffect(() => {
    apiGet(`/sessions/${id}`).then(setSession);
  }, [id]);

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
          {session.segments.map((s, i) => (
            <p key={i}>{s.textSrc}</p>
          ))}
        </div>
        <div style={{ overflowY: 'auto', padding: '1rem' }}>
          {session.segments.map((s, i) => (
            <p key={i}>{s.textEn}</p>
          ))}
        </div>
      </div>
    </div>
  );
}
