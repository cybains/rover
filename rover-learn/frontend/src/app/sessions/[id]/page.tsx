'use client';
import { useEffect, useState } from 'react';
import { apiGet } from '../../../lib/api';

interface Segment {
  textSrc: string;
  textEn: string;
}

export default function SessionDetail({ params }: { params: { id: string } }) {
  const { id } = params;
  const [session, setSession] = useState<{ segments: Segment[] } | null>(null);

  useEffect(() => {
    apiGet(`/sessions/${id}`).then(setSession);
  }, [id]);

  if (!session) return <div>Loading...</div>;

  return (
    <div style={{ height: '100%', display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
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
  );
}
