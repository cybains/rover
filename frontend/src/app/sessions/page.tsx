'use client';
import { useEffect, useState } from 'react';
import { apiGet } from '../../lib/api';

export default function SessionsPage() {
  const [sessions, setSessions] = useState<any[]>([]);
  useEffect(() => {
    apiGet('/sessions').then(setSessions);
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h3>Sessions</h3>
      <ul>
        {sessions.map((s) => (
          <li key={s._id}>
            <a href={`/app/live?session=${s._id}`}>{s.title} - {s.status}</a>
          </li>
        ))}
      </ul>
    </div>
  );
}
