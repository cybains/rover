'use client';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { apiGet } from '../../lib/api';

interface SessionInfo {
  _id: string;
  title: string;
  createdAt: string;
}

export default function SessionsPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);

  useEffect(() => {
    apiGet('/sessions').then(setSessions);
  }, []);

  return (
    <div style={{ padding: '1rem' }}>
      <h1>Sessions</h1>
      <ul>
        {sessions.map((s) => (
          <li key={s._id}>
            <Link href={`/sessions/${s._id}`}>
              {s.title} - {new Date(s.createdAt).toLocaleString()}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
