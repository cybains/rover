'use client';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { apiGet } from '../../lib/api';

interface SessionInfo {
  _id: string;
  title: string;
  createdAt: string;
  updatedAt?: string;
  status: string;
  segmentsCount: number;
}

export default function SessionsPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);

  useEffect(() => {
    apiGet('/sessions').then(setSessions);
  }, []);

  return (
    <div style={{ padding: '1rem' }}>
      <h1>Sessions</h1>
      <table>
        <thead>
          <tr>
            <th>Title</th>
            <th>Status</th>
            <th>Segments</th>
            <th>Created</th>
            <th>Updated</th>
          </tr>
        </thead>
        <tbody>
          {sessions.map((s) => (
            <tr key={s._id}>
              <td>
                <Link href={`/sessions/${s._id}`}>{s.title}</Link>
              </td>
              <td>{s.status}</td>
              <td>{s.segmentsCount}</td>
              <td>{new Date(s.createdAt).toLocaleString()}</td>
              <td>{s.updatedAt ? new Date(s.updatedAt).toLocaleString() : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
