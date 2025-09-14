'use client';
import { useEffect, useState } from 'react';
import { getGenerations } from '../../../../lib/api';

export default function Summaries({ params }: { params: { id: string } }) {
  const { id } = params;
  const [items, setItems] = useState<any[]>([]);
  useEffect(() => {
    getGenerations(id, 'summary').then(setItems);
  }, [id]);
  return (
    <div style={{ padding: '1rem' }}>
      <h1>Summaries</h1>
      <ul>
        {items.map((g: any, i: number) => (
          <li key={g._id || i}>
            <pre>{JSON.stringify(g.output, null, 2)}</pre>
          </li>
        ))}
      </ul>
    </div>
  );
}
