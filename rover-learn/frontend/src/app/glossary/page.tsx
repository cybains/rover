'use client';
import { useEffect, useState } from 'react';
import { getGlossary } from '../../lib/api';

interface Item {
  term: string;
  de: string;
  en: string;
  notes: string;
}

export default function GlossaryPage() {
  const [items, setItems] = useState<Item[]>([]);
  useEffect(() => {
    getGlossary().then(setItems);
  }, []);

  const cell = { border: '1px solid #ccc', padding: '0.25rem' } as const;

  return (
    <div style={{ padding: '1rem' }}>
      <h1>Glossary ({items.length} terms)</h1>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={cell}>term</th>
            <th style={cell}>de</th>
            <th style={cell}>en</th>
            <th style={cell}>notes</th>
          </tr>
        </thead>
        <tbody>
          {items.map((g, i) => (
            <tr key={i}>
              <td style={cell}>{g.term}</td>
              <td style={cell}>{g.de}</td>
              <td style={cell}>{g.en}</td>
              <td style={cell}>{g.notes}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
