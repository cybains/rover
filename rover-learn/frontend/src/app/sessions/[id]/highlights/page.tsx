'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { getHighlights, getGlossary } from '../../../../lib/api';

interface Seg {
  _id: string;
  textSrc?: string;
  textEn?: string;
  idxStart: number;
  glossaryHits?: string[];
}

export default function Highlights({ params }: { params: { id: string } }) {
  const { id } = params;
  const [data, setData] = useState<
    { questions: Seg[]; answers: Seg[]; bookmarks: Seg[]; glossary: Seg[] } | null
  >(null);
  const [glossaryMap, setGlossaryMap] = useState<Record<string, string>>({});
  const router = useRouter();

  useEffect(() => {
    getHighlights(id).then(setData);
    getGlossary().then((list) => {
      const map: Record<string, string> = {};
      list.forEach((t: any) => {
        map[t.id] = t.en;
      });
      setGlossaryMap(map);
    });
  }, [id]);

  const jump = (idx: number) => {
    router.push(`/sessions/${id}#seg-${idx}`);
  };

  if (!data) return <div>Loading...</div>;

  const renderGloss = (text: string, hits?: string[]) => {
    if (!hits || !hits.length) return text;
    let html = text;
    hits.forEach((h) => {
      const term = glossaryMap[h];
      if (!term) return;
      const re = new RegExp(`\\b${term}\\b`, 'g');
      html = html.replace(
        re,
        `<span style="text-decoration:underline dotted" title="${term}">${term}</span>`
      );
    });
    return <span dangerouslySetInnerHTML={{ __html: html }} />;
  };

  const section = (
    title: string,
    items: Seg[],
    getText: (s: Seg) => JSX.Element | string
  ) => (
    <div style={{ marginBottom: '1rem' }}>
      <h2>{title}</h2>
      <ul>
        {items.map((s) => (
          <li key={s._id}>
            {getText(s)}{' '}
            <button onClick={() => jump(s.idxStart)}>Jump</button>
          </li>
        ))}
      </ul>
    </div>
  );

  return (
    <div style={{ padding: '1rem' }}>
      <h1>Highlights</h1>
      {section('Questions', data.questions, (s) => s.textSrc || '')}
      {section('Answers', data.answers, (s) => s.textSrc || '')}
      {section('Bookmarks', data.bookmarks, (s) => s.textSrc || '')}
      {section('Glossary Hits', data.glossary, (s) => renderGloss(s.textEn || '', s.glossaryHits))}
    </div>
  );
}
