import './globals.css';
import Link from 'next/link';
import { ReactNode } from 'react';

export const metadata = {
  title: 'Rover Learn',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div style={{ display: 'flex', height: '100vh' }}>
          <aside style={{ width: '200px', background: '#f0f0f0', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <Link href="/">Live</Link>
            <Link href="/sessions">Sessions</Link>
            <Link href="/glossary">Glossary</Link>
          </aside>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <div style={{ height: '40px', background: '#ddd', padding: '0.5rem' }}>Status Bar</div>
            <main style={{ flex: 1, overflow: 'auto' }}>{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
