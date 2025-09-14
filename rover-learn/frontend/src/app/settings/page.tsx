'use client';
import { useEffect, useState } from 'react';
import { getSettings, updateSettings, getStorageStatus, purgeSessions } from '../../lib/api';

export default function SettingsPage() {
  const [settings, setSettings] = useState<any>({
    asrForceLang: '',
    chunkMs: 500,
    targetLang: 'EN',
    exportDefaults: { format: 'zip' },
  });
  const [storage, setStorage] = useState<any | null>(null);

  useEffect(() => {
    getSettings().then(setSettings);
    getStorageStatus().then(setStorage);
  }, []);

  const save = async () => {
    const s = await updateSettings(settings);
    setSettings(s);
  };

  const purge = async () => {
    const before = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
    await purgeSessions(before);
    const st = await getStorageStatus();
    setStorage(st);
  };

  return (
    <div style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <h2>Settings</h2>
      <label>
        ASR force language:
        <input
          value={settings.asrForceLang}
          onChange={(e) => setSettings({ ...settings, asrForceLang: e.target.value })}
        />
      </label>
      <label>
        ASR chunk ms:
        <input
          type="number"
          value={settings.chunkMs}
          onChange={(e) => setSettings({ ...settings, chunkMs: parseInt(e.target.value) })}
        />
      </label>
      <label>
        Target language:
        <input
          value={settings.targetLang}
          onChange={(e) => setSettings({ ...settings, targetLang: e.target.value })}
        />
      </label>
      <label>
        Export format:
        <input
          value={settings.exportDefaults?.format || ''}
          onChange={(e) =>
            setSettings({
              ...settings,
              exportDefaults: { ...settings.exportDefaults, format: e.target.value },
            })
          }
        />
      </label>
      <button onClick={save}>Save</button>
      <div>
        <h3>Hotkeys</h3>
        <ul>
          <li>Alt+S – Start/Stop</li>
          <li>Alt+P – Pause/Resume</li>
          <li>Alt+B – Bookmark last</li>
          <li>Alt+L – Jump to Live</li>
        </ul>
      </div>
      {storage && (
        <div>
          <h3>Storage</h3>
          <p>
            Total:{' '}
            {(storage.totalBytes / 1024 / 1024).toFixed(1)} MB{' '}
            {storage.overLimit && '(Over limit!)'}
          </p>
          <button onClick={purge}>Delete sessions older than 30 days</button>
        </div>
      )}
    </div>
  );
}

