import React, { useEffect, useState } from "react";

type Headline = {
  code: string;
  year: number;
  value: number | null;
  unit?: string | null;
  yoy?: { pct?: number | null; pp?: number | null };
  pctl: { world?: number | null; region?: number | null; income?: number | null };
  trend: { year: number; value: number | null }[];
};

type Personas = {
  job_seeker?: { score?: number | null };
  entrepreneur?: { score?: number | null };
  digital_nomad?: { score?: number | null };
  expat_family?: { score?: number | null };
  [k: string]: any;
};

type Bundle = {
  country: { id: string; name: string; region?: string; income?: string; iso2?: string };
  year: number;
  headlines: Headline[];
  personas?: Personas;
};

function fmt(v: number | null | undefined, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const abs = Math.abs(v);
  // compact formatting
  if (abs >= 1_000_000_000) return (v / 1_000_000_000).toFixed(digits) + "B";
  if (abs >= 1_000_000) return (v / 1_000_000).toFixed(digits) + "M";
  if (abs >= 1_000) return (v / 1_000).toFixed(digits) + "k";
  return v.toFixed(digits);
}

export default function CountryView({ iso3 = "AUT" }: { iso3?: string }) {
  const [data, setData] = useState<Bundle | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/country/${iso3}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch(e => setErr(String(e)))
      .finally(() => setLoading(false));
  }, [iso3]);

  if (loading) return <div className="p-6 text-sm opacity-70">loading…</div>;
  if (err) return <div className="p-6 text-sm text-red-600">error: {err}</div>;
  if (!data) return null;

  const top = data.headlines?.slice(0, 6) ?? [];
  const personas = data.personas ?? {};

  return (
    <div className="p-6 space-y-6">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold">{data.country.name}</h1>
          <p className="text-sm text-gray-500">
            {data.country.id} · {data.country.region} · {data.country.income} · latest {data.year}
          </p>
        </div>
      </header>

      {/* KPIs */}
      <div className="grid md:grid-cols-3 gap-4">
        {top.map(h => (
          <div key={h.code} className="rounded-2xl border p-4 shadow-sm">
            <div className="text-xs uppercase tracking-wide text-gray-500">{h.code}</div>
            <div className="mt-1 text-2xl font-medium">
              {fmt(h.value, h.code === "FP.CPI.TOTL.ZG" ? 2 : 1)}{h.unit ? ` ${h.unit}` : ""}
            </div>
            <div className="mt-2 text-xs text-gray-500">
              {h.yoy?.pp !== undefined && h.yoy?.pp !== null ? (
                <>YoY: {fmt(h.yoy.pp, 2)} pp</>
              ) : h.yoy?.pct !== undefined && h.yoy?.pct !== null ? (
                <>YoY: {fmt(h.yoy.pct, 2)}%</>
              ) : (
                <>YoY: —</>
              )}
              <span className="mx-2">·</span>
              World pctl: {h.pctl?.world !== undefined && h.pctl?.world !== null ? h.pctl.world.toFixed(0) : "—"}
            </div>
            {/* mini trend */}
            <div className="mt-3 h-10 w-full bg-gray-50 rounded-md overflow-hidden">
              {/* simple sparkline using inline SVG */}
              <svg viewBox="0 0 100 30" className="w-full h-full">
                {(() => {
                  const pts = (h.trend ?? []).filter(t => t.value !== null) as {year:number; value:number}[];
                  if (pts.length < 2) return null;
                  const vs = pts.map(p => p.value);
                  const min = Math.min(...vs), max = Math.max(...vs);
                  const rng = max - min || 1;
                  const xs = pts.map((p, i) => (i / (pts.length - 1)) * 100);
                  const ys = pts.map(p => 30 - ((p.value - min) / rng) * 28 - 1); // padding
                  const d = xs.map((x, i) => `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${ys[i].toFixed(2)}`).join(" ");
                  return <path d={d} fill="none" stroke="currentColor" strokeWidth="1.5" />;
                })()}
              </svg>
            </div>
          </div>
        ))}
      </div>

      {/* Personas */}
      <div className="grid md:grid-cols-4 gap-4">
        {[
          ["job_seeker", "Job Seeker"],
          ["entrepreneur", "Entrepreneur"],
          ["digital_nomad", "Digital Nomad"],
          ["expat_family", "Expat Family"],
        ].map(([key, label]) => {
          const s = (personas as any)[key]?.score as number | undefined;
          return (
            <div key={key} className="rounded-2xl border p-4 shadow-sm">
              <div className="text-sm text-gray-600">{label}</div>
              <div className="mt-1 text-3xl font-semibold">{s != null ? s.toFixed(1) : "—"}</div>
              <div className="text-xs text-gray-500">0–100 (world percentile-based)</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
