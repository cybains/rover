# services/ingest/connectors/cli_country_profile.py
import os, sys, argparse, requests
from typing import Any, Dict, Optional

DEFAULT_API = os.getenv("ROVER_API_BASE", "http://127.0.0.1:8000")

def fmt_num(v: Optional[float], decimals=2):
    if v is None: return "—"
    try:
        return f"{v:,.{decimals}f}"
    except Exception:
        return str(v)

def fmt_value(code: str, v: Optional[float]) -> str:
    if v is None: return "—"
    # light formatting heuristics
    if code in {"SP.POP.TOTL"}:          # population
        return f"{int(round(v)):,.0f}"
    if code.endswith(".ZS") or code in {"FP.CPI.TOTL.ZG"}:  # percentages/rates
        return f"{v:.2f}"
    return fmt_num(v, 2)

def fmt_yoy(yoy: Dict[str, Any]) -> str:
    if not yoy: return "—"
    if "pp" in yoy and yoy["pp"] is not None:
        s = f"{yoy['pp']:+.2f} pp"
        return s
    if "pct" in yoy and yoy["pct"] is not None:
        s = f"{yoy['pct']:+.2f}%"
        return s
    return "—"

def get_bundle(base: str, iso3: str, year: Optional[int]) -> Dict[str, Any]:
    url = f"{base}/api/country/{iso3.upper()}" + (f"/{year}" if year else "")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def print_profile(b: Dict[str, Any], limit: int):
    c = b["country"]; year = b.get("year")
    print(f"\n{c['name']} ({c['id']}) — {c.get('region','?')} · {c.get('income','?')}  |  latest year: {year}")
    print("-" * 80)

    # personas
    personas = (b.get("personas") or {})
    if personas:
        print("Personas (0–100, higher=better):")
        row = []
        for key, label in [("job_seeker","Job Seeker"),("entrepreneur","Entrepreneur"),
                           ("digital_nomad","Digital Nomad"),("expat_family","Expat Family")]:
            score = personas.get(key, {}).get("score")
            row.append(f"{label}: {('—' if score is None else f'{score:.1f}')}")
        print("  " + "  |  ".join(row))
        print("-" * 80)

    # headlines table
    heads = b.get("headlines") or []
    if limit and len(heads) > limit:
        heads = heads[:limit]
    print(f"{'KPI (code)':<28} {'Year':<6} {'Value':>14} {'YoY':>12} {'Pctl W/R/I':>16}")
    print("-" * 80)
    for h in heads:
        code = h["code"]; y = h.get("year")
        val = fmt_value(code, h.get("value"))
        yoy = fmt_yoy(h.get("yoy") or {})
        p = h.get("pctl") or {}
        pw = p.get("world"); pr = p.get("region"); pi = p.get("income")
        pstr = f"{'' if pw is None else int(round(pw))}/{'' if pr is None else int(round(pr))}/{'' if pi is None else int(round(pi))}"
        print(f"{code:<28} {y:<6} {val:>14} {yoy:>12} {pstr:>16}")

    print("-" * 80)
    print("Notes: Pctl W/R/I = World / Region / Income-group percentile (0–100).")
    print("      YoY shows % for growth series and percentage points (pp) for rates.\n")

def main():
    ap = argparse.ArgumentParser(description="Print a country profile (terminal).")
    ap.add_argument("iso3", help="ISO-3 code, e.g., AUT or PRT")
    ap.add_argument("--year", type=int, help="Specific year (defaults to latest)")
    ap.add_argument("--base", default=DEFAULT_API, help="API base (default: %(default)s)")
    ap.add_argument("--all", action="store_true", help="Show all KPIs (not just the first 12)")
    args = ap.parse_args()

    bundle = get_bundle(args.base, args.iso3, args.year)
    limit = 0 if args.all else 12
    print_profile(bundle, limit)

if __name__ == "__main__":
    main()
