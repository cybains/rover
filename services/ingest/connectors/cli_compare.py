import os, argparse, requests
from typing import Dict, Any, Optional

DEFAULT_API = os.getenv("ROVER_API_BASE", "http://127.0.0.1:8000")
HEADLINE_ORDER = [
    "NY.GDP.PCAP.KD","FP.CPI.TOTL.ZG","NE.EXP.GNFS.ZS","SL.UEM.TOTL.ZS",
    "SL.TLF.CACT.ZS","SP.POP.TOTL","SP.DYN.LE00.IN","SE.TER.ENRR",
    "SE.ADT.LITR.ZS","IT.NET.USER.ZS","IT.CEL.SETS.P2","EN.ATM.CO2E.PC"
]

def get_bundle(base: str, iso3: str) -> Dict[str, Any]:
    r = requests.get(f"{base}/api/country/{iso3.upper()}", timeout=30)
    r.raise_for_status()
    return r.json()

def fmt_num(v: Optional[float], digits=2):
    if v is None: return "—"
    return f"{v:,.{digits}f}"

def fmt_value(code: str, v: Optional[float]) -> str:
    if v is None: return "—"
    if code == "SP.POP.TOTL": return f"{int(round(v)):,.0f}"
    if code.endswith(".ZS") or code in {"FP.CPI.TOTL.ZG"}: return f"{v:.2f}"
    return fmt_num(v, 2)

def as_map(heads):
    return {h["code"]: h for h in heads if "code" in h}

def main():
    ap = argparse.ArgumentParser(description="Compare two countries (terminal).")
    ap.add_argument("left", help="ISO-3, e.g., AUT")
    ap.add_argument("right", help="ISO-3, e.g., PRT")
    ap.add_argument("--base", default=DEFAULT_API, help="API base")
    args = ap.parse_args()

    a = get_bundle(args.base, args.left)
    b = get_bundle(args.base, args.right)

    A, B = as_map(a.get("headlines", [])), as_map(b.get("headlines", []))

    print(f"\nCompare: {a['country']['name']} ({a['country']['id']}, {a['year']})  vs  "
          f"{b['country']['name']} ({b['country']['id']}, {b['year']})")
    print("-" * 100)
    print(f"{'KPI (code)':<28} {'A value':>14} {'B value':>14} {'Δ B–A':>14} {'A pctl':>8} {'B pctl':>8}")
    print("-" * 100)

    for code in HEADLINE_ORDER:
        ha, hb = A.get(code), B.get(code)
        if not ha or not hb:  # skip if either missing
            continue
        va = ha.get("value"); vb = hb.get("value")
        pa = ha.get("pctl", {}).get("world"); pb = hb.get("pctl", {}).get("world")
        delta = None if (va is None or vb is None) else (vb - va)

        print(f"{code:<28} {fmt_value(code, va):>14} {fmt_value(code, vb):>14} "
              f"{('—' if delta is None else fmt_num(delta)):>14} "
              f"{('—' if pa is None else str(int(round(pa)))):>8} "
              f"{('—' if pb is None else str(int(round(pb)))):>8}")

    # personas
    pa = a.get("personas") or {}; pb = b.get("personas") or {}
    print("-" * 100)
    print(f"{'Persona':<20} {'A':>8} {'B':>8} {'Δ B–A':>10}")
    print("-" * 100)
    for key, label in [("job_seeker","Job Seeker"),("entrepreneur","Entrepreneur"),
                       ("digital_nomad","Digital Nomad"),("expat_family","Expat Family")]:
        sa = pa.get(key, {}).get("score"); sb = pb.get(key, {}).get("score")
        d  = None if (sa is None or sb is None) else (sb - sa)
        print(f"{label:<20} {('—' if sa is None else f'{sa:.1f}'):>8} "
              f"{('—' if sb is None else f'{sb:.1f}'):>8} "
              f"{('—' if d is None else f'{d:+.1f}'):>10}")

    print()

if __name__ == "__main__":
    main()
