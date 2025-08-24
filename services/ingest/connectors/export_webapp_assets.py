# services/ingest/connectors/export_webapp_assets.py
import os, json, time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pymongo import MongoClient

VERSION = os.getenv("WEBAPP_EXPORT_VERSION", "v1")

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI: raise SystemExit("MONGO_URI missing")

cli = MongoClient(MONGO_URI)
db  = cli["worldbank_core"]
bund= db["evidence_bundles"]

def load_country_list(path="config/europe_broad.json"):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return [c["id"] for c in j["countries"]], j

def _find(headlines, code) -> Optional[Dict[str,Any]]:
    for h in headlines:
        if h.get("code")==code and h.get("status")!="no_data":
            return h
    return None

def fmt_num(x, d=1):
    if x is None: return "—"
    return f"{x:,.{d}f}"

def make_template_narrative(bundle: Dict[str, Any]) -> Dict[str, Any]:
    c = bundle["country"]; y = bundle["year"]; H = bundle.get("headlines", [])
    facts = []
    def pick(code, d=1, suffix=""):
        h = _find(H, code)
        if not h: return "—"
        v = h.get("value"); yr = h.get("year")
        if v is None: return "—"
        facts.append({"code": code, "year": yr, "value": v})
        if code=="SP.POP.TOTL": return f"{int(round(v)):,.0f}"
        if code.endswith(".ZS") or code in {"FP.CPI.TOTL.ZG"}: return f"{v:.2f}{suffix}"
        return f"{v:,.{d}f}{suffix}"

    gdp_pc   = pick("NY.GDP.PCAP.KD", 0, "")
    infl     = pick("FP.CPI.TOTL.ZG", 2, "%")
    unemp    = pick("SL.UEM.TOTL.ZS", 2, "%")
    life     = pick("SP.DYN.LE00.IN", 2, "")
    internet = pick("IT.NET.USER.ZS", 2, "%")
    exports  = pick("NE.EXP.GNFS.ZS", 2, "%")

    # world percentiles for a couple of KPIs
    def pctl(code):
        h = _find(H, code)
        return None if not h else h.get("pctl",{}).get("world")

    p_gdp  = pctl("NY.GDP.PCAP.KD")
    p_inet = pctl("IT.NET.USER.ZS")

    summary_md = (
        f"**{c['name']}** (latest {y}) pairs GDP per capita of **{gdp_pc}** (WDI: NY.GDP.PCAP.KD)"
        + (f", placing it around the **{int(round(p_gdp))}th** world percentile" if p_gdp is not None else "")
        + f". Inflation is **{infl}** (FP.CPI.TOTL.ZG) and unemployment is **{unemp}** (SL.UEM.TOTL.ZS). "
          f"Life expectancy stands at **{life} years** (SP.DYN.LE00.IN). "
          f"Roughly **{internet}** of people are online (IT.NET.USER.ZS)"
        + (f", ~**{int(round(p_inet))}th** percentile globally" if p_inet is not None else "")
        + f". Exports account for **{exports}** of GDP (NE.EXP.GNFS.ZS)."
    )

    sections = {
        "economy_md": f"GDP per capita is **{gdp_pc}** with exports at **{exports} of GDP**. Inflation is **{infl}**.",
        "labor_md":   f"Unemployment is **{unemp}**; see participation/skills KPIs in the dashboard cards.",
        "digital_md": f"Internet usage is **{internet}**; mobile and broadband adoption are shown in connectivity cards.",
        "health_env_md": f"Life expectancy is **{life}**; air quality and CO₂ per capita appear in the environment cards.",
        "method_md": "All values come from World Bank WDI. YoY and percentiles are precomputed in our pipeline."
    }

    # persona highlights (short, data-light)
    personas = bundle.get("personas") or {}
    def ps(name): 
        s = personas.get(name,{}).get("score")
        return None if s is None else f"{s:.1f}"
    persona_highlights = {
        "job_seeker":     [f"Score **{ps('job_seeker') or '—'}**", "Tracks unemployment, LFPR, skills & momentum."],
        "entrepreneur":   [f"Score **{ps('entrepreneur') or '—'}**", "Regulation, finance depth, power & innovation."],
        "digital_nomad":  [f"Score **{ps('digital_nomad') or '—'}**", "Connectivity, stability/affordability, livability."],
        "expat_family":   [f"Score **{ps('expat_family') or '—'}**", "Health, education, safety & environment."]
    }

    return {
        "iso3": c["id"],
        "year": y,
        "summary_md": summary_md,
        "sections": sections,
        "persona_highlights": persona_highlights,
        "facts_used": facts
    }

def main():
    iso3_list, cfg = load_country_list()
    out_root = os.path.join("exports","webapp", VERSION, "countries")
    os.makedirs(out_root, exist_ok=True)

    manifest = {
        "version": VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "countries": []
    }

    for iso3 in iso3_list:
        doc = bund.find_one({"country.id": iso3}, {"_id":0})
        if not doc:
            print(f"[skip] no bundle for {iso3}")
            continue

        # write bundle
        with open(os.path.join(out_root, f"{iso3}.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)

        # write narrative (templated; swap to LLM later if you like)
        nar = make_template_narrative(doc)
        with open(os.path.join(out_root, f"{iso3}_narrative.json"), "w", encoding="utf-8") as f:
            json.dump(nar, f, ensure_ascii=False)

        manifest["countries"].append({
            "id": doc["country"]["id"],
            "name": doc["country"]["name"],
            "year": doc["year"]
        })
        print(f"[ok] {iso3}")

    # manifest
    man_path = os.path.join("exports","webapp", VERSION, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nExport complete → exports/webapp/{VERSION}/ (countries + manifest)")

if __name__ == "__main__":
    main()
