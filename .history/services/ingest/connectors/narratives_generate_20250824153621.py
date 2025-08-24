import os, re, json, time, argparse
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from pymongo import MongoClient

# -------------------- Config --------------------
load_dotenv()
MONGO_URI   = os.getenv("MONGO_URI")
PROVIDER    = (os.getenv("LLM_PROVIDER") or "").upper()  # "OPENAI" or "LLAMACPP"
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
LLAMA_URL   = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080")
LLAMA_MODEL = os.getenv("LLAMA_MODEL")  # optional
VERSION     = os.getenv("WEBAPP_EXPORT_VERSION", "v1")
OUT_DIR     = os.path.join("exports", "webapp", VERSION, "countries")
PROMPT_VERSION = "v1.0"

# -------------------- DB --------------------
if not MONGO_URI:
    raise SystemExit("MONGO_URI missing")
cli = MongoClient(MONGO_URI)
db  = cli["worldbank_core"]
bundles = db["evidence_bundles"]
countries = db["countries"]

# -------------------- Helpers --------------------
def load_europe_broad_ids() -> List[str]:
    path = "config/europe_broad.json"
    if not os.path.exists(path):
        raise SystemExit("config/europe_broad.json not found. Run build_europe_broad first.")
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return [c["id"] for c in j["countries"]]

def _find(headlines: List[Dict[str,Any]], code: str) -> Optional[Dict[str,Any]]:
    for h in headlines:
        if h.get("code")==code and h.get("status")!="no_data":
            return h
    return None

def collect_facts(b: Dict[str,Any]) -> Dict[str,Any]:
    """Pick a small, robust set of facts for narrative."""
    H = b.get("headlines", [])
    facts = {}
    def add(key, code, fmt):
        h = _find(H, code) if code else None
        if not code:
            v = b.get("year")
            yr = b.get("year")
        else:
            if not h or h.get("value") is None:
                return
            v  = h["value"]
            yr = h.get("year")
        facts[key] = {"code": code, "value": v, "year": yr, "fmt": fmt}

    # Year token (no code, just to avoid stray digits)
    add("YEAR", None, "year")

    # Core indicators (match your KPI set)
    add("GDP_PC", "NY.GDP.PCAP.KD", "usd_per_person_0")
    add("INFL",   "FP.CPI.TOTL.ZG", "pct_2")
    add("UNEMP",  "SL.UEM.TOTL.ZS", "pct_2")
    add("LIFE",   "SP.DYN.LE00.IN", "years_2")
    add("INET",   "IT.NET.USER.ZS", "pct_2")
    add("EXPGDP", "NE.EXP.GNFS.ZS", "pct_2")
    return facts

def render_fact(key: str, fact: Dict[str,Any]) -> str:
    v = fact["value"]
    if fact["fmt"] == "year":
        return str(int(v))
    if fact["fmt"] == "usd_per_person_0":
        return f"{v:,.0f} USD per person"
    if fact["fmt"] == "pct_2":
        return f"{v:.2f}%"
    if fact["fmt"] == "years_2":
        return f"{v:.2f} years"
    # fallback
    return f"{v}"

def make_template(bundle: Dict[str,Any], facts: Dict[str,Any]) -> Dict[str,Any]:
    c = bundle["country"]["name"]; y = bundle["year"]
    def get(key, default="—"):
        f = facts.get(key);  return render_fact(key,f) if f else default
    gdp = get("GDP_PC"); infl = get("INFL"); unemp = get("UNEMP")
    life = get("LIFE"); inet = get("INET"); exps = get("EXPGDP")

    summary_md = (
        f"**{c}** (latest {y}) shows GDP per capita of **{gdp}** (NY.GDP.PCAP.KD). "
        f"Inflation is **{infl}** (FP.CPI.TOTL.ZG), unemployment **{unemp}** (SL.UEM.TOTL.ZS). "
        f"Life expectancy is **{life}** (SP.DYN.LE00.IN). Internet usage stands at **{inet}** (IT.NET.USER.ZS), "
        f"and exports are **{exps} of GDP** (NE.EXP.GNFS.ZS)."
    )
    personas = bundle.get("personas") or {}
    def ps(name):
        s = personas.get(name,{}).get("score")
        return None if s is None else f"{s:.1f}"
    persona_highlights = {
        "job_seeker":    [f"Score **{ps('job_seeker') or '—'}**", "Tracks unemployment, LFPR, skills and growth momentum."],
        "entrepreneur":  [f"Score **{ps('entrepreneur') or '—'}**", "Regulation, finance depth, power reliability and innovation."],
        "digital_nomad": [f"Score **{ps('digital_nomad') or '—'}**", "Connectivity, price stability and livability."],
        "expat_family":  [f"Score **{ps('expat_family') or '—'}**", "Health, education, safety and environment."]
    }
    sections = {
        "economy_md":  f"GDP per capita **{gdp}**; exports **{exps}**; inflation **{infl}**.",
        "labor_md":    f"Unemployment **{unemp}**; participation and skills in KPI cards.",
        "digital_md":  f"Internet usage **{inet}**; see mobile and broadband adoption.",
        "health_env_md": f"Life expectancy **{life}**; environmental pressures shown in CO₂/air-quality KPIs."
    }
    facts_used = []
    for k,v in facts.items():
        if v.get("code"):
            facts_used.append({"code": v["code"], "year": v["year"], "value": v["value"]})
    return {
        "iso3": bundle["country"]["id"],
        "year": y,
        "summary_md": summary_md,
        "sections": sections,
        "persona_highlights": persona_highlights,
        "facts_used": facts_used,
        "generator": {"mode":"template","prompt_version": PROMPT_VERSION}
    }

# -------------------- LLM adapters --------------------
def openai_chat(messages: List[Dict[str,str]]) -> str:
    import requests
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type":"application/json"}
    payload = {
        "model": "gpt-4o-mini",  # small + cheap
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type":"json_object"},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def llamacpp_chat(messages: List[Dict[str,str]]) -> str:
    import requests
    # llama.cpp-compatible endpoint (LM Studio / local server)
    url = f"{LLAMA_URL}/v1/chat/completions"
    payload = {
        "model": LLAMA_MODEL or "default",
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type":"json_object"},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# -------------------- Prompt & validation --------------------
SYS = (
  "You write short, neutral country briefs for a public data dashboard. "
  "Only use the facts provided. Do NOT invent any numbers, years, or codes. "
  "When you need to reference a numeric fact, insert the exact placeholder token (e.g., <GDP_PC>) "
  "and write the indicator code in parentheses after it, e.g., (<CODE>). "
  "Do NOT include any other digits anywhere in the output (no years, counts, percents) — only placeholders. "
  "Output strict JSON with keys: summary, sections, persona_highlights."
)

USER_TMPL = """Country: {name} ({iso3})
Latest year: <YEAR>
Allowed facts (use placeholders; never write digits yourself):
{facts_list}

Your task:
- summary: 3–4 sentences, neutral tone, British English.
- sections: economy, labor, digital, health_env (1–2 short sentences each).
- persona_highlights: short bullets for job_seeker, entrepreneur, digital_nomad, expat_family (no digits).

Rules:
- Insert placeholders exactly (e.g., "<GDP_PC>") and codes in parentheses after each number, e.g., (<NY.GDP.PCAP.KD>).
- Do NOT include any digits outside placeholders.
- If a fact is missing, just omit it.

Return JSON:
{{
  "summary": "...",
  "sections": {{
    "economy": "...",
    "labor": "...",
    "digital": "...",
    "health_env": "..."
  }},
  "persona_highlights": {{
    "job_seeker": ["...", "..."],
    "entrepreneur": ["...", "..."],
    "digital_nomad": ["...", "..."],
    "expat_family": ["...", "..."]
  }}
}}
"""

def facts_block(facts: Dict[str,Any]) -> str:
    lines = []
    for key, f in facts.items():
        if key == "YEAR":
            lines.append(f"  <YEAR> = {f['value']} (latest year)")
            continue
        if f["code"] is None: 
            continue
        lines.append(f"  <{key}> = {f['value']}  code=<{f['code']}>  fmt={f['fmt']}")
    return "\n".join(lines)

def has_stray_digits(s: str) -> bool:
    # digits outside placeholder and code brackets are not allowed
    # we’ll remove tokens like <GDP_PC> and <NY.GDP.PCAP.KD> then check for \d
    cleaned = re.sub(r"<[^>]+>", "", s)
    return bool(re.search(r"\d", cleaned))

def postprocess_json(raw: str, facts: Dict[str,Any]) -> Dict[str,Any]:
    data = json.loads(raw)

    # validate no stray digits anywhere
    blob = json.dumps(data, ensure_ascii=False)
    if has_stray_digits(blob):
        raise ValueError("Stray digits found in model output")

    # replace placeholders with rendered values
    def repl(text: str) -> str:
        if not isinstance(text, str):
            return text
        out = text
        # year first
        if "YEAR" in facts:
            out = out.replace("<YEAR>", render_fact("YEAR", facts["YEAR"]))
        for key, f in facts.items():
            if key == "YEAR": 
                continue
            token = f"<{key}>"
            code  = f"<{f['code']}>" if f.get("code") else ""
            if token in out:
                out = out.replace(token, render_fact(key,f))
            if code and code in out:
                out = out.replace(code, f['code'])
        return out

    # walk and replace
    data["summary"] = repl(data.get("summary",""))
    sec = data.get("sections",{})
    for k in list(sec.keys()):
        sec[k] = repl(sec[k])
    ph = data.get("persona_highlights",{})
    for k,v in list(ph.items()):
        if isinstance(v, list):
            ph[k] = [repl(x) for x in v]
        else:
            ph[k] = repl(v)

    return data
# --- additions: polarity, labels, links, callouts ---
LOWER_IS_BETTER = {
    "FP.CPI.TOTL.ZG",   # inflation
    "SL.UEM.TOTL.ZS",   # unemployment
    "EN.ATM.PM25.MC.M3",
    "EN.ATM.CO2E.PC",
    "PA.NUS.PPPC.RF",
}

LABELS = {
    "NY.GDP.PCAP.KD": "GDP per capita",
    "FP.CPI.TOTL.ZG": "Inflation",
    "SL.UEM.TOTL.ZS": "Unemployment",
    "SP.DYN.LE00.IN": "Life expectancy",
    "IT.NET.USER.ZS": "Internet usage",
    "NE.EXP.GNFS.ZS": "Exports share",
    "EN.ATM.PM25.MC.M3": "PM2.5",
    "EN.ATM.CO2E.PC": "CO₂ per capita",
    "PA.NUS.PPPC.RF": "Price level (PPP)"
}

def source_link(code: str) -> str:
    return f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json"

def world_percentile(bundle: Dict[str, Any], code: str) -> Optional[float]:
    for h in bundle.get("headlines", []):
        if h.get("code") == code and h.get("status") != "no_data":
            return (h.get("pctl") or {}).get("world")
    return None

def make_callouts(bundle: Dict[str, Any], codes: List[str]) -> Dict[str, List[Dict[str,str]]]:
    strengths, watchouts = [], []
    for code in codes:
        if not code: 
            continue
        p = world_percentile(bundle, code)
        if p is None:
            continue
        label = LABELS.get(code, code)
        if code in LOWER_IS_BETTER:
            # lower is better → good if <=25th; watchout if >=75th
            if p <= 25:
                strengths.append({"code": code, "label": f"Favourable {label.lower()}"})
            elif p >= 75:
                watchouts.append({"code": code, "label": f"High {label.lower()}"})
        else:
            # higher is better → good if >=75th; watchout if <=25th
            if p >= 75:
                strengths.append({"code": code, "label": f"Strong {label.lower()}"})
            elif p <= 25:
                watchouts.append({"code": code, "label": f"Low {label.lower()}"})
    return {"strengths": strengths, "watchouts": watchouts}


def build_narrative(bundle: Dict[str,Any]) -> Dict[str,Any]:
    facts = collect_facts(bundle)
    # if no provider configured -> template
    if PROVIDER not in {"OPENAI","LLAMACPP"}:
        return make_template(bundle, facts)

    messages = [
        {"role":"system","content": SYS},
        {"role":"user","content": USER_TMPL.format(
            name=bundle["country"]["name"],
            iso3=bundle["country"]["id"],
            facts_list=facts_block(facts)
        )}
    ]
    try:
        if PROVIDER == "OPENAI":
            if not OPENAI_KEY:
                raise RuntimeError("OPENAI_API_KEY missing")
            raw = openai_chat(messages)
        else:
            raw = llamacpp_chat(messages)
        data = postprocess_json(raw, facts)
        facts_used = []
        for k,f in facts.items():
            if f.get("code"):
                facts_used.append({"code": f["code"], "year": f["year"], "value": f["value"]})
        return {
            "iso3": bundle["country"]["id"],
            "year": bundle["year"],
            "summary_md": data["summary"],
            "sections": {
                "economy_md": data["sections"].get("economy",""),
                "labor_md": data["sections"].get("labor",""),
                "digital_md": data["sections"].get("digital",""),
                "health_env_md": data["sections"].get("health_env",""),
            },
            "persona_highlights": data.get("persona_highlights",{}),
            "facts_used": facts_used,
            "generator": {
                "mode": PROVIDER.lower(),
                "prompt_version": PROMPT_VERSION
            }
        }
    except Exception as e:
        # Fallback to template
        print(f"[warn] LLM generation failed: {e} -> using template")
        return make_template(bundle, facts)

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Generate country narratives with placeholder-guarded LLM (or template fallback).")
    ap.add_argument("--iso3", nargs="*", help="ISO3 codes (default: europe_broad list)")
    ap.add_argument("--use-europe-broad", action="store_true", help="Use config/europe_broad.json as source list")
    args = ap.parse_args()

    if args.iso3:
        iso3_list = [x.upper() for x in args.iso3]
    elif args.use_europe_broad:
        iso3_list = load_europe_broad_ids()
    else:
        raise SystemExit("Specify --iso3 ... or --use-europe-broad")

    os.makedirs(OUT_DIR, exist_ok=True)
    for iso3 in iso3_list:
        doc = bundles.find_one({"country.id": iso3}, {"_id":0})
        if not doc:
            print(f"[skip] no bundle for {iso3}")
            continue
        nar = build_narrative(doc)
        path = os.path.join(OUT_DIR, f"{iso3}_narrative.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nar, f, ensure_ascii=False, indent=2)
        print(f"[ok] {iso3} -> {path}")

    print(f"\nDone. Copy exports/webapp/{VERSION}/countries/* into your webapp /public/data/{VERSION}/countries/")

if __name__ == "__main__":
    main()
