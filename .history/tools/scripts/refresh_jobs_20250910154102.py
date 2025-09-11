import os, sys, time, hashlib, json
from datetime import datetime
from typing import Any, Dict, List, Optional
import requests
import yaml
from dateutil import parser as dateparser
from pymongo import MongoClient, UpdateOne

# ---------- ENV / CONFIG ----------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT, "config", "providers.yaml")

APP_UA = os.environ.get("APP_USER_AGENT", "sufoniq-jobs/1.0")
TIMEOUT = 20

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MONGO_URI = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
if not MONGO_URI:
    print("ERROR: Set MONGO_URI (or MONGODB_URI) in your .env")
    sys.exit(1)

client = MongoClient(MONGO_URI)
db = client["jobsdb"]  # change if you want
jobs_col = db["jobs"]

# ---------- HELPERS ----------

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def iso(dt: Optional[str]) -> Optional[str]:
    if not dt:
        return None
    try:
        return dateparser.parse(dt).astimezone().astimezone(tz=None).isoformat()
    except Exception:
        return None

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def make_id(source: str, source_id: Optional[str], url: Optional[str], title: str, company: str, loc: str, posted_at: Optional[str]) -> str:
    if source_id:
        return sha1(f"{source}:{source_id}")
    if url:
        return sha1(f"{source}:{url}")
    key = f"{source}:{title.strip().lower()}|{company.strip().lower()}|{loc.strip().lower()}|{(posted_at or '')[:10]}"
    return sha1(key)

def upsert_normalized(batch: List[Dict[str, Any]]) -> Dict[str, int]:
    if not batch:
        return {"inserted":0,"updated":0}
    ops = []
    for doc in batch:
        ops.append(
            UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
        )
    res = jobs_col.bulk_write(ops, ordered=False)
    return {
        "inserted": res.upserted_count,
        "updated": res.modified_count
    }

def get_session(headers: Optional[Dict[str,str]]=None) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": APP_UA})
    if headers:
        s.headers.update(headers)
    return s

# ---------- NORMALIZATION (Unified Schema v1) ----------

def norm_base(
    source: str,
    source_id: Optional[str],
    title: str,
    company_name: Optional[str],
    apply_url: Optional[str],
    description_html: Optional[str],
    posted_at: Optional[str],
    employment_type: Optional[str],
    tags: Optional[List[str]],
    location_type: Optional[str],
    locations: Optional[List[Dict[str,Any]]],
    salary: Optional[Dict[str,Any]],
    raw: Dict[str, Any],
) -> Dict[str, Any]:
    locs = locations or []
    comp = {"name": company_name or None, "domain": None}
    doc = {
        "_id": make_id(source, source_id, apply_url, title or "", company_name or "", (locs[0].get("country") if locs else "") or "", posted_at),
        "source": source,
        "source_id": source_id,
        "title": title or "",
        "company": comp,
        "locations": locs,
        "employment_type": employment_type,
        "remote": (location_type == "remote"),
        "salary": salary or {"min": None, "max": None, "currency": None, "period": None},
        "description_html": description_html or None,
        "tags": tags or [],
        "apply_url": apply_url or None,
        "posted_at": iso(posted_at),
        "fetched_at": now_iso(),
        "duplicate_of": None,
        "raw": raw,
    }
    return doc

# ---------- CONNECTORS (Fetch + Map) ----------

def fetch_remotive() -> List[Dict[str,Any]]:
    s = get_session()
    url = "https://remotive.com/api/remote-jobs"
    r = s.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    out = []
    for j in data.get("jobs", []):
        title = j.get("title")
        company = j.get("company_name")
        apply_url = j.get("url")
        desc = j.get("description")
        posted = j.get("publication_date")
        job_type = (j.get("job_type") or "").replace(" ", "_").lower() or None
        loc_str = j.get("candidate_required_location") or ""
        locations = [{"type":"remote" if "remote" in loc_str.lower() else "onsite", "city": None, "region": None, "country": None, "lat": None, "lon": None}]
        out.append(norm_base(
            "remotive", str(j.get("id")) if j.get("id") else None, title, company, apply_url, desc, posted, job_type, j.get("tags") or [],
            "remote" if "remote" in loc_str.lower() else "onsite", locations, None, j
        ))
    return out

def fetch_arbeitnow(pages:int=1) -> List[Dict[str,Any]]:
    s = get_session()
    out = []
    for p in range(1, pages+1):
        url = "https://www.arbeitnow.com/api/job-board-api"
        if p > 1:
            url += f"?page={p}"
        r = s.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for j in data.get("data", []):
            title = j.get("title")
            company = j.get("company_name")
            apply_url = j.get("url") or j.get("slug")
            desc = j.get("description")
            posted = j.get("created_at")
            loc = j.get("location") or ""
            locations = [{"type":"remote" if (j.get("remote") is True or "remote" in loc.lower()) else "onsite",
                          "city": None, "region": None, "country": None, "lat": None, "lon": None}]
            out.append(norm_base("arbeitnow", j.get("slug"), title, company, apply_url, desc, posted,
                                 None, [], "remote" if (j.get("remote") or "remote" in loc.lower()) else "onsite", locations, None, j))
    return out

def fetch_jobicy(count:int=100, params:Optional[Dict[str,str]]=None) -> List[Dict[str,Any]]:
    headers = {"Accept": "application/json"}
    s = get_session(headers=headers)
    url = f"https://jobicy.com/api/v2/remote-jobs?count={count}"
    if params:
        for k,v in params.items():
            if v:
                url += f"&{k}={requests.utils.quote(str(v))}"
    r = s.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    out = []
    for j in data.get("jobs", []):
        title = j.get("jobTitle")
        company = j.get("companyName")
        apply_url = j.get("url")
        desc = j.get("jobDescription")
        posted = j.get("pubDate") or j.get("date")
        job_types = j.get("jobType") or []
        job_type = (job_types[0].lower().replace(" ", "_") if job_types else None)
        geo = j.get("jobGeo") or ""
        locations = [{"type":"remote" if "remote" in geo.lower() else "onsite", "city": None, "region": None, "country": None, "lat": None, "lon": None}]
        out.append(norm_base("jobicy", j.get("id") or j.get("slug"), title, company, apply_url, desc, posted,
                             job_type, j.get("jobTags") or [], "remote" if "remote" in geo.lower() else "onsite", locations, None, j))
    return out

def fetch_usajobs(results_per_page:int=500, max_pages:int=3) -> List[Dict[str,Any]]:
    ua = os.environ.get("USAJOBS_USER_AGENT")
    key = os.environ.get("USAJOBS_API_KEY")
    if not ua or not key:
        return []
    headers = {"User-Agent": ua, "Authorization-Key": key, "Host":"data.usajobs.gov", "Accept":"application/json"}
    s = get_session(headers=headers)
    params = {"ResultsPerPage": results_per_page, "Page": 1}
    out = []
    # first page
    r = s.get("https://data.usajobs.gov/api/search", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    number_of_pages = int(data.get("SearchResult", {}).get("NumberOfPages") or 1)
    pages = min(number_of_pages, max_pages)
    def map_items(payload):
        for item in payload.get("SearchResult", {}).get("SearchResultItems", []):
            mid = item.get("MatchedObjectId")
            d = item.get("MatchedObjectDescriptor", {})
            title = d.get("PositionTitle")
            company = d.get("OrganizationName")
            posted = d.get("PublicationStartDate")
            apply_urls = d.get("ApplyURI") or []
            apply_url = apply_urls[0] if apply_urls else None
            locs = []
            for L in (d.get("PositionLocation") or []):
                locs.append({"type":"onsite", "city":L.get("LocationName"), "region": None, "country": L.get("CountryCode"), "lat": None, "lon": None})
            sched = d.get("PositionSchedule") or []
            job_type = (sched[0].get("Name").lower().replace(" ", "_") if sched else None)
            desc = d.get("UserArea", {}).get("Details", {}).get("JobSummary") or d.get("QualificationSummary")
            yield norm_base("usajobs", mid, title, company, apply_url, desc, posted, job_type, [], "onsite", locs, None, item)
    out.extend(list(map_items(data)))
    # remaining pages
    for page in range(2, pages+1):
        params["Page"] = page
        r = s.get("https://data.usajobs.gov/api/search", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        out.extend(list(map_items(r.json())))
    return out

def fetch_jooble(locations: List[str], keywords: str="") -> List[Dict[str,Any]]:
    key = os.environ.get("JOOBLE_API_KEY")
    if not key:
        return []
    s = get_session(headers={"Accept":"application/json","Content-Type":"application/json"})
    out = []
    for loc in locations:
        url = f"https://jooble.org/api/{key}"
        body = {"keywords": keywords, "location": loc}
        r = s.post(url, data=json.dumps(body), timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for j in data.get("jobs", []):
            title = j.get("title")
            company = None
            apply_url = j.get("link")
            desc = j.get("snippet")
            posted = None
            salary = {"min": None, "max": None, "currency": None, "period": None}
            if j.get("salary"):
                salary["min"] = None
                salary["max"] = None
            locations = [{"type":"onsite", "city": None, "region": None, "country": None, "lat": None, "lon": None}]
            out.append(norm_base("jooble", j.get("id") or j.get("link"), title, company, apply_url, desc, posted, None, [], "onsite", locations, salary, j))
        time.sleep(0.3)  # polite
    return out

def fetch_adzuna(countries: List[str], max_pages:int=5) -> List[Dict[str,Any]]:
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        return []
    s = get_session(headers={"Accept":"application/json"})
    base = "https://api.adzuna.com/v1/api/jobs"
    out = []
    for country in countries:
        for page in range(1, max_pages+1):
            url = f"{base}/{country}/search/{page}"
            params = {
                "app_id": app_id,
                "app_key": app_key,
                "results_per_page": 50,
            }
            r = s.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 404:
                break
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            for j in results:
                title = j.get("title")
                company = (j.get("company") or {}).get("display_name")
                apply_url = j.get("redirect_url")
                desc = j.get("description")
                posted = j.get("created")
                salary = {"min": j.get("salary_min"), "max": j.get("salary_max"), "currency": None, "period": None}
                loc = j.get("location") or {}
                # location.area might be ["Austria","Vienna","Vienna"]
                areas = loc.get("area") or []
                country_code = None
                if areas:
                    country_code = None  # Adzuna areas are names; you can map later if you want
                locs = [{"type":"onsite", "city": areas[1] if len(areas)>1 else None, "region": None, "country": None, "lat": j.get("latitude"), "lon": j.get("longitude")}]
                job_type = (j.get("contract_type") or "").lower() or None
                out.append(norm_base("adzuna", str(j.get("id")), title, company, apply_url, desc, posted, job_type, [], "onsite", locs, salary, j))
            time.sleep(0.2)  # polite
    return out

# ---------- ORCHESTRATOR ----------

def main():
    enabled = {p["name"]: p for p in CONFIG.get("providers", []) if p.get("enabled")}
    totals = {}

    # Remotive
    if "remotive" in enabled:
        batch = fetch_remotive()
        totals["remotive"] = upsert_normalized(batch)

    # Arbeitnow
    if "arbeitnow" in enabled:
        pages = (enabled["arbeitnow"].get("pagination") or {}).get("max_pages_per_run", 1)
        batch = fetch_arbeitnow(pages=pages)
        totals["arbeitnow"] = upsert_normalized(batch)

    # Jobicy
    if "jobicy" in enabled:
        qd = (enabled["jobicy"].get("endpoints") or {}).get("list", {}).get("query_defaults", {})
        count = int(qd.get("count", 100)) if isinstance(qd, dict) else 100
        batch = fetch_jobicy(count=count)
        totals["jobicy"] = upsert_normalized(batch)

    # USAJOBS
    if "usajobs" in enabled:
        qd = (enabled["usajobs"].get("endpoints") or {}).get("list", {}).get("query_defaults", {})
        rpp = int(qd.get("ResultsPerPage", 500)) if isinstance(qd, dict) else 500
        # Limit initial pages to be polite; you can raise later
        batch = fetch_usajobs(results_per_page=rpp, max_pages=3)
        totals["usajobs"] = upsert_normalized(batch)

    # Jooble
    if "jooble" in enabled:
        locs = enabled["jooble"].get("locations") or ["Austria","Germany","United Kingdom","United States"]
        batch = fetch_jooble(locs, keywords="")
        totals["jooble"] = upsert_normalized(batch)

    # Adzuna
    if "adzuna" in enabled:
        countries = enabled["adzuna"].get("countries") or ["at"]
        max_pages = (enabled["adzuna"].get("pagination") or {}).get("max_pages_per_run", 3)
        batch = fetch_adzuna(countries=countries, max_pages=max_pages)
        totals["adzuna"] = upsert_normalized(batch)

    print(json.dumps({"ok": True, "totals": totals}, indent=2))

if __name__ == "__main__":
    main()
python scripts/refresh_jobs.py