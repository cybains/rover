from __future__ import annotations
DATA = Path(__file__).resolve().parents[2] / "data"
DOCS = DATA / "docs.jsonl"
STATE = DATA / "state.last_hash.json"
DATA.mkdir(parents=True, exist_ok=True)


CONNECTORS = {
"rest_countries": RestCountries(),
"crawl": SimpleCrawler(),
"pdf": PdfIntake(),
}




def load_state() -> Dict[str, str]:
if STATE.exists():
return json.loads(STATE.read_text(encoding="utf-8"))
return {}




def save_state(state: Dict[str, str]):
STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")




def append_docs(docs):
with DOCS.open("a", encoding="utf-8") as f:
for nd in docs:
f.write(json.dumps(nd.to_dict(), ensure_ascii=False) + "\n")




def main():
ap = argparse.ArgumentParser(description="Run rover ingestion")
ap.add_argument("connector", choices=CONNECTORS.keys())
ap.add_argument("--url", help="for crawl connector")
ap.add_argument("--pdf", help="for pdf connector (file path)")
args = ap.parse_args()


state = load_state()
c = CONNECTORS[args.connector]


if args.connector == "crawl":
payloads = c.fetch(url=args.url) if args.url else c.fetch(url="https://example.com")
elif args.connector == "pdf":
if not args.pdf:
ap.error("--pdf path is required for pdf connector")
payloads = c.fetch(path=args.pdf)
else:
payloads = c.fetch()


changed = []
for p in payloads:
nd = c.normalize(p)
sid = nd.core.source_id
chash = nd.core.content_hash or ""
if state.get(sid) == chash:
continue # skip unchanged
changed.append(nd)
state[sid] = chash


if changed:
append_docs(changed)
save_state(state)
print(f"ingested/updated {len(changed)} docs → {DOCS}")
else:
print("no changes detected; nothing to write")




if __name__ == "__main__":
main()