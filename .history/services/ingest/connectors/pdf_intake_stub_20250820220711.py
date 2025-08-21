from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import json

# pip install pypdf
from pypdf import PdfReader

from ..schemas import make_core, NormalizedDoc
from ..facets import facet_pdf
from ..utils_id import source_id_pdf
from ..utils_hash import content_hash


class PdfIntake:
    name = "pdf_upload"

    def fetch(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        return [{"path": str(p.resolve())}]

    def _read_pdf(self, fpath: str) -> Dict[str, Any]:
        reader = PdfReader(fpath)
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return {"pages": pages, "page_count": len(reader.pages)}

    def normalize(self, payload: Dict[str, Any]) -> NormalizedDoc:
        fpath = payload["path"]
        meta = self._read_pdf(fpath)
        text = "\n\n".join(meta["pages"]).strip()
        filename = Path(fpath).name

        core = make_core(
            source_id=source_id_pdf(filename),
            source=self.name,
            doc_type="pdf",
            title=filename,
            text=text,
            links=[f"file://{fpath}"],
            tags=["pdf"],
        )

        facet = facet_pdf(
            filename=filename,
            client_id=None,
            pages=meta["page_count"],
            mime="application/pdf",
        )

        core.content_hash = content_hash(
            core.text, json.dumps(facet, ensure_ascii=False, sort_keys=True)
        )

        return NormalizedDoc(core=core, facet=facet, raw={"path": fpath})
