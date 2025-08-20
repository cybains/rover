from urllib.parse import urlsplit




def source_id_url(url: str) -> str:
return f"url:{url}"




def source_id_pdf(path: str) -> str:
return f"pdf:{path}"




def source_id_kv(namespace: str, key: str) -> str:
# Generic helper e.g., ("ecb_fx", "2025-08-20") -> "ecb_fx:2025-08-20"
return f"{namespace}:{key}"




def canonical_host(url: str) -> str:
return urlsplit(url).netloc.lower()