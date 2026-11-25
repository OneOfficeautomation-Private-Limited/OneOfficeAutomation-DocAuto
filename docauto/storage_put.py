"""Cloud storage deliveries without extra packages: a file PUT to Amazon S3 (or an S3-compatible store: MinIO, Wasabi,
Cloudflare R2...) signed with AWS Signature Version 4, to an Azure Blob container through its SAS address, or to any
web address that takes uploads (a URL with {file} where the file name goes).

Secrets (the S3 secret key, the SAS token) are never kept in the process: the delivery names a vault reference and the
Command Center gives the value at delivery time (docauto.docflow, the "secret" callback), kept in memory only.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
from typing import Optional
from urllib.parse import quote, urlsplit

import httpx


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def s3_request(method: str, url: str, region: str, access_key: str, secret_key: str, body: bytes,
               headers: Optional[dict] = None, now: Optional[_dt.datetime] = None, service: str = "s3") -> dict:
    """The headers of a SigV4-signed request (Authorization, x-amz-date, x-amz-content-sha256 and the given ones)."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    amz_date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    u = urlsplit(url)
    payload = hashlib.sha256(body).hexdigest()
    hdrs = {k.lower(): str(v).strip() for k, v in (headers or {}).items()}
    hdrs.update({"host": u.netloc, "x-amz-date": amz_date, "x-amz-content-sha256": payload})
    names = sorted(hdrs)
    canonical = "\n".join([method, quote(u.path or "/", safe="/-_.~%"), u.query,
                           "".join(f"{n}:{hdrs[n]}\n" for n in names), ";".join(names), payload])
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k = _sign(_sign(_sign(_sign(("AWS4" + secret_key).encode(), day), region), service), "aws4_request")
    sig = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    out = {n: hdrs[n] for n in names if n != "host"}
    out["Authorization"] = (f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={';'.join(names)}, "
                            f"Signature={sig}")
    return out


def object_url(p: dict, name: str) -> str:
    """Where a file goes, from the delivery's params: provider s3 (bucket, region, endpoint), azure_blob (container
    address) or http_put (address with {file})."""
    key = (str(p.get("prefix") or "").strip("/") + "/" + name).lstrip("/")
    provider = p.get("provider")
    if provider == "s3":
        endpoint = str(p.get("endpoint") or f"https://s3.{p.get('region') or 'us-east-1'}.amazonaws.com").rstrip("/")
        return f"{endpoint}/{p['bucket']}/{quote(key)}"
    if provider == "azure_blob":
        return str(p["container"]).rstrip("/") + "/" + quote(key)
    return str(p["url"]).replace("{file}", quote(key))


def put_file(p: dict, name: str, data: bytes, content_type: str, secret: str = "", timeout: float = 60) -> str:
    """Uploads one file; returns where it went. Raises httpx errors or RuntimeError on a refusal."""
    url = object_url(p, name)
    headers = {"Content-Type": content_type}
    extra = p.get("headers") or {}
    if isinstance(extra, dict):
        headers.update({str(k): str(v) for k, v in extra.items()})
    provider = p.get("provider")
    if provider == "s3":
        headers = s3_request("PUT", url, str(p.get("region") or "us-east-1"), str(p.get("accessKey") or ""), secret, data,
                             headers)
        send_url = url
    elif provider == "azure_blob":
        headers["x-ms-blob-type"] = "BlockBlob"
        send_url = url + ("?" + secret.lstrip("?") if secret else "")
    else:
        if secret:
            headers.setdefault("Authorization", secret)
        send_url = url
    r = httpx.put(send_url, content=data, headers=headers, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return url
