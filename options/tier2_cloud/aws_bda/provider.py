"""Tier 2 option: Amazon Bedrock Data Automation (3.4.0), recognising and splitting a whole file.

A Data Automation project in the customer's AWS account, with the splitter on and one blueprint per document type
(named as the process's types: BDA matches a part to a blueprint by its name, description and fields). The file is
sent with the synchronous InvokeDataAutomation call (the bytes inline: no S3 bucket). Each output segment says which
blueprint matched (MATCH / NO_MATCH / FALLBACK) and its pages.
"""
from __future__ import annotations

import json
from typing import Any

from docauto.errors import AuthFailed, InvalidInput, ProviderError, ProviderTimeout, RateLimited
from docauto.provider import HealthStatus, Segment, SplitProvider

AUTH_CODES = {"AccessDeniedException", "UnrecognizedClientException", "ExpiredTokenException", "InvalidSignatureException"}
RATE_CODES = {"ThrottlingException", "ServiceQuotaExceededException", "TooManyRequestsException"}


def _json(s: Any) -> dict:
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s) if s else {}
    except (TypeError, ValueError):
        return {}


def _pages(custom: dict, standard: dict) -> list[int]:
    """1-based pages of a segment: the custom output's split_document.page_indices, else the standard output's
    start and end page indices (both 0-based)."""
    idx = (custom.get("split_document") or {}).get("page_indices")
    if idx:
        return sorted({int(i) + 1 for i in idx})
    meta = standard.get("metadata") or {}
    if "start_page_index" in meta:
        start = int(meta["start_page_index"])
        return list(range(start + 1, int(meta.get("end_page_index", start)) + 2))
    return []


def segments_from_output(r: dict, page_count: int) -> list[Segment]:
    out: list[Segment] = []
    segs = r.get("outputSegments") or []
    for s in segs:
        custom, standard = _json(s.get("customOutput")), _json(s.get("standardOutput"))
        bp = custom.get("matched_blueprint") or {}
        matched = str(s.get("customOutputStatus", "")).upper() == "MATCH"
        pages = _pages(custom, standard) or (list(range(1, page_count + 1)) if len(segs) == 1 else [])
        out.append(Segment(label=bp.get("name") if matched else None, pages=pages,
                           confidence=bp.get("confidence") if matched else None))
    return out


def default_profile(region: str, account: str) -> str:
    """AWS's own cross-region profile for the region's geography (us, eu, apac)."""
    geo = "us" if region.startswith("us-") else "eu" if region.startswith("eu-") else "apac"
    return f"arn:aws:bedrock:{region}:{account}:data-automation-profile/{geo}.data-automation-v1"


class AwsBdaProvider(SplitProvider):
    def init(self, ctx):
        super().init(ctx)
        cfg = ctx.config
        self.region = (cfg.get("region") or "").strip()
        self.project = (cfg.get("projectArn") or "").strip()
        self.stage = (cfg.get("stage") or "LIVE").strip()
        self.profile = (cfg.get("profileArn") or "").strip()
        if self.region:
            self.info.regions = [self.region]

    def health_check(self) -> HealthStatus:
        try:
            import boto3  # noqa: F401
        except Exception:
            return HealthStatus(False, "boto3 not installed (pip install boto3)")
        if not self.region or not self.project:
            return HealthStatus(False, "region/projectArn not configured")
        return HealthStatus(True, f"project configured ({self.region})")

    def _session_kwargs(self) -> dict:
        creds = self.ctx.credentials
        kwargs: dict[str, Any] = {"region_name": self.region}
        if creds.get("accessKeyId"):
            kwargs.update(aws_access_key_id=creds["accessKeyId"], aws_secret_access_key=creds.get("secretAccessKey"),
                          aws_session_token=creds.get("sessionToken") or None)
        return kwargs

    def _client(self):
        import boto3
        return boto3.client("bedrock-data-automation-runtime", **self._session_kwargs())

    def _profile(self) -> str:
        if not self.profile:
            import boto3
            account = boto3.client("sts", **self._session_kwargs()).get_caller_identity()["Account"]
            self.profile = default_profile(self.region, account)
        return self.profile

    def split(self, data, mime, page_count, labels, descriptions=None, segment=True) -> list[Segment]:
        if not data:
            raise InvalidInput("aws-bda needs the file", self.info.id)
        try:
            r = self._client().invoke_data_automation(
                inputConfiguration={"bytes": data},
                dataAutomationConfiguration={"dataAutomationProjectArn": self.project, "stage": self.stage},
                dataAutomationProfileArn=self._profile())
        except Exception as e:   # botocore errors: mapped by their code
            code = (getattr(e, "response", None) or {}).get("Error", {}).get("Code", "")
            name = type(e).__name__
            if code in AUTH_CODES or name in AUTH_CODES:
                raise AuthFailed(str(e)[:300], self.info.id)
            if code in RATE_CODES or name in RATE_CODES:
                raise RateLimited(str(e)[:300], self.info.id)
            if "Timeout" in name:
                raise ProviderTimeout(str(e)[:300], self.info.id)
            if code == "ValidationException":
                raise InvalidInput(str(e)[:300], self.info.id)
            raise ProviderError(str(e)[:300], self.info.id)
        return segments_from_output(r, page_count)
