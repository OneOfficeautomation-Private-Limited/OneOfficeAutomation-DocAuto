"""Tier 3 option: Amazon Bedrock (3.4.0): Claude, Amazon Nova and other models in the customer's own AWS account.

Bedrock's Converse API (one call shape for every model on Bedrock), through boto3. The prompt, JSON shape and
hallucination guard are the OpenAI-compatible connector's; page images are sent as Converse image blocks.
Credentials: an access key and secret (as Amazon Textract's), or the server's own AWS role when they are empty.
"""
from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from typing import Any

from docauto.errors import AuthFailed, ProviderError, ProviderTimeout, RateLimited
from docauto.provider import HealthStatus


def _load_shared():
    path = Path(__file__).resolve().parent.parent / "llm_openai_compatible" / "provider.py"
    spec = importlib.util.spec_from_file_location("docauto_option_llm_shared_bedrock", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_llm = _load_shared()
AUTH_CODES = {"AccessDeniedException", "UnrecognizedClientException", "ExpiredTokenException", "InvalidSignatureException"}
RATE_CODES = {"ThrottlingException", "ServiceQuotaExceededException", "TooManyRequestsException"}


def converse_content(user_content: Any) -> list[dict]:
    """The OpenAI-style message content (text, or text and data-URL images) as Converse content blocks."""
    if isinstance(user_content, str):
        return [{"text": user_content}]
    out = []
    for p in user_content:
        if p.get("type") == "text":
            out.append({"text": p["text"]})
        elif p.get("type") == "image_url":
            head, _, data = p["image_url"]["url"].partition(",")
            fmt = head[len("data:"):].split(";")[0].split("/")[-1] or "jpeg"
            out.append({"image": {"format": "jpeg" if fmt == "jpg" else fmt, "source": {"bytes": base64.b64decode(data)}}})
    return out


class BedrockProvider(_llm.LlmExtractionProvider):
    def init(self, ctx):
        super().init(ctx)
        cfg = ctx.config
        self.region = (cfg.get("region") or "").strip()
        self.model = (cfg.get("model") or "").strip()
        self.send_images = str(cfg.get("sendImages", True)).lower() not in ("false", "0", "no")
        if self.region:
            self.info.regions = [self.region]

    def health_check(self) -> HealthStatus:
        try:
            import boto3  # noqa: F401
        except Exception:
            return HealthStatus(False, "boto3 not installed (pip install boto3)")
        if not self.region or not self.model:
            return HealthStatus(False, "region/model not configured")
        return HealthStatus(True, f"model={self.model} ({self.region})")

    def _client(self):
        import boto3

        creds = self.ctx.credentials
        kwargs = {"region_name": self.region}
        if creds.get("accessKeyId"):
            kwargs.update(aws_access_key_id=creds["accessKeyId"], aws_secret_access_key=creds.get("secretAccessKey"),
                          aws_session_token=creds.get("sessionToken") or None)
        return boto3.client("bedrock-runtime", **kwargs)

    def _chat(self, user_content: Any) -> str:
        try:
            r = self._client().converse(
                modelId=self.model, system=[{"text": _llm.SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": converse_content(user_content)}],
                inferenceConfig={"temperature": self.temperature})
        except Exception as e:   # botocore errors: mapped by their code, nothing else is assumed
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
            name = type(e).__name__
            if code in AUTH_CODES or name in AUTH_CODES:
                raise AuthFailed(str(e)[:300], self.info.id)
            if code in RATE_CODES or name in RATE_CODES:
                raise RateLimited(str(e)[:300], self.info.id)
            if "Timeout" in name or code == "ModelTimeoutException":
                raise ProviderTimeout(str(e)[:300], self.info.id)
            raise ProviderError(str(e)[:300], self.info.id)
        blocks = ((r.get("output") or {}).get("message") or {}).get("content") or []
        return "".join(b.get("text", "") for b in blocks)
