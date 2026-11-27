"""Tier 2 option: Google Document AI splitter or classifier (3.4.0), recognising and splitting a whole file.

A Custom Splitter (the pretrained splitter works with no training) or Custom Classifier processor made in the
customer's Google Cloud project. Its kinds of document are set up in Google with the same names as the process's
document types. A splitter answers with one entity per part (its type, confidence and pages); a classifier with one
entity per kind and its confidence, for the whole file.
"""
from __future__ import annotations

from typing import Any

from docauto.errors import AuthFailed, InvalidInput, ProviderError, RateLimited
from docauto.provider import HealthStatus, Segment, SplitProvider


def segments_from_document(document: dict[str, Any], page_count: int) -> list[Segment]:
    """Entities with pages (a splitter) are parts; entities without (a classifier) rate the whole file: the most
    confident one is taken. Page references are 0-based (an absent one is page 0)."""
    parts, whole = [], []
    for e in document.get("entities") or []:
        refs = (e.get("pageAnchor") or {}).get("pageRefs") or []
        label = e.get("type") or None
        conf = e.get("confidence")
        if refs:
            parts.append(Segment(label=label, pages=sorted({int(r.get("page", 0)) + 1 for r in refs}), confidence=conf))
        else:
            whole.append(Segment(label=label, pages=list(range(1, page_count + 1)), confidence=conf))
    if parts:
        return parts
    return [max(whole, key=lambda s: s.confidence or 0.0)] if whole else []


class GoogleSplitProvider(SplitProvider):
    def init(self, ctx):
        super().init(ctx)
        cfg = ctx.config
        self.project = (cfg.get("projectId") or "").strip()
        self.location = (cfg.get("location") or "us").strip()
        self.processor = (cfg.get("processorId") or "").strip()
        self.version = (cfg.get("processorVersion") or "").strip()
        self.info.regions = [self.location] if self.location else []

    def health_check(self) -> HealthStatus:
        try:
            from google.cloud import documentai  # noqa: F401
        except Exception:
            return HealthStatus(False, "google-cloud-documentai not installed")
        if not (self.project and self.processor):
            return HealthStatus(False, "projectId/processorId not configured")
        return HealthStatus(True, "configured")

    def split(self, data, mime, page_count, labels, descriptions=None, segment=True) -> list[Segment]:
        if not data:
            raise InvalidInput("google-docai-split needs the file", self.info.id)
        return segments_from_document(self._process(data, mime or "application/pdf"), page_count)

    def _process(self, data: bytes, mime: str) -> dict:
        """The processor's Document, as a camelCase dict."""
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai
        from google.protobuf.json_format import MessageToDict

        creds_file = self.ctx.credentials.get("serviceAccountFile")
        opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
        client = (documentai.DocumentProcessorServiceClient.from_service_account_file(creds_file, client_options=opts)
                  if creds_file else documentai.DocumentProcessorServiceClient(client_options=opts))
        name = (client.processor_version_path(self.project, self.location, self.processor, self.version) if self.version
                else client.processor_path(self.project, self.location, self.processor))
        try:
            result = client.process_document(request=documentai.ProcessRequest(
                name=name, raw_document=documentai.RawDocument(content=data, mime_type=mime)))
        except Exception as e:
            s = str(e)
            if "PERMISSION_DENIED" in s or "UNAUTHENTICATED" in s:
                raise AuthFailed(s[:300], self.info.id)
            if "RESOURCE_EXHAUSTED" in s:
                raise RateLimited(s[:300], self.info.id)
            if "INVALID_ARGUMENT" in s:
                raise InvalidInput(s[:300], self.info.id)
            raise ProviderError(s[:300], self.info.id)
        return MessageToDict(documentai.Document.pb(result.document))
