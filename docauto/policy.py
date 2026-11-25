"""Deployment profile enforcement: decides which engines may receive a document."""
from __future__ import annotations

from fnmatch import fnmatch

from .provider import ProviderInfo

# custom: the administrator's own profile: exactly the engines in the allow list (local or cloud), in the allowed
# regions, with cloud opt-in as for hybrid
PROFILES = ("onprem_airgapped", "onprem_hybrid", "cloud_saas", "custom")

# Region groups for residency rules: an allowed region may name a group ("eu"), and an engine's region may be a
# cloud's own name for a place ("westeurope", "europe-west3", "eu-central-1"). Patterns use * like the allow list.
REGION_GROUPS: dict[str, list[str]] = {
    "eu": ["eu", "eu-*", "europe-*", "*europe", "westeurope", "northeurope", "francecentral", "francesouth", "germany*",
           "italynorth", "spaincentral", "swedencentral", "polandcentral", "belgiumcentral", "austriaeast", "ireland",
           "frankfurt", "paris", "milan", "stockholm", "eu-west-*", "eu-central-*", "eu-north-*", "eu-south-*"],
    "uk": ["uk", "uk*", "eu-west-2", "europe-west2", "london"],
    "switzerland": ["switzerland*", "eu-central-2", "europe-west6", "zurich"],
    "gcc": ["gcc", "me-*", "uae*", "qatar*", "saudi*", "me-central-*", "me-south-*", "me-west1", "me-central2", "riyadh",
            "dammam", "jeddah", "dubai", "abudhabi", "doha", "bahrain"],
    "ksa": ["ksa", "saudi*", "me-central2", "riyadh", "dammam", "jeddah"],
    "uae": ["uae*", "me-central-1", "me-central1", "dubai", "abudhabi"],
    "india": ["india", "*india", "centralindia", "southindia", "westindia", "jioindia*", "ap-south-*", "asia-south*",
              "mumbai", "hyderabad", "pune", "chennai"],
    "us": ["us", "us-*", "*us", "*us2", "*us3", "eastus*", "westus*", "centralus", "northcentralus", "southcentralus",
           "westcentralus", "virginia", "ohio", "oregon", "iowa"],
    "canada": ["canada*", "ca-central-*", "ca-west-*", "northamerica-northeast*", "toronto", "montreal"],
    "australia": ["australia*", "ap-southeast-2", "ap-southeast-4", "sydney", "melbourne"],
    "japan": ["japan*", "ap-northeast-1", "ap-northeast-3", "asia-northeast1", "asia-northeast2", "tokyo", "osaka"],
    "singapore": ["singapore", "southeastasia", "ap-southeast-1", "asia-southeast1"],
    "brazil": ["brazil*", "sa-east-1", "southamerica-east1", "saopaulo"],
}


def region_matches(engine_region: str, allowed: str) -> bool:
    """An engine's region is allowed by an allowed entry: the same name, a pattern, or a group that holds it."""
    r, a = engine_region.strip().lower(), allowed.strip().lower()
    if not r or not a:
        return False
    if fnmatch(r, a):
        return True
    return any(fnmatch(r, p) for p in REGION_GROUPS.get(a, []))


def _as_list(v) -> list[str]:
    """Accept YAML lists or comma-separated strings (from environment variables)."""
    if not v:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return [str(x).strip() for x in v if str(x).strip()]


class PolicyEngine:
    def __init__(self, config: dict):
        self.profile = config.get("deployment_profile", "onprem_airgapped")
        if self.profile not in PROFILES:
            raise ValueError(f"unknown deployment_profile '{self.profile}', expected one of {PROFILES}")
        prov = config.get("providers", {})
        self.allow = _as_list(prov.get("allow")) or ["*"]
        self.deny = _as_list(prov.get("deny"))
        res = config.get("residency", {})
        self.allowed_regions = _as_list(res.get("allowed_regions"))
        self.allow_cross_border = bool(res.get("allow_cross_border", False))
        # client consent for outbound engines; the shipped config starts local-only (OCR strategy plan)
        opt_in = (config.get("routing") or {}).get("cloud_opt_in", True)
        self.cloud_opt_in = opt_in if isinstance(opt_in, bool) else str(opt_in).strip().lower() in ("true", "1", "yes")

    def check(self, info: ProviderInfo) -> tuple[bool, str]:
        if any(fnmatch(info.id, d) for d in self.deny):
            return False, f"'{info.id}' is on the deny list"
        if not any(fnmatch(info.id, a) for a in self.allow):
            return False, f"'{info.id}' is not on the allow list"
        if info.runsLocally:
            return True, "local engine"
        if self.profile == "onprem_airgapped":
            return False, "profile onprem_airgapped forbids outbound engines"
        if not self.cloud_opt_in:
            return False, "cloud opt-in is off: documents stay on this machine"
        if self.allowed_regions and not self.allow_cross_border:
            if not info.regions or not any(region_matches(r, a) for r in info.regions for a in self.allowed_regions):
                return False, (f"'{info.id}' regions {info.regions or ['unknown']} not in allowed "
                               f"regions {self.allowed_regions}")
        return True, "allowed"

    def allows(self, info: ProviderInfo) -> bool:
        return self.check(info)[0]
