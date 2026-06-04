"""Fetch the official NOAA CPC ENSO outlook (synopsis + headline probabilities).

This lets the dashboard show the authoritative dynamical-model consensus next to
its own in-house statistical forecast. It is fetched live and parsed defensively:
if the page is unreachable or the wording changes, it returns ``available=False``
and the app simply links out to the official product instead of breaking.
"""

from __future__ import annotations

import html
import re

import requests

from . import cache

CPC_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml"


def fetch_cpc_outlook(max_age_hours: float = 12.0) -> dict:
    """Return the CPC synopsis + (probability, period) pairs, or available=False."""
    cached = cache.get("cpc", {"u": CPC_URL}, max_age_hours)
    if cached is not None:
        text = cached["text"]
    else:
        try:
            r = requests.get(CPC_URL, timeout=20)
            r.raise_for_status()
            text = r.text
            cache.put("cpc", {"u": CPC_URL}, {"text": text})
        except requests.RequestException:
            return {"available": False}

    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", html.unescape(plain)).strip()

    syn = re.search(r"Synopsis:\s*([^.]*\.)", plain)
    # Periods come in two forms: "May-July 2026" and "December 2026-February 2027".
    # Capture greedily up to the closing parenthesis, ending on a 4-digit year.
    raw = re.findall(r"(\d{1,3})\s*%\s*chance\s*in\s*([^)]+\d{4})", plain)
    status = re.search(r"(El Ni[^ ]*o|La Ni[^ ]*a|ENSO[- ]neutral)\s+"
                       r"(Watch|Advisory|Warning)", plain)

    probs, seen = [], set()
    for p, period in raw:
        period = re.sub(r"\s+", " ", period).strip()
        if period not in seen:
            seen.add(period)
            probs.append((int(p), period))

    if not syn and not probs:
        return {"available": False}
    return {
        "available": True,
        "synopsis": syn.group(1).strip() if syn else "",
        "status": (status.group(0) if status else ""),
        "probs": probs,
    }
