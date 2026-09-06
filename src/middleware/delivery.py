"""Deterministic pre-delivery audit (whitepaper §4.2–4.4, §8.1).

These are structural checks, not external verification. A citation's existence
does not establish its truth. Preserve the raw result and store the audit beside
it; the CLI renders labels from this audit before exposing model content.
"""
from __future__ import annotations

import re
from src.middleware import confirmation as checks

_ZONE = re.compile(r"Zone\s*([ABC])\b|【\s*([ABC])\s*[：:]", re.I)
_LABELS = {"A": "事實陳述・來源未獨立查證", "B": "推測／待驗證", "C": "立場／建議"}


def audit_delivery(text: str, *, mode="default", degraded=False) -> dict:
    segments = []
    # Keep each non-empty line addressable. No single citation upgrades a whole answer.
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _ZONE.search(line)
        declared = next((g.upper() for g in match.groups() if g), "B") if match else checks.classify_zone(line)
        zone = declared if declared in {"A", "B", "C"} else "B"
        source = checks.check_source_traceability(line, zone)
        downgraded = bool(source["should_downgrade"] or (degraded and zone != "C"))
        if downgraded:
            zone = "B"
        falsification = checks.check_falsification(line)
        segments.append({
            "text": line, "zone": zone, "declared_zone": declared,
            "downgraded": downgraded, "source_check": source,
            "falsification_condition": line if zone == "B" and falsification.get("has_falsification")
                and not falsification.get("is_generic") else None,
        })
    falsification = checks.check_falsification(text)
    warnings = []
    if any(s["downgraded"] for s in segments):
        warnings.append("無可追溯來源的事實陳述或品質未通過的內容，已降為待驗證。")
    if any(s["zone"] == "B" for s in segments) and (
        not falsification.get("has_falsification") or falsification.get("is_generic")
    ):
        warnings.append("未提供可操作的反駁條件；請勿將推測當作已驗證結論。")
    zones = {s["zone"] for s in segments}
    return {
        "version": 1, "mode": mode, "raw_text": text, "segments": segments,
        "zone": next(iter(zones)) if len(zones) == 1 else "B",
        "mixed_zones": len(zones) > 1,
        "external_verification": "not_performed",
        "warnings": warnings,
        "atl_action_check": checks.check_action_specificity(text),
        "atl_falsification_check": falsification,
        "degraded": degraded or bool(warnings),
    }


def render_delivery(audit: dict) -> str:
    lines = ["[資訊分層] 規則式檢查；來源內容尚未獨立查證。"]
    lines.extend("[待驗證] " + warning for warning in audit["warnings"])
    in_code = False
    segments = iter(audit["segments"])
    for raw_line in audit["raw_text"].splitlines():
        if not raw_line.strip():
            lines.append(raw_line)
            continue
        segment = next(segments)
        zone = segment["zone"]
        if segment["text"].lstrip().startswith("```"):
            in_code = not in_code
            lines.append(segment["text"])
            continue
        if in_code:
            lines.append(segment["text"])
            continue
        # Raw model labels may be wrong. The outer label is the runtime's decision.
        lines.append(f"[Zone {zone}｜{_LABELS[zone]}] {segment['text']}")
    return "\n".join(lines)
