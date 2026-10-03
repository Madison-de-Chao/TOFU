"""Bounded, auditable API memory selection; complete records remain local."""
from src.middleware.endpoint import encode_retrieved_endpoints, DEFAULT_TOP_K

MAX_MEMORY_CHARS = 9000  # Character budget, deliberately not called a token limit.


def pack_lookup(rows, *, categories=(), max_chars=MAX_MEMORY_CHARS, top_k=DEFAULT_TOP_K):
    if max_chars < 0 or top_k < 0:
        raise ValueError("memory budgets must be non-negative")
    selected, omitted, blocks = [], [], []
    used = 0
    for row in rows:
        block = encode_retrieved_endpoints([row])
        block = block.replace("[E001]", f"[E{len(selected) + 1:03d}]", 1)
        cost = len(block) + (2 if blocks else 0)
        if len(selected) >= top_k or used + cost > max_chars:
            omitted.append(row.get("endpoint_id"))
            continue
        selected.append(row)
        blocks.append(block)
        used += cost
    # Being asked a question is not evidence that its category was answered.
    known = sorted({c for row in selected
                    for c in (row.get("start_data") or {}).get("answered_categories", [])
                    if c in categories})
    audit = {
        "performed": True, "selected_endpoint_ids": [r.get("endpoint_id") for r in selected],
        "pending_confirmation_ids": [r.get("endpoint_id") for r in selected
                                     if r.get("status") == "pending_confirmation"],
        "omitted_endpoint_ids": omitted,
        "known_categories": known, "unknown_categories": sorted(set(categories) - set(known)),
        "encoded_chars": used, "max_chars": max_chars, "top_k": top_k,
        "unknown_scope": "尚無已確認歷史的分類；不代表本次輸入一定缺少該資訊",
    }
    return selected, "\n\n".join(blocks), audit
