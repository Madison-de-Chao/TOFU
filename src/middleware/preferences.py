"""Conservative preference deduplication; never merge opposite polarities."""
from copy import deepcopy
import unicodedata


def normalize_preference(item: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", item).casefold().split())


def merge_preferences(existing: list, incoming: list) -> list:
    result, index = [], {}
    for original in [*existing, *incoming]:
        if not isinstance(original, dict) or not isinstance(original.get("item"), str):
            continue
        key = (normalize_preference(original["item"]), original.get("type", "implicit"))
        if not key[0]:
            continue
        if key not in index:
            index[key] = len(result)
            result.append(deepcopy(original))
            continue
        prior = result[index[key]]
        # Preserve every distinct observation and original spelling for audit.
        evidence = prior.setdefault("evidence", [])
        for record in (prior, original):
            observation = {k: record.get(k, "") for k in ("item", "type", "context", "session_ref")}
            if observation not in evidence:
                evidence.append(observation)
        for observation in original.get("evidence", []):
            if observation not in evidence:
                evidence.append(deepcopy(observation))
    return result
