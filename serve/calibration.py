"""Post-hoc calibration of decision probabilities (serve/decision.py).

A fitted entry is a MONOTONE map per field that keeps 0.5 fixed - a scale on the logit for a boolean (Platt scaling
without its intercept: an intercept moves the 0.5 crossing and with it the chosen value), a temperature for an enum -
so the chosen value (the argmax) and the ranking of contexts (AUC) never change; only how far the probability sits
from 0.5 does.
Entries live in `serve/calibration.json`, keyed by the request they were fitted on: the compiled schema (field names,
types, values in the caller's order, descriptions) together with the request's `instructions`, hashed.  The same
schema under other instructions gives another distribution, so it gets no entry and stays identity.  A missing or
unreadable file, an unknown key or a malformed entry is identity too (one line on stdout, once).

`tools/fit_calibration.py` fits the entries from a labelled set and the raw distributions a serve logs when
`STRATA_DECISION_LOG=<file>` is set (one JSON line per context: the key and every field's probabilities, no text).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
_table: dict | None = None
_lock = threading.Lock()
_said: set = set()


def _say(msg: str) -> None:
    if msg not in _said:
        _said.add(msg)
        print(f"[strata] calibration: {msg}", flush=True)


def key(specs, instructions: str) -> str:
    """sha256 of the compiled schema + instructions: compact fields and JSON Schema `properties` of one schema hash
    equal; another description, value set, order or instruction text is another key."""
    canon = {"instructions": instructions,
             "fields": [{"name": s.name, "type": s.type, "description": s.description, "values": list(s.values)}
                        for s in specs]}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _valid(entry) -> bool:
    if not isinstance(entry, dict) or len(entry) != 1:
        return False
    if "platt" in entry:
        v = entry["platt"]
        return (isinstance(v, list) and len(v) == 1 and isinstance(v[0], (int, float)) and math.isfinite(v[0])
                and v[0] > 0)                                       # [a], a > 0: the order of probabilities is kept
    if "temperature" in entry:
        t = entry["temperature"]
        return isinstance(t, (int, float)) and math.isfinite(t) and t > 0
    return False


def load(path: str = PATH) -> dict:
    """{key: {field: entry}} from `path`; identity ({}) when the file is missing or malformed (said once)."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        _say(f"{os.path.basename(path)} unreadable, calibration off: {e}")
        return {}
    if not isinstance(raw, dict):
        _say(f"{os.path.basename(path)} is not an object, calibration off")
        return {}
    out: dict = {}
    for k, fields in raw.items():
        if not isinstance(fields, dict):
            _say(f"entry {k[:12]} is not an object, skipped")
            continue
        good = {name: e for name, e in fields.items() if _valid(e)}
        if len(good) != len(fields):
            _say(f"entry {k[:12]} has malformed fields, they stay identity: "
                 + ", ".join(sorted(set(fields) - set(good))))
        if good:
            out[k] = good
    return out


def table() -> dict:
    global _table
    with _lock:
        if _table is None:
            _table = load()
        return _table


def reset(entries: dict | None = None) -> None:
    """Tests: replace the table (None = read the file again on the next use)."""
    global _table
    with _lock:
        _table = entries
    _said.clear()


def entries() -> int:
    return sum(len(v) for v in table().values())


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def platt(probs: list[float], a: float) -> list[float]:
    """A boolean's [P(true), P(false)] through sigmoid(a * logit(P(true))); a > 0 keeps the order, 0.5 stays 0.5, so
    the argmax is kept (Platt's intercept would move the crossing and is therefore not a parameter)."""
    q = _sigmoid(a * _logit(probs[0]))
    return [q, 1.0 - q]


def temperature(probs: list[float], t: float) -> list[float]:
    """An enum's distribution with its log-probabilities divided by t, renormalised; the argmax stays."""
    lp = [math.log(max(p, 1e-12)) / t for p in probs]
    mx = max(lp)
    z = sum(math.exp(x - mx) for x in lp)
    return [math.exp(x - mx) / z for x in lp]


def apply(k: str, plans, probs: list) -> list:
    """The per-field distributions of one context after calibration: identity where no entry applies.  A greedy
    field has no distribution (None) and is left alone."""
    fields = table().get(k)
    if not fields:
        return probs
    out = list(probs)
    for f, p in enumerate(plans):
        e, pr = fields.get(p.spec.name), probs[f]
        if not e or pr is None:
            continue
        if "platt" in e and p.spec.type == "boolean" and len(pr) == 2:
            out[f] = platt(pr, *e["platt"])
        elif "temperature" in e and p.spec.type == "enum":
            out[f] = temperature(pr, e["temperature"])
    return out


def context_id(context: str) -> str:
    """How the fit tool pairs a logged row with its labelled text: a sha256 prefix of the context, never the text."""
    return hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]


def log_distributions(k: str, plans, probs: list, context: str = "") -> None:
    """STRATA_DECISION_LOG=<file>: append the raw distributions of one context (the fit tool's input; no text)."""
    path = os.environ.get("STRATA_DECISION_LOG")
    if not path:
        return
    row = {"key": k, "context": context_id(context),
           "fields": {p.spec.name: pr for p, pr in zip(plans, probs) if pr is not None}}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as e:
        _say(f"STRATA_DECISION_LOG not writable: {e}")
