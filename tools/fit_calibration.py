"""Fit one decision calibration entry (serve/calibration.json) from a labelled set and a serve's distribution log.

usage: python tools/fit_calibration.py --schema schema.json --field NAME --data labelled.jsonl --log decisions.jsonl
           [--instructions TEXT] [--text-key text] [--folds 5] [--min-gain 0.02] [--out serve/calibration.json] [--dry]

schema.json holds the request's `schema` (compact fields or JSON Schema); with --instructions it names the request the
entry is for (serve/calibration.py `key`).  decisions.jsonl is what a serve writes with STRATA_DECISION_LOG set: one
row per context with the key, the context's sha256 prefix and every field's raw distribution.  labelled.jsonl has the
context text (--text-key) and the field's true value; rows are paired by the context hash, so the log never carries
text.  A boolean field gets Platt scaling without its intercept (a scale on the logit, with Platt's label
smoothing; an intercept would move the 0.5 crossing and change chosen values), an enum field a temperature.

The fit is judged on held-out folds (split by context, so repeats of one context stay together): the entry is written
only if the mean held-out Brier score drops by --min-gain or more and no held-out chosen value changes (a monotone
map cannot change one; it is checked anyway).  Prints the numbers the record needs.  Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from serve import calibration, decision  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def logit(p):
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def fit_platt(samples):
    """samples: [(P(true), truth bool)] -> a > 0 minimising the smoothed log loss of sigmoid(a * logit(p)).
    Platt scaling without its intercept: the intercept would move the 0.5 crossing, i.e. change chosen values,
    which a calibration must not do.  Newton's method on a convex objective."""
    n_pos = sum(1 for _, y in samples if y)
    n_neg = len(samples) - n_pos
    t_pos, t_neg = (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0)      # Platt 1999: targets off 0 and 1
    rows = [(logit(p), t_pos if y else t_neg) for p, y in samples]
    a = 1.0
    for _ in range(100):
        g = h = 0.0
        for d, t in rows:
            q = sigmoid(a * d)
            g += (q - t) * d
            h += q * (1.0 - q) * d * d
        if h <= 1e-12:
            break
        da = g / h
        step = 1.0
        while a - step * da <= 1e-3:                                       # stay on the order-keeping side
            step *= 0.5
        a -= step * da
        if abs(da) < 1e-9:
            break
    return a


def fit_temperature(samples):
    """samples: [(distribution, truth index)] -> t > 0 minimising the log loss of softmax(log p / t)."""
    def nll(t):
        s = 0.0
        for pr, k in samples:
            lp = [math.log(max(x, 1e-12)) / t for x in pr]
            mx = max(lp)
            s -= lp[k] - (mx + math.log(sum(math.exp(x - mx) for x in lp)))
        return s
    lo, hi = -3.0, 3.0                                                     # log t: 0.05 .. 20
    r = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - r * (hi - lo), lo + r * (hi - lo)
    f1, f2 = nll(math.exp(x1)), nll(math.exp(x2))
    for _ in range(80):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - r * (hi - lo)
            f1 = nll(math.exp(x1))
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + r * (hi - lo)
            f2 = nll(math.exp(x2))
    return math.exp((lo + hi) / 2)


def brier(kind, pr, truth):
    if kind == "platt":
        return (pr[0] - (1.0 if truth else 0.0)) ** 2
    return sum((p - (1.0 if k == truth else 0.0)) ** 2 for k, p in enumerate(pr))


def apply(kind, params, pr):
    return calibration.platt(pr, params) if kind == "platt" else calibration.temperature(pr, params)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--schema", required=True, help="JSON file: the request's schema")
    ap.add_argument("--field", required=True)
    ap.add_argument("--data", required=True, help="labelled jsonl: the context text and the field's true value")
    ap.add_argument("--log", required=True, help="the serve's STRATA_DECISION_LOG file")
    ap.add_argument("--instructions", default="")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--out", default=str(ROOT / "serve" / "calibration.json"))
    ap.add_argument("--dry", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    with open(a.schema, encoding="utf-8") as f:
        schema = json.load(f)
    specs = decision.compile_schema(schema.get("schema", schema))
    spec = next((s for s in specs if s.name == a.field), None)
    if spec is None:
        sys.exit(f"field {a.field!r} is not in the schema ({', '.join(s.name for s in specs)})")
    kind = "platt" if spec.type == "boolean" else "temperature" if spec.type == "enum" else None
    if kind is None:
        sys.exit(f"field {a.field!r} is {spec.type}: only boolean and enum fields are calibrated")
    key = calibration.key(specs, a.instructions)
    truth = {}
    for row in read_jsonl(a.data):
        v = row.get(a.field)
        if kind == "platt":
            if not isinstance(v, bool):
                sys.exit(f"labelled row without a boolean {a.field!r}: {row}")
            t = v
        else:
            if v not in spec.values:
                sys.exit(f"labelled row with {a.field!r} = {v!r}, not one of the schema's values")
            t = spec.values.index(v)
        truth[calibration.context_id(row[a.text_key])] = t
    samples, by_ctx, other_keys, unmatched = [], {}, 0, 0
    for row in read_jsonl(a.log):
        if row.get("key") != key:
            other_keys += 1
            continue
        pr = (row.get("fields") or {}).get(a.field)
        cid = row.get("context")
        if pr is None or cid not in truth:
            unmatched += 1
            continue
        if kind == "platt" and len(pr) != 2 or kind == "temperature" and len(pr) != len(spec.values):
            sys.exit(f"a logged distribution has {len(pr)} entries, the field has {len(spec.values)} values")
        samples.append((pr, truth[cid]))
        by_ctx.setdefault(cid, []).append(len(samples) - 1)
    if other_keys:
        print(f"skipped {other_keys} log rows of other requests (key {key[:12]}... wanted)")
    if unmatched:
        print(f"skipped {unmatched} log rows without a labelled context")
    if len(by_ctx) < 2 * a.folds:
        sys.exit(f"{len(by_ctx)} labelled contexts in the log: too few for {a.folds} folds")
    print(f"{a.field} ({kind}): {len(samples)} samples over {len(by_ctx)} contexts, key {key[:12]}...")
    fit = (lambda s: fit_platt([(pr[0], y) for pr, y in s])) if kind == "platt" else fit_temperature
    ctxs = sorted(by_ctx)
    random.Random(0).shuffle(ctxs)
    folds = [ctxs[i::a.folds] for i in range(a.folds)]
    before, after, flips, held = 0.0, 0.0, 0, 0
    for fold in folds:
        test = {i for c in fold for i in by_ctx[c]}
        params = fit([samples[i] for i in range(len(samples)) if i not in test])
        for i in test:
            pr, t = samples[i]
            cal = apply(kind, params, pr)
            before += brier(kind, pr, t)
            after += brier(kind, cal, t)
            flips += pr.index(max(pr)) != cal.index(max(cal))
            held += 1
    before, after = before / held, after / held
    params = fit(samples)
    shown = f"a={params:.4f}" if kind == "platt" else f"t={params:.4f}"
    print(f"held-out Brier {before:.4f} -> {after:.4f} (gain {before - after:+.4f}, {a.folds} folds, {held} rows); "
          f"argmax changes {flips}; fit on all: {shown}")
    ok = before - after >= a.min_gain and flips == 0
    print("ACCEPT" if ok else f"REJECT (needs a gain of {a.min_gain} and no argmax change)")
    if not ok or a.dry:
        return 0 if ok else 1
    entry = {"platt": [round(params, 6)]} if kind == "platt" else {"temperature": round(params, 6)}
    table = {}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            table = json.load(f)
    table.setdefault(key, {})[a.field] = entry
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"written {a.out}: {key[:12]}... {a.field} = {entry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
