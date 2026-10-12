"""Decision spike: one 8-field decision (POST /strata/score) against the same 8 fields as one-token chat calls.

usage: python tools/decide_spike.py --url http://127.0.0.1:18990 --key <api key> [--rounds 5] [--out spike.json]
Prints one JSON summary: median decision ms (with and without the shared head pinned), median ms of the 8 chat
calls, the speedup, whether both pick the same option per field, and whether a temperature-0 chat reads the same
before and after a decision (a decision must leave nothing behind in the session).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request

FIELDS = [
    ("category", "What type of request is this?", ["billing", "technical", "cancellation", "other"], True),
    ("urgent", "Does it need urgent handling?", ["true", "false"], False),
    ("priority", "How should it be prioritised?", ["low", "medium", "high", "critical"], True),
    ("mood", "What is the customer's mood?", ["calm", "frustrated", "angry"], True),
    ("refund", "Does it ask for a refund or a reversal?", ["true", "false"], False),
    ("language", "What language is it written in?", ["english", "portuguese", "spanish", "other"], True),
    ("mobile_app", "Is it about the mobile app?", ["true", "false"], False),
    ("second_issue", "Does it mention a second, unrelated problem?", ["true", "false"], False),
]
REQUEST = ("Hello, I was charged twice for my subscription this month and I need it fixed today because my rent "
           "is due on Friday. I already contacted you last week and nobody answered. Also, your mobile app keeps "
           "logging me out every time I open it, which makes it impossible to check my invoices. ")
FILLER = ("Earlier in this thread: the customer described their account history, the plan they are on, the "
          "devices they use, the dates of each payment, and the support tickets they opened before. ")
LETTERS = "ABCDEFGH"


def post(url, key, path, body, timeout=600):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {
        "Content-Type": "application/json", "Authorization": "Bearer " + key, "x-api-key": key})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000.0


def context():
    text = REQUEST
    while len(text) < 4000:                 # about 1k tokens
        text = FILLER + text
    return text


def decision_body(pin):
    catalogue = "\n".join(f'"{n}": {q}\nAllowed values: {", ".join(o)}' for n, q, o, _ in FIELDS)
    return {"system": "Select the requested field value for the context. Answer with a JSON object.\n" + catalogue,
            "user": context(), "lead": "{\n", "pin_head": pin,
            "fields": [{"suffix": f'  "{n}": ' + ('"' if quoted else ""), "options": o}
                       for n, _, o, quoted in FIELDS]}


def chat_field(url, key, i):
    n, q, opts, _ = FIELDS[i]
    body = {"model": "m", "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 10,
            "reasoning_effort": "none",
            "messages": [{"role": "system", "content": "Answer with one letter only."},
                         {"role": "user", "content": context() + "\n\n" + q + "\n" +
                          "\n".join(f"{LETTERS[j]}) {o}" for j, o in enumerate(opts))}]}
    out, ms = post(url, key, "/v1/chat/completions", body)
    top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    p = {}
    for t in top:
        k = t["token"].strip()
        if len(k) == 1 and k in LETTERS[:len(opts)]:
            p[k] = p.get(k, 0.0) + math.exp(t["logprob"])
    best = max(p, key=p.get) if p else None
    return (opts[LETTERS.index(best)] if best else None), ms


def probe_chat(url, key):
    out, _ = post(url, key, "/v1/chat/completions", {
        "model": "m", "max_tokens": 40, "temperature": 0, "reasoning_effort": "none",
        "messages": [{"role": "user", "content": "Name three uses of a paperclip, one line each."}]})
    return out["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    post(a.url, a.key, "/strata/score", decision_body(False))                     # warm-up: model loaded, caches
    chat_field(a.url, a.key, 0)
    before = probe_chat(a.url, a.key)
    pinned, plain, last = [], [], None
    for _ in range(a.rounds):
        last, ms = post(a.url, a.key, "/strata/score", decision_body(True))
        pinned.append(ms)
    for _ in range(a.rounds):
        r, ms = post(a.url, a.key, "/strata/score", decision_body(False))
        plain.append(ms)
    after = probe_chat(a.url, a.key)
    chats, picks = [], []
    for _ in range(max(1, a.rounds // 2)):
        total, picks = 0.0, []
        for i in range(len(FIELDS)):
            pick, ms = chat_field(a.url, a.key, i)
            total += ms
            picks.append(pick)
        chats.append(total)
    decided = [max(f, key=f.get) if f else None for f in last["fields"]]
    best = min(statistics.median(pinned), statistics.median(plain))
    summary = {
        "fields": len(FIELDS), "context_tokens": last["prompt_tokens"], "pin": last["pin"],
        "decision_ms_pinned_median": round(statistics.median(pinned), 1),
        "decision_ms_plain_median": round(statistics.median(plain), 1),
        "engine_ms_last": last["ms"],
        "chat_8_calls_ms_median": round(statistics.median(chats), 1),
        "speedup": round(statistics.median(chats) / best, 2) if best else None,
        "floor_1000ms": best <= 1000.0, "floor_3x": statistics.median(chats) >= 3 * best,
        "decision": dict(zip([f[0] for f in FIELDS], decided)), "decision_probs": last["fields"],
        "chat_picks": dict(zip([f[0] for f in FIELDS], picks)),
        "agree": sum(d == c for d, c in zip(decided, picks)),
        "chat_identical_after_decisions": before == after,
        "raw": {"pinned_ms": [round(x, 1) for x in pinned], "plain_ms": [round(x, 1) for x in plain],
                "chat_ms": [round(x, 1) for x in chats], "done": last["done"]},
    }
    text = json.dumps(summary, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0 if summary["floor_1000ms"] and summary["floor_3x"] and summary["chat_identical_after_decisions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
