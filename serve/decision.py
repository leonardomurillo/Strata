"""POST /v1/decision: answer a finite JSON schema by scoring token paths instead of generating them.

The algorithm is ported from thecodacus/llama.cpp tools/parallel-decision (MIT, the ggml authors): every field has a
finite set of allowed values; each value is a token path, the paths form a trie, and the model's log-probabilities at
the trie's divergence nodes give each value's probability (`tree`), or one branch per field per round walks the trie
(`greedy`).  Every field is scored as the first key after `{`, so fields never see each other.  The engine side is
StrataEngine.score(): the branches follow the prompt's last token and nothing is decoded.

Nothing the caller writes becomes a special token: the system text (field names, descriptions, values, instructions)
and every context are encoded as plain-text spans, and so is each branch.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

MAX_FIELDS, MAX_VALUES, MAX_CONTEXTS = 32, 255, 64
MAX_NAME, MAX_VALUE, MAX_TEXT, MAX_SYSTEM = 64, 256, 16384, 262144   # bytes; the instructions + catalogue, in chars
MAX_WINDOWS, MAX_BRANCH_TOKENS, DEFAULT_ROWS = 256, 256, 4
MAX_LABELS, DEFAULT_LABELS = 10, "words"                    # "numbers": options are scored as the labels 0-9
MARK_L, MARK_R = chr(0xF0EA0), chr(0xF0EA1)                 # private use: not whitespace, so no template trims it
SYSTEM_MARK, CONTEXT_MARK = MARK_L + "decision-system" + MARK_R, MARK_L + "decision-context" + MARK_R
SYSTEM_HEAD = ("Select the requested field value from its allowed values, based on the context. "
               "Respond with the JSON value only.\n\nFields:\n")
NUMBERS_HEAD = ("Select the requested field value from its allowed values, based on the context. "
                "Where the allowed values are numbered, respond with the number of the chosen value only; "
                "otherwise respond with the JSON value only.\n\nFields:\n")


class NonFinite(ArithmeticError):
    """A scored row held a NaN or an infinity (the engine said `nan`): not the request's fault."""


def _json_text(s) -> str:
    return json.dumps(s, ensure_ascii=False)


def _ids(tok, text: str, parse_special=False, plain=None) -> list[int]:
    """`text` as ordinary text (one plain span) unless `plain` names the spans to keep plain."""
    return tok.encode(text, parse_special=parse_special, plain=[(0, len(text))] if plain is None else plain)


# ----------------------------------------------------------------------------------------------- schema compiler
class _Spec:
    """One compiled field: its allowed values (as JSON values), their encoded text, and how to aggregate."""

    def __init__(self, name, type_, description, values, encoded, numbers, aggregate, shown=None, labelled=False):
        self.name, self.type, self.description = name, type_, description
        self.values, self.encoded, self.numbers, self.aggregate = values, encoded, numbers, aggregate
        self.shown, self.labelled = encoded if shown is None else shown, labelled   # shown: the options' JSON text


def _places(x: float) -> int:
    for k in range(10):
        v = x * 10 ** k
        if abs(v - round(v)) < 1e-9 * max(1.0, abs(v)):
            return k
    return 9


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _make_field(name: str, spec: dict, json_schema: bool, labels: str = DEFAULT_LABELS) -> _Spec:
    err = lambda why: ValueError(f'field "{name}": {why}')          # noqa: E731
    if len(name.encode("utf-8")) > MAX_NAME or not name:
        raise ValueError(f"a field name is 1-{MAX_NAME} bytes")
    typ = "enum" if "enum" in spec else spec.get("type")
    description = spec.get("description", "")
    if not isinstance(description, str) or len(description.encode("utf-8")) > MAX_TEXT:
        raise err(f"description: a string of at most {MAX_TEXT} bytes")
    if not json_schema and not description:
        raise err("needs a description")
    numbers: list[float] = []
    if typ == "boolean":
        values, encoded = [True, False], ["true", "false"]
    elif typ in ("enum", "choice", "selection"):
        choices = spec.get("enum") if "enum" in spec else spec.get("choices")
        if not isinstance(choices, list) or not all(isinstance(c, str) for c in choices):
            raise err("enum fields need a list of string choices")
        values, encoded = list(choices), [_json_text(c) for c in choices]
    elif typ == "integer":
        lo, hi = spec.get("minimum"), spec.get("maximum")
        if not (_is_int(lo) and _is_int(hi)):
            raise err("integer fields need integer minimum and maximum")
        if hi < lo or hi - lo + 1 > MAX_VALUES:
            raise err(f"integer bounds must define 1-{MAX_VALUES} values")
        values = list(range(lo, hi + 1))
        encoded, numbers = [str(v) for v in values], [float(v) for v in values]
    elif typ == "number":
        key = "multipleOf" if json_schema else "step"
        lo, hi, step = spec.get("minimum"), spec.get("maximum"), spec.get(key)
        if not (_is_num(lo) and _is_num(hi) and _is_num(step)):
            raise err(f"number fields need minimum, maximum and {key}")
        if not step > 0 or not hi >= lo:
            raise err("number needs ordered bounds and a positive step")
        count = (hi - lo) / step
        n = round(count)
        if abs(count - n) > 1e-7 or n < 0 or n > MAX_VALUES - 1:
            raise err(f"the number grid must include both ends and hold 1-{MAX_VALUES} values")
        places = max(_places(lo), _places(hi), _places(step))
        encoded = ["%.*f" % (places, lo + i * step) for i in range(n + 1)]   # fixed width: one shape for every value
        values = numbers = [float(e) for e in encoded]
    else:
        raise err("supported types are boolean, enum, integer and number")
    if not 1 <= len(values) <= MAX_VALUES:
        raise err(f"needs 1-{MAX_VALUES} allowed values")
    if any(len(e.encode("utf-8")) > MAX_VALUE for e in encoded):
        raise err(f"a value is at most {MAX_VALUE} bytes")
    if len(set(encoded)) != len(encoded):
        raise err("has duplicate allowed values")
    agg = spec.get("aggregate", spec.get("x-aggregate", "mode"))
    if agg != "mode" and not (numbers and agg in ("median", "mean")):
        raise err("aggregate must be mode, or median/mean for numeric fields")
    shown, labelled = encoded, False
    if labels == "numbers":
        if len(values) > MAX_LABELS:
            raise err(f"numbered labels allow at most {MAX_LABELS} allowed values")
        if typ in ("boolean", "enum", "choice", "selection"):
            encoded, labelled = [str(k) for k in range(len(values))], True
    return _Spec(name, typ if typ in ("boolean", "integer", "number") else "enum", description, values, encoded,
                 numbers, agg, shown, labelled)


def compile_schema(schema, labels: str = DEFAULT_LABELS) -> list[_Spec]:
    if not isinstance(schema, dict):
        raise ValueError('"schema" must be an object')
    json_schema = "properties" in schema
    props = schema["properties"] if json_schema else schema
    if not isinstance(props, dict) or not 1 <= len(props) <= MAX_FIELDS:
        raise ValueError(f"the schema must define 1-{MAX_FIELDS} fields")
    out = []
    for name, spec in props.items():
        if not isinstance(spec, dict):
            raise ValueError(f'field "{name}" must be an object')
        out.append(_make_field(name, spec, json_schema, labels))
    return out


# ------------------------------------------------------------------------------------------------- trie and maths
def _split(tok, suffix_text: str, encoded: list[str]):
    """-> (suffix tokens, one token path per value).  Each `suffix + value + "\\n"` is tokenized whole and cut at the
    longest token prefix every value shares (keeping one token per path), so the first scored token is the one the
    model would write itself."""
    common_text = os.path.commonprefix(encoded)
    seqs = [_ids(tok, suffix_text + e[len(common_text):] + "\n") for e in encoded]
    common = len(seqs[0])
    for s in seqs:
        c = 0
        while c < min(common, len(s)) and s[c] == seqs[0][c]:
            c += 1
        common = min(common, c, len(s) - 1)
    paths = [s[common:] for s in seqs]
    for a in range(len(paths)):
        for b in range(a):
            m = min(len(paths[a]), len(paths[b]))
            if paths[a][:m] == paths[b][:m]:
                raise ValueError("two allowed values tokenise to colliding paths")
    return seqs[0][:common], paths


def _build_nodes(paths):
    """The trie's divergence nodes: [(token prefix, allowed next tokens)]."""
    nodes, stack = [], [(list(range(len(paths))), 0)]
    while stack:
        act, depth = stack.pop()
        if len(act) <= 1:
            continue
        opts = []
        for i in act:
            if depth >= len(paths[i]):
                raise ValueError("candidate paths must be distinct and terminated")
            if paths[i][depth] not in opts:
                opts.append(paths[i][depth])
        if len(opts) > 1:
            nodes.append((paths[act[0]][:depth], opts))
        for t in opts:
            stack.append(([i for i in act if paths[i][depth] == t], depth + 1))
    return nodes


def finish_tree(paths, nodes, node_scores):
    """Exact constrained distribution: log-softmax at each node over its allowed tokens, summed along every path,
    then a softmax over the paths.  -> probabilities of the values."""
    node_lp = []
    for s in node_scores:
        mx = max(s)
        lz = mx + math.log(sum(math.exp(x - mx) for x in s))
        node_lp.append([x - lz for x in s])
    path_lp = [0.0] * len(paths)
    for i, p in enumerate(paths):
        for (pre, opts), lp in zip(nodes, node_lp):
            if len(pre) < len(p) and p[:len(pre)] == pre:
                path_lp[i] += lp[opts.index(p[len(pre)])]
    best = max(path_lp)
    z = sum(math.exp(x - best) for x in path_lp)
    return [math.exp(x - best) / z for x in path_lp]


class _Walk:
    """Greedy: follow single-path edges for free, score the options at the next divergence, keep the best."""

    def __init__(self, paths):
        self.paths, self.active, self.chosen = paths, list(range(len(paths))), []
        self.winner, self.score, self.scored = -1, 1.0, 0

    def options(self) -> list[int]:
        while len(self.active) > 1:
            depth, opts = len(self.chosen), []
            for i in self.active:
                if depth >= len(self.paths[i]):
                    raise ValueError("candidate paths must be distinct and terminated")
                if self.paths[i][depth] not in opts:
                    opts.append(self.paths[i][depth])
            if len(opts) > 1:
                return opts
            self.chosen.append(opts[0])
        self.winner = self.active[0] if self.active else -1
        return []

    def select(self, tok, p):
        depth = len(self.chosen)
        self.active = [i for i in self.active if self.paths[i][depth] == tok]
        self.chosen.append(tok)
        self.score *= p
        self.scored += 1


class _Plan:
    """One field, ready to score: the spec, its suffix and paths, and tree or greedy."""

    def __init__(self, tok, spec: _Spec, mode: str, tree_max: int):
        text = "  " + _json_text(spec.name) + ": " + os.path.commonprefix(spec.encoded)
        self.spec = spec
        self.suffix, self.paths = _split(tok, text, spec.encoded)
        if not self.suffix:
            raise ValueError("a field suffix must not be empty")
        self.nodes = _build_nodes(self.paths)
        self.tree = mode == "tree" or (mode == "auto" and len(spec.encoded) <= tree_max)
        longest = max((len(pre) for pre, _ in self.nodes), default=0) if self.tree else \
            max(len(p) for p in self.paths) - 1
        self.longest = len(self.suffix) + longest       # the longest branch's tokens
        self.branches = max(1, len(self.nodes)) if self.tree else 1

    def windows(self, rows: int) -> int:
        """Engine windows the field's first round costs (a branch is the prompt's last token + its tokens)."""
        if not self.tree:
            return -(-(1 + self.longest) // rows)
        return sum(-(-(1 + len(self.suffix) + len(pre)) // rows) for pre, _ in self.nodes)


# ----------------------------------------------------------------------------------------------- the request
def _text(req, key, limit=MAX_TEXT) -> str:
    v = req.get(key, "")
    if not isinstance(v, str) or len(v.encode("utf-8")) > limit:
        raise ValueError(f'"{key}": a string of at most {limit} bytes')
    return v


def prepare(svc, req: dict) -> dict:
    """Validate the request and build every context's prompt: no engine needed, so a bad request never waits for
    the turn.  Raises ValueError (HTTP 400)."""
    schema = req.get("schema")
    instructions = _text(req, "instructions")
    mode, tree_max = req.get("mode", "auto"), req.get("tree_max", 128)
    if mode not in ("auto", "tree", "greedy"):
        raise ValueError('"mode" must be auto, tree or greedy')
    if not _is_int(tree_max) or not 1 <= tree_max <= MAX_VALUES:
        raise ValueError(f'"tree_max" must be a whole number 1-{MAX_VALUES}')
    contexts = req.get("contexts")
    if not isinstance(contexts, list) or not 1 <= len(contexts) <= MAX_CONTEXTS:
        raise ValueError(f'"contexts" must be a list of 1-{MAX_CONTEXTS} strings')
    if not all(isinstance(c, str) and c.strip() for c in contexts):
        raise ValueError("every context must be a non-empty string")
    labels = req.get("labels", DEFAULT_LABELS)
    if labels not in ("words", "numbers"):
        raise ValueError('"labels" must be words or numbers')
    specs = compile_schema(schema, labels)
    allowed = lambda s: ", ".join(f"{k} = {t}" for k, t in enumerate(s.shown)) if s.labelled else \
        ", ".join(s.encoded)                                        # noqa: E731
    catalog = "\n".join(_json_text(s.name) + (": " + s.description if s.description else "") +
                        "\nAllowed values: " + allowed(s) for s in specs)
    system = (NUMBERS_HEAD if labels == "numbers" else SYSTEM_HEAD) + catalog + "\n" + instructions
    if len(system) > MAX_SYSTEM:
        raise ValueError(f"the schema and instructions are too long ({MAX_SYSTEM} characters at most)")
    plans = [_Plan(svc.tok, s, mode, tree_max) for s in specs]
    if any(p.spec.labelled and len({q[0] for q in p.paths}) != len(p.paths) for p in plans):
        raise ValueError("numbered labels must be single tokens")
    score_rows = getattr(svc.engine, "score_rows", None)
    rows = score_rows if _is_int(score_rows) and score_rows >= 1 else \
        max(1, int(getattr(svc.engine, "spec_rows", DEFAULT_ROWS) or DEFAULT_ROWS))
    windows = len(contexts) * sum(p.windows(rows) for p in plans)
    if windows > MAX_WINDOWS:
        raise ValueError(f"too many branch windows ({windows}; {MAX_WINDOWS} at most): fewer contexts, fields or "
                         "values, or mode greedy")
    if max(p.longest for p in plans) > MAX_BRANCH_TOKENS:
        raise ValueError(f"a field's suffix and value are over {MAX_BRANCH_TOKENS} tokens")
    # the chat template with the thinking off; the caller's text goes in afterwards, as plain spans
    rendered = svc.render_prompt([{"role": "system", "content": SYSTEM_MARK},
                                  {"role": "user", "content": CONTEXT_MARK}], None, {"enable_thinking": False})
    a, b = rendered.find(SYSTEM_MARK), rendered.find(CONTEXT_MARK)
    if not 0 <= a < b:
        raise ValueError("the chat template did not keep the system and user messages")
    head, mid, tail = rendered[:a], rendered[a + len(SYSTEM_MARK):b], rendered[b + len(CONTEXT_MARK):] + "{\n"
    prompts = []
    for c in contexts:
        text = head + system + mid + c + tail
        s0 = len(head) + len(system) + len(mid)
        prompts.append(_ids(svc.tok, text, True, [(len(head), len(head) + len(system)), (s0, s0 + len(c))]))
    return {"plans": plans, "prompts": prompts, "mode": mode, "windows": windows}


def run(svc, job: dict) -> dict:
    """Score every context (the caller holds the engine's turn) and assemble the answer."""
    from serve.server import EngineNonFinite                    # not at import: server.py imports this module
    engine, plans = svc.engine, job["plans"]
    if getattr(engine, "batch", 0):
        raise ValueError("decisions need an engine without parallel slots (no --batch)")
    limit = int(getattr(engine, "max_context", 0) or 0)
    longest = max(p.longest for p in plans)
    for ids in job["prompts"]:
        if limit and len(ids) + longest > limit:
            raise ValueError(f"a context and the schema need {len(ids) + longest} tokens; the context window is {limit}")
    t0 = time.perf_counter()
    results, prefill_ms, scoring_ms, rounds, rows, cached = [], 0.0, 0.0, 0, 0, 0
    cancel = threading.Event()  # a client that hangs up does not stop its decision; chat watches the socket, this path does not yet
    prompt_tokens = 0
    for ids in job["prompts"]:
        prompt_tokens += len(ids)
        walks = [None if p.tree else _Walk(p.paths) for p in plans]
        probs: list = [None] * len(plans)
        ctx_rows, first = 0, True
        while True:
            todo, owner = [], []
            for f, p in enumerate(plans):
                if p.tree:
                    if first:
                        for pre, opts in p.nodes:
                            todo.append((p.suffix + pre, opts))
                            owner.append(f)
                else:
                    opts = walks[f].options()
                    if opts:
                        todo.append((p.suffix + walks[f].chosen, opts))
                        owner.append(f)
            if todo:
                scored, done = engine.score(ids, todo, cancel, pin=0)
                rounds += 1
                ctx_rows += sum(len(t) for t, _ in todo)
                try:   # DONE <gen> <prompt tokens> <prompt ms> <decode ms> <finish> <drafts ok> <drafts> <reused> ...
                    fields_ = done.split()
                    prefill_ms += float(fields_[3])
                    scoring_ms += float(fields_[4])
                    cached += int(fields_[8])
                except (IndexError, ValueError):
                    pass
            node_scores = [[] for _ in plans]
            for f, (_, opts), r in zip(owner, todo, scored if todo else []):
                if r is None:
                    raise EngineNonFinite("a decision's scores were not finite")
                s = [r[c] for c in opts]
                if plans[f].tree:
                    node_scores[f].append(s)
                else:
                    best = max(range(len(s)), key=s.__getitem__)
                    walks[f].select(opts[best], 1.0 / sum(math.exp(x - s[best]) for x in s))
            if first:
                for f, p in enumerate(plans):
                    if p.tree:
                        probs[f] = finish_tree(p.paths, p.nodes, node_scores[f]) if p.nodes else [1.0]
            first = False
            if not todo:
                break
        results.append((ids, probs, walks, ctx_rows))
        rows += ctx_rows
    total_ms = (time.perf_counter() - t0) * 1000.0
    out = [_assemble(plans, probs, walks) | {"usage": {"context_tokens": len(ids), "scored_rows": r}}
           for ids, probs, walks, r in results]
    return {"object": "decision", "model": svc.model, "created": int(time.time()), "results": out,
            "usage": {"prompt_tokens": prompt_tokens, "cached_tokens": cached,
                      "context_tokens": prompt_tokens, "scored_rows": rows},
            "timings": {"prefill_ms": round(prefill_ms, 1), "scoring_ms": round(scoring_ms, 1),
                        "total_ms": round(total_ms, 1), "rounds": rounds,
                        "per_decision_ms": round(total_ms / len(results), 1)}}


def _assemble(plans, probs, walks) -> dict:
    """mode = argmax; median = where the CDF crosses 0.5; mean = the grid value nearest the expectation; numeric tree
    fields also get the p10-p90 interval.  Greedy fields have no distribution: median/mean give the greedy winner."""
    decision, fields = {}, {}
    for f, p in enumerate(plans):
        sp, pr, w = p.spec, probs[f], walks[f]
        idx = max(range(len(pr)), key=pr.__getitem__) if pr else w.winner
        entry = {}
        if sp.numbers and pr:
            order = sorted(range(len(sp.numbers)), key=sp.numbers.__getitem__)

            def quantile(q):
                acc = 0.0
                for k in order:
                    acc += pr[k]
                    if acc >= q:
                        return k
                return order[-1]
            if sp.aggregate == "median":
                idx = quantile(0.5)
            elif sp.aggregate == "mean":
                mean = sum(n * q for n, q in zip(sp.numbers, pr))
                idx = 0
                for k in range(1, len(sp.numbers)):
                    if abs(sp.numbers[k] - mean) < abs(sp.numbers[idx] - mean):
                        idx = k
            entry["interval_p10_p90"] = [sp.values[quantile(0.1)], sp.values[quantile(0.9)]]
        if not 0 <= idx < len(sp.values):
            raise ValueError(f'field "{sp.name}" has no selected value')
        decision[sp.name] = sp.values[idx]
        fields[sp.name] = {"value": sp.values[idx], "probability": pr[idx] if pr else w.score,
                           "scored_nodes": len(p.nodes) if p.tree else w.scored, "tree": p.tree, **entry}
        if pr:
            fields[sp.name]["options"] = [{"value": v, "probability": q} for v, q in zip(sp.values, pr)]
    return {"decision": decision, "fields": fields}
