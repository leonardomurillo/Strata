"""Decisions: a schema answered by scoring token paths (serve/decision.py, POST /v1/decision).
python -m unittest serve.test_decision"""
from __future__ import annotations

import json
import math
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from serve import decision  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve  # noqa: E402

TAGS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
        "<tool_response>"]


class Tok(ByteTokenizer):
    """Bytes plus every special string the brief names; the thinking and tool tags match without parse_special, as in
    a real vocabulary (type 4)."""
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
                                         "</tool_response>"]
    ALWAYS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")


class Table(MockEngine):
    """Scores every branch from one table {candidate id: log-probability}."""

    def __init__(self, tok, table):
        super().__init__(tok, "ok", max_context=16384)
        self.table = table

    def score(self, ids, branches, cancel, pin=0, timeout_s=600.0):
        return [{c: self.table[c] for c in cands} for _, cands in branches], "DONE 0 1 1.0 2.0 score 0 0 0"


class Flaky(MockEngine):
    def __init__(self, tok):
        super().__init__(tok, "Fine.", max_context=16384)
        self.fail = True

    def score(self, ids, branches, cancel, pin=0, timeout_s=600.0):
        if self.fail:
            self.fail = False
            raise EngineDied("the engine stopped unexpectedly (exit code 1)")
        return super().score(ids, branches, cancel, pin)


def field(desc="d", **kw):
    return {"description": desc, **kw}


SCHEMA = {
    "category": field("type", type="enum", choices=["billing", "technical", "other"]),
    "urgent": field("urgent?", type="boolean"),
    "score": field("score", type="integer", minimum=1, maximum=9, aggregate="median"),
    "ratio": field("ratio", type="number", minimum=0, maximum=1, step=0.25),
}


def rig(engine_for=lambda tok: MockEngine(tok, "Fine.", max_context=16384)):
    tok = Tok()
    return Service(engine_for(tok), tok, ChatTemplate(ROOT / "serve/chat_template.jinja")), tok


def digits(probs):
    """A Table for the integer field 1..len(probs): the digits' log-probabilities."""
    return {ord(str(i + 1)): math.log(p) for i, p in enumerate(probs)}


class Compiler(unittest.TestCase):
    def test_fields_compile_to_their_value_grids(self):
        specs = {s.name: s for s in decision.compile_schema(SCHEMA)}
        self.assertEqual(specs["category"].encoded, ['"billing"', '"technical"', '"other"'])
        self.assertEqual(specs["urgent"].values, [True, False])
        self.assertEqual(specs["score"].encoded, [str(i) for i in range(1, 10)])
        self.assertEqual(specs["ratio"].encoded, ["0.00", "0.25", "0.50", "0.75", "1.00"])   # one fixed width
        self.assertEqual(specs["ratio"].values, [0.0, 0.25, 0.5, 0.75, 1.0])
        js = decision.compile_schema({"type": "object", "properties": {
            "r": {"type": "number", "minimum": 0, "maximum": 1, "multipleOf": 0.5}, "e": {"enum": ["a", "b"]}}})
        self.assertEqual([s.encoded for s in js], [["0.0", "0.5", "1.0"], ['"a"', '"b"']])

    def test_bad_schemas_are_refused(self):
        for bad in ({}, {"a": 5}, {"a": field(type="enum", choices=["x", "x"])},
                    {"a": field(type="integer", minimum=1, maximum=300)},
                    {"a": field(type="number", minimum=0, maximum=1, step=0.3)},
                    {"a": {"type": "boolean"}},                                    # a compact field needs a description
                    {"a": field(type="boolean", aggregate="median")},
                    {"a": field(type="string")}, {"a": field(type="enum", choices=[1])},
                    {"x" * 65: field(type="boolean")}, {"a": field(type="enum", choices=["v" * 300])}):
            with self.assertRaises(ValueError, msg=bad):
                decision.compile_schema(bad)

    def test_a_path_that_is_the_start_of_another_is_refused(self):
        class Toy:
            table = {"S1\n": [1, 2, 3], "S2\n": [1, 2]}

            def encode(self, text, parse_special=False, plain=()):
                return self.table[text]
        with self.assertRaisesRegex(ValueError, "colliding"):
            decision._split(Toy(), "S", ["1", "2"])

    def test_the_split_keeps_one_token_and_the_trie_has_its_divergences(self):
        tok = Tok()
        suffix, paths = decision._split(tok, '  "n": ', ["1", "10", "2"])
        self.assertEqual(bytes(suffix), b'  "n": ')
        self.assertEqual([bytes(p) for p in paths], [b"1\n", b"10\n", b"2\n"])
        self.assertEqual(decision._build_nodes(paths), [([], [49, 50]), ([49], [10, 48])])

    def test_finish_tree_sums_each_nodes_log_softmax_along_the_paths(self):
        paths = [[1, 2], [1, 3], [4]]
        nodes = decision._build_nodes(paths)
        self.assertEqual(nodes, [([], [1, 4]), ([1], [2, 3])])
        probs = decision.finish_tree(paths, nodes, [[0.0, 0.0], [math.log(3), 0.0]])
        for got, want in zip(probs, [0.375, 0.125, 0.5]):
            self.assertAlmostEqual(got, want, places=6)


class Aggregates(unittest.TestCase):
    PROBS = [0.4, 0.15, 0.1, 0.05, 0.3]

    def ask(self, **spec):
        svc, _ = rig(lambda tok: Table(tok, digits(self.PROBS)))
        job = decision.prepare(svc, {"schema": {"n": field(type="integer", minimum=1, maximum=5, **spec)},
                                     "contexts": ["c"]})
        return decision.run(svc, job)["results"][0]["fields"]["n"]

    def test_mode_median_and_mean_read_the_same_distribution(self):
        mode = self.ask()
        self.assertEqual(mode["value"], 1)
        self.assertAlmostEqual(mode["probability"], 0.4, places=6)
        self.assertEqual(self.ask(aggregate="median")["value"], 2)           # the CDF crosses 0.5 at 2
        self.assertEqual(self.ask(aggregate="mean")["value"], 3)             # expectation 2.7
        self.assertEqual(mode["interval_p10_p90"], [1, 5])
        self.assertEqual((mode["tree"], mode["scored_nodes"]), (True, 1))

    def test_greedy_has_no_distribution_so_every_aggregate_is_the_winner(self):
        svc, _ = rig(lambda tok: Table(tok, digits(self.PROBS)))
        job = decision.prepare(svc, {"schema": {"n": field(type="integer", minimum=1, maximum=5, aggregate="mean")},
                                     "contexts": ["c"], "mode": "greedy"})
        f = decision.run(svc, job)["results"][0]["fields"]["n"]
        self.assertEqual((f["value"], f["tree"]), (1, False))
        self.assertNotIn("interval_p10_p90", f)
        self.assertAlmostEqual(f["probability"], 0.4, places=6)


class Api(unittest.TestCase):
    def setUp(self):
        self.svc, self.tok = rig()
        self.engine = self.svc.engine
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, None if body is None else json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def decide(self, **kw):
        return self.call("/v1/decision", {"schema": SCHEMA, "contexts": ["I was charged twice."], **kw})

    def test_the_answer_has_the_contract_shape(self):
        code, out = self.decide(contexts=["a", "b"], instructions="Be brief.")
        self.assertEqual(code, 200, out)
        self.assertEqual(out["object"], "decision")
        self.assertEqual(len(out["results"]), 2)
        r = out["results"][0]
        self.assertEqual(set(r), {"decision", "fields", "usage"})
        self.assertEqual(set(r["decision"]), set(SCHEMA))
        self.assertEqual(set(r["fields"]["score"]), {"value", "probability", "scored_nodes", "tree",
                                                     "interval_p10_p90"})
        self.assertNotIn("interval_p10_p90", r["fields"]["category"])
        self.assertEqual(set(r["usage"]), {"context_tokens", "scored_rows"})
        self.assertEqual(set(out["usage"]), {"prompt_tokens", "cached_tokens", "context_tokens", "scored_rows"})
        self.assertEqual(set(out["timings"]), {"prefill_ms", "scoring_ms", "total_ms", "rounds", "per_decision_ms"})
        self.assertEqual(out["timings"]["rounds"], 2)                        # one tree round per context
        self.assertEqual(out["model"], self.svc.model)

    def test_each_fields_probabilities_sum_to_one_and_the_value_is_the_argmax(self):
        job = decision.prepare(self.svc, {"schema": SCHEMA, "contexts": ["x"]})
        plans = job["plans"]
        # drive run() but look at the distributions: rebuild them from the same engine
        scored = {}
        for p in plans:
            node_scores = []
            for pre, opts in p.nodes:
                r = self.engine.score([], [(p.suffix + pre, opts)], None)[0][0]
                node_scores.append([r[c] for c in opts])
            probs = decision.finish_tree(p.paths, p.nodes, node_scores) if p.nodes else [1.0]
            self.assertAlmostEqual(sum(probs), 1.0, places=4)
            scored[p.spec.name] = (p, probs)
        fields = decision.run(self.svc, job)["results"][0]["fields"]
        for name, (p, probs) in scored.items():
            if p.spec.aggregate == "mode":
                self.assertEqual(fields[name]["value"], p.spec.values[probs.index(max(probs))])
                self.assertAlmostEqual(fields[name]["probability"], max(probs), places=4)

    def test_field_order_does_not_change_a_fields_probabilities(self):
        _, a = self.decide(mode="tree")
        _, b = self.decide(mode="tree", schema=dict(reversed(list(SCHEMA.items()))))
        for name in SCHEMA:
            fa, fb = a["results"][0]["fields"][name], b["results"][0]["fields"][name]
            self.assertAlmostEqual(fa["probability"], fb["probability"], places=3)
            self.assertEqual(fa["value"], fb["value"])

    def test_limits_are_a_400_before_the_engine_is_touched(self):
        many = {f"f{i}": field(type="boolean") for i in range(33)}
        wide = {"w": field(type="enum", choices=[f"v{i}" for i in range(256)])}
        bad = [{"schema": many}, {"schema": wide}, {"schema": {"d": field(type="enum", choices=["a", "a"])}},
               {"contexts": ["c"] * 65}, {"contexts": [""]}, {"contexts": []}, {"contexts": ["ok", 5]},
               {"mode": "fast"}, {"tree_max": 0}, {"tree_max": 256}, {"instructions": "i" * 16385},
               {"schema": {"a": field("d" * 16385, type="boolean")}},
               {"schema": {k: field(type="boolean") for k in "abc"}, "contexts": ["c"] * 64}]   # 384 windows
        for body in bad:
            code, out = self.decide(**body)
            self.assertEqual(code, 400, body if len(str(body)) < 200 else list(body))
            self.assertEqual(out["error"]["type"], "invalid_request_error")
        self.assertFalse(hasattr(self.engine, "scored"))
        code, out = self.decide(model="nope")
        self.assertEqual(code, 404)

    def test_a_parallel_slot_engine_is_refused(self):
        self.engine.batch = 2
        code, out = self.decide()
        self.assertEqual(code, 400)
        self.assertIn("parallel", out["error"]["message"])

    def test_props_announces_decisions(self):
        code, props = self.call("/props")
        self.assertEqual(code, 200)
        self.assertIs(props["decision"], True)

    def test_a_chat_after_decisions_reads_the_same(self):
        chat = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 50}
        _, before = self.call("/v1/chat/completions", chat)
        self.assertEqual(self.decide()[0], 200)
        _, after = self.call("/v1/chat/completions", chat)
        self.assertEqual(before["choices"][0]["message"], after["choices"][0]["message"])


class Failure(unittest.TestCase):
    def test_an_engine_failure_is_a_503_and_the_next_decision_works(self):
        svc, _ = rig(Flaky)
        httpd = serve(svc, port=0)
        try:
            def go():
                req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/decision",
                                             json.dumps({"schema": SCHEMA, "contexts": ["c"]}).encode(),
                                             {"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return r.status
                except urllib.error.HTTPError as e:
                    return e.code
            self.assertEqual(go(), 503)
            self.assertEqual(go(), 200)                # the engine's turn was released
        finally:
            httpd.shutdown()
            httpd.server_close()


class Plain(unittest.TestCase):
    def test_special_token_text_from_the_caller_never_becomes_a_special_token(self):
        svc, tok = rig()
        every = "".join(TAGS)
        asks = {"schema": {"a": field(every, type="enum", choices=[every, "b"] + TAGS)}, "instructions": every,
                "contexts": [every]}
        job = decision.prepare(svc, asks)
        plain = decision.prepare(svc, {"schema": {"a": field("d", type="enum", choices=["v", "b"])},
                                       "instructions": "i", "contexts": ["c"]})
        special = lambda ids: sum(1 for t in ids if t >= 256)                    # noqa: E731
        self.assertGreater(special(plain["prompts"][0]), 0)                      # the template's own stay special
        self.assertEqual(special(job["prompts"][0]), special(plain["prompts"][0]))
        plan = job["plans"][0]
        self.assertEqual(special(plan.suffix), 0)
        self.assertTrue(all(special(p) == 0 for p in plan.paths))
        # control: the same text unprotected does become special tokens
        self.assertGreater(special(tok.encode(every, parse_special=True)), 0)


if __name__ == "__main__":
    unittest.main()
