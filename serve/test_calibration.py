"""Decision calibration (serve/calibration.py): monotone maps keyed by the fitted request.
python -m unittest serve.test_calibration"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from serve import calibration, decision  # noqa: E402


def field(desc="d", **kw):
    return {"description": desc, **kw}


class Spec:
    def __init__(self, name, type_):
        self.spec = type("S", (), {"name": name, "type": type_})()


class Keys(unittest.TestCase):
    def test_the_two_schema_syntaxes_hash_equal_and_the_prompt_text_tells_them_apart(self):
        compact = decision.compile_schema({"e": field("kind", type="enum", choices=["a", "b"]),
                                           "b": field("flag", type="boolean")})
        js = decision.compile_schema({"type": "object", "properties": {
            "e": {"description": "kind", "enum": ["a", "b"]}, "b": {"description": "flag", "type": "boolean"}}})
        self.assertEqual(calibration.key(compact, "x"), calibration.key(js, "x"))
        self.assertNotEqual(calibration.key(compact, "x"), calibration.key(compact, "y"))      # instructions
        other = decision.compile_schema({"e": field("KIND", type="enum", choices=["a", "b"]),
                                         "b": field("flag", type="boolean")})
        self.assertNotEqual(calibration.key(compact, "x"), calibration.key(other, "x"))        # a description
        swapped = decision.compile_schema({"e": field("kind", type="enum", choices=["b", "a"]),
                                           "b": field("flag", type="boolean")})
        self.assertNotEqual(calibration.key(compact, "x"), calibration.key(swapped, "x"))      # value order
        self.assertEqual(len(calibration.key(compact, "x")), 64)


class Loading(unittest.TestCase):
    def setUp(self):
        calibration.reset()

    def tearDown(self):
        calibration.reset()

    def write(self, obj):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "calibration.json")
        with open(p, "w", encoding="utf-8") as f:
            f.write(obj if isinstance(obj, str) else json.dumps(obj))
        return p

    def test_a_missing_or_broken_file_is_identity(self):
        self.assertEqual(calibration.load(os.path.join(tempfile.mkdtemp(), "none.json")), {})
        self.assertEqual(calibration.load(self.write("{not json")), {})
        self.assertEqual(calibration.load(self.write([1, 2])), {})

    def test_malformed_entries_stay_identity_and_good_ones_load(self):
        table = calibration.load(self.write({"k": {
            "good_p": {"platt": [1.5]}, "good_t": {"temperature": 2.0},
            "flat": {"platt": [0.0]},                      # a = 0 would erase the ranking
            "neg": {"platt": [-1.0]},                      # a < 0 would flip the argmax
            "shift": {"platt": [1.0, 0.7]},                # an intercept would move the 0.5 crossing: not a parameter
            "cold": {"temperature": 0}, "two": {"platt": [1], "temperature": 1}, "none": 5}}))
        self.assertEqual(set(table["k"]), {"good_p", "good_t"})
        self.assertEqual(calibration.load(self.write({"k": {"neg": {"platt": [-1.0]}}})), {})


class Maps(unittest.TestCase):
    def test_platt_keeps_the_order_the_argmax_and_sums_to_one(self):
        ps = [0.05, 0.3, 0.49, 0.5, 0.51, 0.8, 0.99]
        for a in (0.3, 1.0, 4.0):
            out = [calibration.platt([p, 1 - p], a) for p in ps]
            self.assertEqual([x[0] for x in out], sorted(x[0] for x in out))
            for p, x in zip(ps, out):
                self.assertAlmostEqual(sum(x), 1.0, places=9)
                self.assertEqual(x[0] > 0.5, p > 0.5)                                # the chosen value is kept
        self.assertAlmostEqual(calibration.platt([0.5, 0.5], 3.0)[0], 0.5, places=9)
        self.assertAlmostEqual(calibration.platt([0.6, 0.4], 1.0)[0], 0.6, places=9)   # a = 1: identity
        self.assertGreater(calibration.platt([0.6, 0.4], 4.0)[0], 0.6)                 # a > 1: sharper
        self.assertLess(calibration.platt([0.6, 0.4], 0.3)[0], 0.6)                    # a < 1: softer

    def test_temperature_keeps_the_argmax_and_sums_to_one(self):
        pr = [0.1, 0.6, 0.25, 0.05]
        for t in (0.5, 1.0, 3.0):
            out = calibration.temperature(pr, t)
            self.assertEqual(out.index(max(out)), 1)
            self.assertAlmostEqual(sum(out), 1.0, places=9)
            self.assertEqual([i for i, _ in sorted(enumerate(out), key=lambda x: x[1])],
                             [i for i, _ in sorted(enumerate(pr), key=lambda x: x[1])])
        self.assertLess(max(calibration.temperature(pr, 3.0)), 0.6)                # hotter: flatter
        self.assertGreater(max(calibration.temperature(pr, 0.5)), 0.6)             # colder: sharper


class Applying(unittest.TestCase):
    def tearDown(self):
        calibration.reset()

    def test_apply_is_identity_without_an_entry_and_leaves_greedy_fields_alone(self):
        plans = [Spec("b", "boolean"), Spec("e", "enum"), Spec("g", "enum")]
        probs = [[0.7, 0.3], [0.2, 0.8], None]
        calibration.reset({})
        self.assertEqual(calibration.apply("k", plans, probs), probs)
        calibration.reset({"k": {"b": {"platt": [2.0]}, "e": {"temperature": 2.0}, "g": {"temperature": 2.0}}})
        out = calibration.apply("k", plans, probs)
        self.assertNotEqual(out[0], probs[0])
        self.assertEqual(out[0].index(max(out[0])), 0)                     # the argmax stays
        self.assertNotEqual(out[1], probs[1])
        self.assertEqual(out[1].index(max(out[1])), 1)
        self.assertIsNone(out[2])                                          # greedy: no distribution
        self.assertEqual(calibration.apply("other", plans, probs), probs)   # another request: identity
        self.assertEqual(calibration.entries(), 3)

    def test_a_platt_entry_on_an_enum_or_a_temperature_on_a_boolean_is_ignored(self):
        plans = [Spec("b", "boolean"), Spec("e", "enum")]
        probs = [[0.7, 0.3], [0.2, 0.8]]
        calibration.reset({"k": {"b": {"temperature": 2.0}, "e": {"platt": [2.0]}}})
        self.assertEqual(calibration.apply("k", plans, probs), probs)

    def test_the_distribution_log_is_opt_in_and_carries_no_text(self):
        plans = [Spec("b", "boolean"), Spec("g", "enum")]
        d = tempfile.mkdtemp()
        p = os.path.join(d, "log.jsonl")
        old = os.environ.pop("STRATA_DECISION_LOG", None)
        try:
            calibration.log_distributions("k", plans, [[0.7, 0.3], None])
            self.assertFalse(os.path.exists(p))
            os.environ["STRATA_DECISION_LOG"] = p
            calibration.log_distributions("k", plans, [[0.7, 0.3], None], "the text")
            calibration.log_distributions("k", plans, [[0.2, 0.8], None], "other text")
            rows = [json.loads(x) for x in open(p, encoding="utf-8")]
            cid = calibration.context_id("the text")
            self.assertEqual(len(cid), 16)
            self.assertEqual(rows, [{"key": "k", "context": cid, "fields": {"b": [0.7, 0.3]}},
                                    {"key": "k", "context": calibration.context_id("other text"),
                                     "fields": {"b": [0.2, 0.8]}}])
            self.assertNotIn("the text", open(p, encoding="utf-8").read())
        finally:
            os.environ.pop("STRATA_DECISION_LOG", None)
            if old is not None:
                os.environ["STRATA_DECISION_LOG"] = old

    def test_the_shipped_file_loads(self):
        table = calibration.load()
        self.assertIsInstance(table, dict)
        for k, fields in table.items():
            self.assertEqual(len(k), 64)
            for e in fields.values():
                self.assertTrue(calibration._valid(e))
                self.assertTrue(math.isfinite(e.get("temperature", 1.0)))


if __name__ == "__main__":
    unittest.main()
