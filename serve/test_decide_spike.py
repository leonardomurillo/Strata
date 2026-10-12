"""The engine side of decisions (StrataEngine.score, the score= key): python -m unittest serve.test_decide_spike."""
from __future__ import annotations

import io
import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from serve.server import EngineDied, StrataEngine  # noqa: E402


class Score(unittest.TestCase):
    def engine(self, *lines):
        e = StrataEngine.__new__(StrataEngine)
        e.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None, wait=lambda timeout=None: 1, returncode=1)
        e.lines, e.can_stop, e.max_context, e.silence_s = queue.Queue(), False, 262144, 0
        e.alive = lambda: True
        for line in lines:
            e.lines.put(line)
        return e

    def test_one_request_carries_every_branch_and_reads_its_lines(self):
        e = self.engine("PP 4 4", "REUSED 0", "DS 0 65:-0.010000 66:-4.600000", "DS 1 nan",
                        "DS 2 70:-0.500000", "DONE 0 4 1.0 0.0 score 0 0 0")
        out, done = e.score([1, 2, 3, 4], [([10, 11], [65, 66]), ([], [67]), ([12], [70])], threading.Event())
        sent = e.proc.stdin.getvalue()
        self.assertTrue(sent.startswith("GEN 1 ckpt=0 score=10,11:65,66;:67;12:70 1,2,3,4\n"), sent)
        self.assertEqual(out, [{65: -0.01, 66: -4.6}, None, {70: -0.5}])
        self.assertIn(" score ", done)

    def test_a_pin_keeps_the_request_checkpointed(self):
        e = self.engine("DS 0 65:-0.1", "DONE 0 4 1.0 0.0 score 0 0 0")
        e.score([1, 2, 3, 4], [([10], [65])], threading.Event(), pin=2)
        self.assertTrue(e.proc.stdin.getvalue().startswith("GEN 1 pin=2 score=10:65 "))

    def test_a_refused_decision_is_the_requests_fault_any_other_error_the_engines(self):
        with self.assertRaises(ValueError):
            self.engine("ERR score: branch 0 runs past the context").score([1, 2], [([3], [4])], threading.Event())
        with self.assertRaises(EngineDied):
            self.engine("ERR a CUDA error").score([1, 2], [([3], [4])], threading.Event())
        with self.assertRaises(EngineDied):
            self.engine(None).score([1, 2], [([3], [4])], threading.Event())


if __name__ == "__main__":
    unittest.main()
