"""Tests that need no API key: marker parsing and the convergence rule.

Run with:  python -m unittest test_debate -v
"""

import io
import contextlib
import unittest

import local_debate as ld


class TestSplitMarker(unittest.TestCase):
    def test_marker_at_end(self):
        self.assertEqual(ld.split_marker("blah\n[CONVERGED]"), ("blah", True))
        self.assertEqual(ld.split_marker("blah\n[OPEN]"), ("blah", False))

    def test_marker_at_start(self):
        # Local models often lead with the marker instead of ending on it.
        self.assertEqual(ld.split_marker("[CONVERGED]\n\nblah"), ("blah", True))
        self.assertEqual(ld.split_marker("[OPEN]\n\nblah"), ("blah", False))

    def test_no_marker_is_not_converged(self):
        self.assertEqual(ld.split_marker("no marker"), ("no marker", False))

    def test_last_marker_wins(self):
        self.assertEqual(ld.split_marker("[OPEN] mid text [CONVERGED]"), ("mid text", True))


class StubAgent(ld.Agent):
    """Emits a scripted marker per round instead of calling an API."""

    def __init__(self, name: str, script: list[str]):
        super().__init__(name, None, "stub")
        self.script = script
        self.turn = 0

    def speak(self, question, transcript):
        out = f"turn {self.turn + 1} of {self.name}\n{self.script[self.turn]}"
        self.turn += 1
        return out

    def chat(self, messages, temperature=0.7, retries=3):
        return "(stub summary)"


class TestConvergence(unittest.TestCase):
    C = "[CONVERGED]"
    O = "[OPEN]"

    def run_debate(self, a_script, b_script, max_rounds=4, min_rounds=2):
        ld.build_agents = lambda: (StubAgent("A", a_script), StubAgent("B", b_script))
        with contextlib.redirect_stdout(io.StringIO()):
            return ld.run_debate("q", max_rounds, min_rounds)

    def test_both_converge_ends_early(self):
        r = self.run_debate([self.O, self.C, self.C, self.C], [self.O, self.C, self.C, self.C])
        self.assertEqual((r["rounds_run"], r["converged_at"]), (2, 2))

    def test_one_sided_convergence_runs_full_length(self):
        r = self.run_debate([self.C] * 4, [self.O] * 4)
        self.assertEqual((r["rounds_run"], r["converged_at"]), (4, None))

    def test_min_rounds_blocks_instant_agreement(self):
        # Both agree in round 1, but min_rounds=2 forces a real exchange first.
        r = self.run_debate([self.C] * 4, [self.C] * 4)
        self.assertEqual((r["rounds_run"], r["converged_at"]), (2, 2))


if __name__ == "__main__":
    unittest.main()
