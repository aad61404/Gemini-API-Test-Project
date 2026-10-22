"""Tests that need no API key: marker parsing, the verdict probe, and the
convergence rule.

Run with:  python -m unittest test_debate -v
"""

import contextlib
import io
import os
import unittest

import local_debate as ld
import show_debate


class TestSplitMarker(unittest.TestCase):
    def test_marker_at_end(self):
        self.assertEqual(ld.split_marker("blah\n[CONVERGED]"), ("blah", True))
        self.assertEqual(ld.split_marker("blah\n[OPEN]"), ("blah", False))

    def test_marker_at_start(self):
        # Local models often lead with the marker instead of ending on it.
        self.assertEqual(ld.split_marker("[CONVERGED]\n\nblah"), ("blah", True))
        self.assertEqual(ld.split_marker("[OPEN]\n\nblah"), ("blah", False))

    def test_missing_marker_is_none_not_false(self):
        # "no verdict" and "not converged" are different; the caller decides.
        self.assertEqual(ld.split_marker("no marker"), ("no marker", None))

    def test_last_standalone_marker_wins(self):
        self.assertEqual(ld.split_marker("[OPEN]\nmid\n[CONVERGED]"), ("mid", True))

    def test_inline_mention_is_prose_not_a_verdict(self):
        # Quoting the marker mid-sentence must not be read as a verdict.
        text = "Last round I wrote [CONVERGED], but I now see a problem.\n[OPEN]"
        body, verdict = ld.split_marker(text)
        self.assertFalse(verdict)
        self.assertIn("[CONVERGED]", body)

    def test_inline_mention_alone_yields_no_verdict(self):
        text = "You said [CONVERGED] but I disagree."
        self.assertEqual(ld.split_marker(text), (text, None))


class TestShowDebate(unittest.TestCase):
    DATA = {
        "question": "q?",
        "model_a": "model-a", "model_b": "model-b",
        "rounds_run": 2, "converged_at": 2,
        "judge_rulings": [
            {"round": 1, "agreed": False, "note": "scope undefined", "ruled": True},
            {"round": 2, "agreed": True, "note": "none", "ruled": True},
        ],
        "transcript": [
            {"round": 1, "speaker": "Agent-A", "verdict": False, "text": "first"},
            {"round": 1, "speaker": "Agent-B", "verdict": None, "text": "second"},
            {"round": 2, "speaker": "Agent-A", "verdict": True, "text": "third"},
        ],
        "summary": "the summary",
        "error": None,
    }

    def test_render_groups_by_round_and_labels_verdicts(self):
        out = show_debate.render(self.DATA)
        self.assertIn("## Round 1", out)
        self.assertIn("## Round 2", out)
        self.assertIn("_OPEN_", out)
        self.assertIn("_no verdict_", out)
        self.assertIn("converged in round 2", out)
        self.assertIn("the summary", out)

    def test_render_shows_judge_rulings(self):
        out = show_debate.render(self.DATA)
        self.assertIn("overruled: scope undefined", out)
        self.assertIn("agreed: none", out)

    def test_render_marks_a_judge_that_could_not_rule(self):
        data = dict(self.DATA, judge_rulings=[
            {"round": 1, "agreed": True, "note": "judge unavailable: 429", "ruled": False}])
        self.assertIn("could not rule", show_debate.render(data))

    def test_render_reports_a_partial_debate(self):
        data = dict(self.DATA, converged_at=None, error="Agent-B: boom")
        out = show_debate.render(data)
        self.assertIn("without converging", out)
        self.assertIn("Stopped early", out)

    def test_stats_tallies_each_speaker(self):
        out = show_debate.render_stats(self.DATA)
        self.assertIn("Agent-A", out)
        self.assertIn("OPEN 1", out)
        self.assertIn("CONVERGED 1", out)


class TestBuildAgents(unittest.TestCase):
    """Guard the provider wiring: which side talks to which endpoint."""

    def setUp(self):
        # load_dotenv() must not pull the developer's own .env into these.
        self._real_load = ld.load_dotenv
        ld.load_dotenv = lambda *a, **k: None
        self._env = {k: os.environ.get(k) for k in
                     ("GEMINI_API_KEY", "GEMINI_MODEL", "OLLAMA_MODEL",
                      "OLLAMA_MODEL_A", "OLLAMA_BASE_URL")}
        for key in self._env:
            os.environ.pop(key, None)

    def tearDown(self):
        ld.load_dotenv = self._real_load
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_default_puts_agent_a_on_gemini(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["GEMINI_MODEL"] = "gemini-test"
        os.environ["OLLAMA_MODEL"] = "local-test"
        a, b = ld.build_agents()
        self.assertEqual((a.model, b.model), ("gemini-test", "local-test"))
        self.assertIn("googleapis.com", str(a.client.base_url))
        self.assertIn("11434", str(b.client.base_url))

    def test_local_only_needs_no_key_and_uses_ollama_for_both(self):
        os.environ["OLLAMA_MODEL"] = "local-test"
        a, b = ld.build_agents(local_only=True)
        self.assertEqual((a.model, b.model), ("local-test", "local-test"))
        for agent in (a, b):
            self.assertIn("11434", str(agent.client.base_url))

    def test_local_only_can_give_the_two_sides_different_models(self):
        os.environ["OLLAMA_MODEL"] = "local-b"
        os.environ["OLLAMA_MODEL_A"] = "local-a"
        a, b = ld.build_agents(local_only=True)
        self.assertEqual((a.model, b.model), ("local-a", "local-b"))

    def test_missing_key_exits_and_mentions_the_escape_hatch(self):
        with self.assertRaises(SystemExit) as cm:
            ld.build_agents()
        self.assertIn("--local-only", str(cm.exception))

    def test_local_only_judge_stays_local(self):
        os.environ["OLLAMA_MODEL"] = "local-test"
        judge = ld.build_judge(local_only=True)
        self.assertIn("11434", str(judge.client.base_url))


class TestRetryAfter(unittest.TestCase):
    def test_reads_the_delay_gemini_buries_in_the_payload(self):
        msg = ("Error code: 429 - [{'error': {'code': 429, 'details': "
               "[{'@type': 'type.googleapis.com/google.rpc.RetryInfo', "
               "'retryDelay': '17701s'}]}}]")
        self.assertEqual(ld.retry_after(RuntimeError(msg)), 17701)

    def test_short_rate_limit_delay(self):
        self.assertEqual(ld.retry_after(RuntimeError("'retryDelay': '12s'")), 12)

    def test_absent_delay_is_zero(self):
        self.assertEqual(ld.retry_after(RuntimeError("Error code: 429")), 0)


class ProbeAgent(ld.Agent):
    """Agent whose chat() replies are scripted, to exercise the verdict probe."""

    def __init__(self, replies: list[str]):
        super().__init__("Probe", None, "stub")
        self.replies = replies
        self.calls = 0

    def chat(self, messages, temperature=0.7, retries=3):
        reply = self.replies[self.calls]
        self.calls += 1
        return reply


class TestVerdictProbe(unittest.TestCase):
    def test_no_probe_when_marker_present(self):
        agent = ProbeAgent(["argument\n[OPEN]"])
        body, verdict = agent.speak("q", [])
        self.assertEqual((body, verdict), ("argument", False))
        self.assertEqual(agent.calls, 1, "should not spend a second call")

    def test_probe_recovers_a_missing_marker(self):
        agent = ProbeAgent(["argument with no marker", "[CONVERGED]"])
        body, verdict = agent.speak("q", [])
        self.assertEqual((body, verdict), ("argument with no marker", True))
        self.assertEqual(agent.calls, 2)

    def test_probe_accepts_an_unframed_token(self):
        agent = ProbeAgent(["argument", "My verdict is [OPEN]."])
        _, verdict = agent.speak("q", [])
        self.assertFalse(verdict)

    def test_ambiguous_probe_answer_yields_no_verdict(self):
        agent = ProbeAgent(["argument", "maybe [CONVERGED] or maybe [OPEN]"])
        with contextlib.redirect_stderr(io.StringIO()):
            _, verdict = agent.speak("q", [])
        self.assertIsNone(verdict)


class TestPreflight(unittest.TestCase):
    class Reachable(ld.Agent):
        def __init__(self, name="Agent-A (Gemini)"):
            super().__init__(name, None, "stub")

        def chat(self, messages, temperature=0.7, retries=3):
            return "OK"

    class Unreachable(ld.Agent):
        def __init__(self, name="Agent-A (Gemini)"):
            super().__init__(name, None, "missing-model")

        def chat(self, messages, temperature=0.7, retries=3):
            raise RuntimeError("404 model not found")

    def test_all_reachable_returns(self):
        with contextlib.redirect_stdout(io.StringIO()):
            ld.preflight((self.Reachable(), self.Reachable("Agent-B (Local)")))

    def test_unreachable_agent_exits_nonzero(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                ld.preflight((self.Reachable(), self.Unreachable("Agent-B (Local)")))
        self.assertEqual(cm.exception.code, 1)

    def test_failure_names_the_broken_agent_and_model(self):
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit):
                # Not a "(Local)" agent, so this stays offline: no Ollama probe.
                ld.preflight((self.Unreachable(),))
        self.assertIn("missing-model", err.getvalue())
        self.assertIn("404", err.getvalue())


class StubAgent(ld.Agent):
    """Emits a scripted verdict per round instead of calling an API."""

    def __init__(self, name: str, script: list):
        super().__init__(name, None, "stub")
        self.script = script
        self.turn = 0

    def speak(self, question, transcript):
        verdict = self.script[self.turn]
        self.turn += 1
        return f"turn {self.turn} of {self.name}", verdict

    def chat(self, messages, temperature=0.7, retries=3):
        return "(stub summary)"


class TestConvergence(unittest.TestCase):
    def run_debate(self, a_script, b_script, max_rounds=4, min_rounds=2):
        ld.build_agents = lambda *a, **k: (StubAgent("A", a_script), StubAgent("B", b_script))
        with contextlib.redirect_stdout(io.StringIO()):
            return ld.run_debate("q", max_rounds, min_rounds, check=False)

    def test_both_converge_ends_early(self):
        r = self.run_debate([False, True, True, True], [False, True, True, True])
        self.assertEqual((r["rounds_run"], r["converged_at"]), (2, 2))

    def test_one_sided_convergence_runs_full_length(self):
        r = self.run_debate([True] * 4, [False] * 4)
        self.assertEqual((r["rounds_run"], r["converged_at"]), (4, None))

    def test_min_rounds_blocks_instant_agreement(self):
        # Both agree in round 1, but min_rounds=2 forces a real exchange first.
        r = self.run_debate([True] * 4, [True] * 4)
        self.assertEqual((r["rounds_run"], r["converged_at"]), (2, 2))

    def test_saved_turns_carry_round_and_verdict(self):
        ld.build_agents = lambda *a, **k: (StubAgent("A", [False, True, True, True]),
                                   StubAgent("B", [None, True, True, True]))
        with contextlib.redirect_stdout(io.StringIO()):
            r = ld.run_debate("q", 4, 2, check=False)
        self.assertEqual([(t["round"], t["speaker"], t["verdict"]) for t in r["transcript"]],
                         [(1, "A", False), (1, "B", None),
                          (2, "A", True), (2, "B", True)])

    def test_failure_midway_keeps_the_rounds_already_done(self):
        class Failing(StubAgent):
            def speak(self, question, transcript):
                if self.turn >= 2:
                    raise RuntimeError("boom")
                return super().speak(question, transcript)

        ld.build_agents = lambda *a, **k: (StubAgent("A", [False] * 4), Failing("B", [False] * 4))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = ld.run_debate("q", 4, 2, check=False)
        self.assertEqual(r["error"], "boom")
        # A spoke in rounds 1-3, B only in rounds 1-2 before failing.
        self.assertEqual(len(r["transcript"]), 5)
        self.assertEqual(r["rounds_run"], 3)

    def test_failure_on_the_very_first_turn_returns_empty(self):
        class Failing(StubAgent):
            def speak(self, question, transcript):
                raise RuntimeError("boom")

        ld.build_agents = lambda *a, **k: (Failing("A", []), StubAgent("B", [False] * 4))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = ld.run_debate("q", 4, 2, check=False)
        self.assertEqual(r["transcript"], [])
        self.assertIsNone(r["summary"])
        self.assertEqual(r["error"], "boom")

    def judge_saying(self, *replies):
        class J(ld.Agent):
            def __init__(self):
                super().__init__("Judge", None, "stub")
                self.calls = 0

            def chat(inner, messages, temperature=0.7, retries=3):
                reply = replies[min(inner.calls, len(replies) - 1)]
                inner.calls += 1
                if reply is None:
                    raise RuntimeError("judge down")
                return reply

        return J()

    def test_judge_can_veto_an_early_convergence(self):
        # Agents agree from round 1; the judge rejects it once, then accepts.
        judge = self.judge_saying("still unresolved: scope\n[OPEN]", "none\n[CONVERGED]")
        ld.build_agents = lambda *a, **k: (StubAgent("A", [True] * 4), StubAgent("B", [True] * 4))
        with contextlib.redirect_stdout(io.StringIO()):
            r = ld.run_debate("q", 4, 2, check=False, judge=judge)
        self.assertEqual(r["converged_at"], 3, "round 2 was vetoed, round 3 accepted")
        self.assertEqual([x["agreed"] for x in r["judge_rulings"]], [False, True])

    def test_vetoed_objection_is_put_back_into_the_debate(self):
        judge = self.judge_saying("you never defined 'small team'\n[OPEN]", "none\n[CONVERGED]")
        ld.build_agents = lambda *a, **k: (StubAgent("A", [True] * 4), StubAgent("B", [True] * 4))
        with contextlib.redirect_stdout(io.StringIO()):
            r = ld.run_debate("q", 4, 2, check=False, judge=judge)
        judge_turns = [t for t in r["transcript"] if t["speaker"] == "Judge"]
        self.assertEqual(len(judge_turns), 1)
        self.assertIn("small team", judge_turns[0]["text"])

    def test_unavailable_judge_defers_to_the_agents(self):
        # A broken judge must not block a convergence the agents both declared.
        judge = self.judge_saying(None)
        ld.build_agents = lambda *a, **k: (StubAgent("A", [True] * 4), StubAgent("B", [True] * 4))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = ld.run_debate("q", 4, 2, check=False, judge=judge)
        self.assertEqual(r["converged_at"], 2)

    def test_an_unavailable_judge_is_recorded_as_not_having_ruled(self):
        # Otherwise a quota-exhausted judge is indistinguishable in the saved
        # record from one that approved the convergence.
        judge = self.judge_saying(None)
        ld.build_agents = lambda *a, **k: (StubAgent("A", [True] * 4), StubAgent("B", [True] * 4))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = ld.run_debate("q", 4, 2, check=False, judge=judge)
        ruling = r["judge_rulings"][0]
        self.assertFalse(ruling["ruled"])
        self.assertTrue(ruling["agreed"], "deferring to the agents, not blocking them")
        self.assertIn("judge unavailable", ruling["note"])

    def test_a_real_ruling_is_marked_as_ruled(self):
        judge = self.judge_saying("none\n[CONVERGED]")
        ld.build_agents = lambda *a, **k: (StubAgent("A", [True] * 4), StubAgent("B", [True] * 4))
        with contextlib.redirect_stdout(io.StringIO()):
            r = ld.run_debate("q", 4, 2, check=False, judge=judge)
        self.assertTrue(r["judge_rulings"][0]["ruled"])

    def test_judge_without_a_verdict_defers_to_the_agents(self):
        judge = self.judge_saying("I am not sure either way")
        ld.build_agents = lambda *a, **k: (StubAgent("A", [True] * 4), StubAgent("B", [True] * 4))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = ld.run_debate("q", 4, 2, check=False, judge=judge)
        self.assertEqual(r["converged_at"], 2)

    def test_missing_verdict_never_ends_the_debate(self):
        # Fail safe: an unparseable verdict must not be read as agreement.
        r = self.run_debate([True] * 4, [None] * 4)
        self.assertEqual((r["rounds_run"], r["converged_at"]), (4, None))


if __name__ == "__main__":
    unittest.main()
