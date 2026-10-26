"""Tests that need no API key: marker parsing, the verdict probe, and the
convergence rule.

Run with:  python -m unittest test_debate -v
"""

import contextlib
import io
import os
import re
import sys
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

    def test_marker_ending_a_sentence_counts(self):
        # gemma3 does this: no line of its own, but unambiguously the verdict.
        self.assertEqual(ld.split_marker("...and so it stands. [OPEN]"),
                         ("...and so it stands.", False))

    def test_trailing_marker_beats_an_earlier_inline_mention(self):
        body, verdict = ld.split_marker(
            "Earlier I said [CONVERGED], but no. [OPEN]")
        self.assertFalse(verdict)
        self.assertIn("[CONVERGED]", body, "the quoted one is prose, keep it")

    def test_standalone_and_trailing_are_not_double_counted(self):
        # A marker on its own final line matches both patterns.
        self.assertEqual(ld.split_marker("argued it\n\n[CONVERGED]"),
                         ("argued it", True))


class TestStreaming(unittest.TestCase):
    class Chunk:
        """Minimal stand-in for an OpenAI streaming chunk."""

        def __init__(self, text=None, usage=None):
            self.usage = usage
            if text is None:
                self.choices = []
            else:
                delta = type("D", (), {"content": text})()
                self.choices = [type("C", (), {"delta": delta})()]

    def streaming_agent(self, pieces, usage=None):
        chunks = [self.Chunk(p) for p in pieces]
        if usage is not None:
            chunks.append(self.Chunk(usage=usage))

        class Client:
            class chat:  # mirrors the SDK's shape
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        assert kwargs.get("stream") is True
                        return iter(chunks)

        return ld.Agent("A", Client(), "m")

    def test_streamed_pieces_are_joined(self):
        agent = self.streaming_agent(["Hel", "lo ", "world\n[OPEN]"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            answer = agent.chat([], stream=True)
        self.assertEqual(answer, "Hello world\n[OPEN]")
        self.assertIn("Hello world", out.getvalue(), "it printed as it arrived")

    def test_usage_from_the_final_chunk_is_counted(self):
        usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 4})()
        agent = self.streaming_agent(["hi"], usage=usage)
        with contextlib.redirect_stdout(io.StringIO()):
            agent.chat([], stream=True)
        self.assertEqual((agent.prompt_tokens, agent.completion_tokens), (11, 4))

    def test_a_stream_with_no_usage_chunk_still_works(self):
        agent = self.streaming_agent(["hi"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agent.chat([], stream=True), "hi")
        self.assertEqual(agent.prompt_tokens, 0)


class TestWindowed(unittest.TestCase):
    TURNS = [(f"S{i}", f"turn {i}") for i in range(8)]

    def test_zero_window_changes_nothing(self):
        self.assertEqual(ld.windowed(self.TURNS, 0), self.TURNS)

    def test_short_transcript_is_untouched(self):
        self.assertEqual(ld.windowed(self.TURNS[:4], 4), self.TURNS[:4])

    def test_keeps_the_opening_and_the_last_n(self):
        out = ld.windowed(self.TURNS, 3)
        self.assertEqual(out[:2], self.TURNS[:2], "both opening positions survive")
        self.assertEqual(out[-3:], self.TURNS[-3:], "the live argument survives")
        self.assertEqual(out[2], ("Narrator", ld.SKIPPED_NOTE))
        self.assertEqual(len(out), 6)

    def test_the_gap_is_announced_not_silent(self):
        # An agent that cannot see a turn should at least know it existed.
        out = ld.windowed(self.TURNS, 2)
        self.assertTrue(any(speaker == "Narrator" for speaker, _ in out))

    def test_window_is_applied_to_the_prompt(self):
        seen = {}

        class Recording(ld.Agent):
            def chat(self, messages, temperature=0.7, retries=3, **kwargs):
                seen["messages"] = messages
                return "said it\n[OPEN]"

        agent = Recording("A", None, "m")
        agent.speak("q", self.TURNS, window=2)
        rendered = " ".join(m["content"] for m in seen["messages"])
        self.assertIn(ld.SKIPPED_NOTE, rendered)
        self.assertNotIn("turn 4", rendered)
        self.assertIn("turn 7", rendered)


class TestPromptShapes(unittest.TestCase):
    """The prompts show models what a reply looks like, so a prompt that
    writes a marker with text after it teaches them to do the same -- and
    split_marker then reads no verdict at all."""

    # A line that opens with a marker reads as a template for the reply, so
    # models copy it. Prose that mentions a marker mid-sentence does not.
    TEMPLATE_LINE = re.compile(r"^\s*\[(CONVERGED|OPEN)\]\s*\S")

    def assert_no_template_lines(self, prompt: str, name: str):
        for line in prompt.splitlines():
            with self.subTest(prompt=name, line=line):
                self.assertIsNone(
                    self.TEMPLATE_LINE.match(line),
                    f"{name} shows a marker followed by text, which models copy "
                    f"verbatim and split_marker then cannot read: {line!r}")

    def test_judge_prompt_shows_no_copyable_template_line(self):
        # Regression: it used to show "[CONVERGED] - the disagreement really is
        # resolved", which the judge copied verbatim, so every approval was
        # read as no verdict and silently deferred to the agents.
        self.assert_no_template_lines(ld.JUDGE_PROMPT, "JUDGE_PROMPT")

    def test_verdict_prompt_shows_no_copyable_template_line(self):
        self.assert_no_template_lines(ld.VERDICT_PROMPT, "VERDICT_PROMPT")

    def test_system_prompt_shows_no_copyable_template_line(self):
        self.assert_no_template_lines(
            ld.SYSTEM_PROMPT.format(name="A", converged=ld.CONVERGED), "SYSTEM_PROMPT")

    def test_an_echoed_note_is_not_quoted_at_the_debaters(self):
        # The judge once answered "Line 1: [CONVERGED]\nLine 2:", which would
        # have been injected into the debate as the objection to address.
        for echoed in ("Line 1: [OPEN]\nLine 2:", "[OPEN]", "none", "", "Line 2:"):
            with self.subTest(note=echoed):
                self.assertEqual(ld.as_objection(echoed), ld.GENERIC_OBJECTION)

    def test_a_real_note_is_passed_through_unchanged(self):
        note = "Agent-A 沒有回答 CI 時間成長的成本由誰吸收"
        self.assertEqual(ld.as_objection(note), note)

    def test_the_echoed_template_shape_is_still_treated_as_no_verdict(self):
        # The parser must not start guessing; the prompt is what got fixed.
        _, verdict = ld.split_marker("none\n[CONVERGED] - the disagreement is resolved")
        self.assertIsNone(verdict)


class TestStances(unittest.TestCase):
    def test_parses_two_labelled_lines(self):
        self.assertEqual(ld.split_stances("A: ship fast\nB: test first"),
                         ("ship fast", "test first"))

    def test_tolerates_a_full_width_colon_and_stray_prose(self):
        reply = "好的：\nA：先做功能\nB：先寫測試\n"
        self.assertEqual(ld.split_stances(reply), ("先做功能", "先寫測試"))

    def test_rejects_a_reply_missing_a_side(self):
        self.assertIsNone(ld.split_stances("A: only one position here"))

    def test_rejects_echoed_placeholders(self):
        # The prompt used to show "A: <first position>", and a stance goes
        # straight into a system prompt -- an agent would argue the placeholder.
        self.assertIsNone(ld.split_stances("A: <first position>\nB: <opposing position>"))
        self.assertIsNone(ld.split_stances("A: [your position]\nB: [the other one]"))

    def test_a_short_chinese_stance_is_still_a_stance(self):
        # Four characters is a complete position; any English-tuned length
        # threshold would throw it away.
        self.assertEqual(ld.split_stances("A: 先做功能\nB: 先寫測試"),
                         ("先做功能", "先寫測試"))

    def test_rejects_two_identical_stances(self):
        # Not opposing positions, so not worth assigning.
        self.assertIsNone(ld.split_stances("A: same thing\nB: same thing"))

    def test_stance_is_added_to_the_system_prompt(self):
        agent = ld.Agent("A", None, "m")
        self.assertNotIn("assigned opening position", agent.system_prompt())
        agent.stance = "ship fast"
        prompt = agent.system_prompt()
        self.assertIn("assigned opening position", prompt)
        self.assertIn("ship fast", prompt)

    def test_explicit_stances_are_assigned_and_recorded(self):
        ld.build_agents = lambda *a, **k: (StubAgent("A", [False] * 2),
                                           StubAgent("B", [False] * 2))
        with contextlib.redirect_stdout(io.StringIO()):
            r = ld.run_debate("q", 1, 1, check=False, stances=("for", "against"))
        self.assertEqual((r["stance_a"], r["stance_b"]), ("for", "against"))

    def test_a_failed_stance_generation_does_not_stop_the_debate(self):
        class NoStances(StubAgent):
            def chat(self, messages, temperature=0.7, retries=3, **kwargs):
                if any("opposing positions" in m["content"] for m in messages):
                    raise RuntimeError("down")
                return super().chat(messages, temperature, retries, **kwargs)

        ld.build_agents = lambda *a, **k: (NoStances("A", [False] * 2),
                                           StubAgent("B", [False] * 2))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = ld.run_debate("q", 1, 1, check=False, stances=True)
        self.assertIsNone(r["stance_a"])
        self.assertEqual(len(r["transcript"]), 2, "the debate still ran")


class TestCli(unittest.TestCase):
    """main() carries all the option plumbing and had no coverage at all."""

    def setUp(self):
        self.seen = {}
        self._real_run = ld.run_debate
        self._real_judge = ld.build_judge
        self._real_argv = sys.argv

        def fake_run(question, max_rounds, min_rounds, **kwargs):
            self.seen = {"question": question, "rounds": max_rounds,
                         "min_rounds": min_rounds, **kwargs}
            return {"transcript": [{"round": 1, "speaker": "A", "text": "t"}],
                    "error": None, "question": question}

        ld.run_debate = fake_run
        ld.build_judge = lambda *a, **k: "JUDGE"

    def tearDown(self):
        ld.run_debate = self._real_run
        ld.build_judge = self._real_judge
        sys.argv = self._real_argv

    def run_cli(self, *argv):
        sys.argv = ["local_debate.py", *argv]
        with contextlib.redirect_stdout(io.StringIO()):
            ld.main()
        return self.seen

    def test_defaults(self):
        seen = self.run_cli("why?")
        self.assertEqual((seen["question"], seen["rounds"], seen["min_rounds"]),
                         ("why?", 4, 2))
        self.assertIsNone(seen["judge"])
        self.assertFalse(seen["local_only"])
        self.assertEqual(seen["window"], 0)

    def test_every_flag_reaches_run_debate(self):
        seen = self.run_cli("why?", "--rounds", "7", "--min-rounds", "3",
                            "--window", "2", "--local-only", "--judge",
                            "--stream", "--no-check")
        self.assertEqual((seen["rounds"], seen["min_rounds"], seen["window"]), (7, 3, 2))
        self.assertEqual(seen["judge"], "JUDGE")
        self.assertTrue(seen["local_only"])
        self.assertTrue(seen["stream"])
        self.assertFalse(seen["check"])

    def test_explicit_stances_win_over_the_flag(self):
        seen = self.run_cli("why?", "--stances", "--stance-a", "x", "--stance-b", "y")
        self.assertEqual(seen["stances"], ("x", "y"))

    def test_bare_stances_flag_asks_for_generated_ones(self):
        self.assertIs(self.run_cli("why?", "--stances")["stances"], True)

    def test_half_a_stance_pair_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_cli("why?", "--stance-a", "only one")

    def test_a_missing_question_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_cli()

    def test_a_failed_debate_exits_nonzero(self):
        ld.run_debate = lambda *a, **k: {"transcript": [], "error": "boom"}
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stdout(io.StringIO()):
            self.run_cli("why?")
        self.assertEqual(cm.exception.code, 1)


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

    def test_render_shows_stances_and_call_counts(self):
        data = dict(self.DATA,
                    stance_a="ship fast", stance_b="test first",
                    usage={"agent_a": {"calls": 3}, "agent_b": {"calls": 2}})
        out = show_debate.render(data)
        self.assertIn("ship fast", out)
        self.assertIn("test first", out)
        self.assertIn("agent_a=3", out)

    def test_render_handles_a_transcript_saved_before_those_fields_existed(self):
        old = {k: v for k, v in self.DATA.items() if k != "judge_rulings"}
        out = show_debate.render(old)
        self.assertIn("## Round 1", out)
        self.assertNotIn("Opening positions", out)

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

    def chat(self, messages, temperature=0.7, retries=3, **kwargs):
        reply = self.replies[self.calls]
        self.calls += 1
        return reply


class TestVerdictProbe(unittest.TestCase):
    def test_no_probe_when_marker_present(self):
        agent = ProbeAgent(["argument\n[OPEN]"])
        body, verdict = agent.speak("q", [("Other", "an opening argument")])
        self.assertEqual((body, verdict), ("argument", False))
        self.assertEqual(agent.calls, 1, "should not spend a second call")

    def test_the_opening_statement_is_never_a_convergence(self):
        # Nothing has been said yet to converge with, whatever the model says.
        agent = ProbeAgent(["my position\n[CONVERGED]"])
        body, verdict = agent.speak("q", [])
        self.assertEqual((body, verdict), ("my position", False))

    def test_the_opening_statement_does_not_pay_for_a_probe(self):
        agent = ProbeAgent(["my position with no marker"])
        _, verdict = agent.speak("q", [])
        self.assertFalse(verdict)
        self.assertEqual(agent.calls, 1, "no verdict probe for the opener")

    def test_the_second_speaker_in_round_one_is_still_asked(self):
        # Agent B has heard A, so its verdict means something.
        agent = ProbeAgent(["replying\n[CONVERGED]"])
        _, verdict = agent.speak("q", [("Other", "an opening argument")])
        self.assertTrue(verdict)

    def test_probe_recovers_a_missing_marker(self):
        agent = ProbeAgent(["argument with no marker", "[CONVERGED]"])
        body, verdict = agent.speak("q", [("Other", "an opening argument")])
        self.assertEqual((body, verdict), ("argument with no marker", True))
        self.assertEqual(agent.calls, 2)

    def test_probe_accepts_an_unframed_token(self):
        agent = ProbeAgent(["argument", "My verdict is [OPEN]."])
        _, verdict = agent.speak("q", [("Other", "an opening argument")])
        self.assertFalse(verdict)

    def test_ambiguous_probe_answer_yields_no_verdict(self):
        # Both markers quoted mid-sentence, neither in a verdict position.
        agent = ProbeAgent(["argument", "maybe [CONVERGED] or maybe [OPEN], hard to say"])
        with contextlib.redirect_stderr(io.StringIO()):
            _, verdict = agent.speak("q", [("Other", "an opening argument")])
        self.assertIsNone(verdict)


class TestPreflight(unittest.TestCase):
    class Reachable(ld.Agent):
        def __init__(self, name="Agent-A (Gemini)"):
            super().__init__(name, None, "stub")

        def chat(self, messages, temperature=0.7, retries=3, **kwargs):
            return "OK"

    class Unreachable(ld.Agent):
        def __init__(self, name="Agent-A (Gemini)"):
            super().__init__(name, None, "missing-model")

        def chat(self, messages, temperature=0.7, retries=3, **kwargs):
            raise RuntimeError("404 model not found")

    def test_all_reachable_returns(self):
        with contextlib.redirect_stdout(io.StringIO()):
            ld.preflight((self.Reachable(), self.Reachable("Agent-B (Local)")))

    def test_unreachable_agent_exits_nonzero(self):
        with (contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO()),
              self.assertRaises(SystemExit) as cm):
            ld.preflight((self.Reachable(), self.Unreachable("Agent-B (Local)")))
        self.assertEqual(cm.exception.code, 1)

    def test_failure_names_the_broken_agent_and_model(self):
        err = io.StringIO()
        with (contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(err),
              self.assertRaises(SystemExit)):
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

    def speak(self, question, transcript, *args, **kwargs):
        verdict = self.script[self.turn]
        self.turn += 1
        return f"turn {self.turn} of {self.name}", verdict

    def chat(self, messages, temperature=0.7, retries=3, **kwargs):
        self.calls += 1          # the summary goes through here
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

    def test_usage_counts_every_call_including_the_judge(self):
        class Counting(StubAgent):
            def speak(self, question, transcript, *args, **kwargs):
                self.calls += 1
                return super().speak(question, transcript, *args, **kwargs)

        judge = self.judge_saying("none\n[CONVERGED]")
        ld.build_agents = lambda *a, **k: (Counting("A", [True] * 4),
                                           Counting("B", [True] * 4))
        with contextlib.redirect_stdout(io.StringIO()):
            r = ld.run_debate("q", 4, 2, check=False, judge=judge)
        # Two rounds each, plus the summary, which StubAgent.chat serves.
        self.assertEqual(r["usage"]["agent_a"]["calls"], 3)
        self.assertEqual(r["usage"]["agent_b"]["calls"], 2)
        self.assertIn("judge", r["usage"])

    def test_usage_is_present_without_a_judge(self):
        ld.build_agents = lambda *a, **k: (StubAgent("A", [False] * 4),
                                           StubAgent("B", [False] * 4))
        with contextlib.redirect_stdout(io.StringIO()):
            r = ld.run_debate("q", 1, 1, check=False)
        self.assertNotIn("judge", r["usage"])
        self.assertEqual(set(r["usage"]), {"agent_a", "agent_b"})

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
            def speak(self, question, transcript, *args, **kwargs):
                if self.turn >= 2:
                    raise RuntimeError("boom")
                return super().speak(question, transcript, *args, **kwargs)

        ld.build_agents = lambda *a, **k: (StubAgent("A", [False] * 4), Failing("B", [False] * 4))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = ld.run_debate("q", 4, 2, check=False)
        self.assertEqual(r["error"], "boom")
        # A spoke in rounds 1-3, B only in rounds 1-2 before failing.
        self.assertEqual(len(r["transcript"]), 5)
        self.assertEqual(r["rounds_run"], 3)

    def test_failure_on_the_very_first_turn_returns_empty(self):
        class Failing(StubAgent):
            def speak(self, question, transcript, *args, **kwargs):
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
