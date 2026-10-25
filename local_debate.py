"""Two agents debate one question until they converge.

Agent A runs on Gemini's OpenAI-compatible endpoint (cloud, free tier).
Agent B runs on a local model served by Ollama. Both are driven through the
same openai SDK, so if the Gemini quota runs out only Agent A is affected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

CONVERGED = "[CONVERGED]"

SYSTEM_PROMPT = """You are {name}, one of two AI agents debating a question.

Rules:
- Answer in the same language as the question.
- Be concrete. Cite reasoning, not authority. Keep each turn under 200 words.
- Read the other agent's last turn and engage with it directly: say what you
  accept, what you reject, and why. Do not restate your own position verbatim.
- Do not agree just to be agreeable, and do not disagree just to seem critical.
- End your turn with exactly one of these two markers on its own line:
  {converged}   - if the remaining disagreement is not material
  [OPEN]        - if a material disagreement remains

Never write the {converged} marker before you have genuinely addressed the
other agent's strongest objection."""

STANCE_PROMPT = """
Your assigned opening position:

{stance}

Argue this position at its strongest, and give the real reasons for it rather
than a caricature. You are not required to hold it to the end: if the other
agent answers your case, say so and concede that point. Abandoning a position
you cannot defend is the debate working, not you losing it."""

STANCE_SPLIT_PROMPT = """Give the two strongest opposing positions on this
question, in the language of the question.

Rules:
- They must genuinely conflict: someone holding one would reject the other.
- Each is one sentence, stating what to actually do and the core reason.
- No hedging, no "it depends", no restating the question.

Reply with exactly two lines and nothing else:
A: <first position>
B: <opposing position>"""

SUMMARY_PROMPT = """Below is a debate transcript between two agents.

Write the integrated conclusion, in the language of the original question:
1. What both agents agreed on.
2. What remained disputed, and what the strongest case on each side was.
3. Your integrated answer to the original question.

Do not invent agreement that is not in the transcript."""

JUDGE_PROMPT = """You are judging a debate between two agents. They have both
declared they are done. Your job is to check whether that is actually true.

Answer in the language of the question.

Be strict. They converged too early if any of this holds:
- they agree on wording but still differ on what to actually do;
- one side dropped an objection without it being answered;
- they settled on "it depends" without saying what it depends on.

Reply in exactly this shape:
<one line naming the strongest unresolved point, or "none">
[CONVERGED] - the disagreement really is resolved
[OPEN]      - something material is still unresolved"""

VERDICT_PROMPT = """Based on your immediately preceding answer, is any material
disagreement still unresolved?

Reply with exactly one token and nothing else:
[CONVERGED] - no material disagreement remains
[OPEN]      - a material disagreement remains"""


RETRY_DELAY_RE = re.compile(r"'?retryDelay'?:\s*'?(\d+)s", re.I)


def retry_after(exc: Exception) -> int:
    """Seconds the API asked us to wait, or 0 if it did not say.

    Gemini reports this inside the error payload rather than in a header, so
    it has to be read out of the message text.
    """
    match = RETRY_DELAY_RE.search(str(exc))
    return int(match.group(1)) if match else 0


@dataclass
class Agent:
    """One debater: a name, a client, a model, and its convergence state."""

    name: str
    client: OpenAI
    model: str
    converged: bool = False
    # The free tier is metered in requests per day, not tokens, so the call
    # count is the number that decides whether another debate will run.
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stance: str | None = None

    def system_prompt(self) -> str:
        prompt = SYSTEM_PROMPT.format(name=self.name, converged=CONVERGED)
        if self.stance:
            prompt += "\n" + STANCE_PROMPT.format(stance=self.stance)
        return prompt

    def usage(self) -> dict:
        return {"model": self.model, "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}

    def chat(self, messages: list[dict], temperature: float = 0.7, retries: int = 3) -> str:
        """Call the model, retrying only errors that a retry could actually fix."""
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                self.calls += 1
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    timeout=120,
                )
                if resp.usage:
                    self.prompt_tokens += resp.usage.prompt_tokens or 0
                    self.completion_tokens += resp.usage.completion_tokens or 0
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
                last_error = RuntimeError("empty response")
            except OpenAIError as exc:
                status = getattr(exc, "status_code", None)
                # 404 (bad model name) or 401/403 (bad key) will never succeed
                # on retry; fail fast instead of burning three attempts.
                if status is not None and status not in (408, 429) and status < 500:
                    raise RuntimeError(f"{self.name}: {exc}") from exc
                # A 429 is usually a per-minute rate limit, worth waiting out.
                # A spent daily quota is also a 429, but the retry window is
                # hours away, so backing off for seconds is pure waste.
                if status == 429 and retry_after(exc) > 300:
                    raise RuntimeError(
                        f"{self.name}: daily quota exhausted for {self.model}; "
                        f"retry in {retry_after(exc) // 60} min, or switch models "
                        f"(the local agent is unaffected)") from exc
                last_error = exc
            wait = 2 ** attempt
            print(f"  ! {self.name} failed ({last_error}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
        raise RuntimeError(f"{self.name} failed after {retries} attempts: {last_error}")

    def speak(self, question: str, transcript: list[tuple[str, str]]) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": f"The question under debate:\n\n{question}"},
        ]
        for speaker, text in transcript:
            role = "assistant" if speaker == self.name else "user"
            prefix = "" if role == "assistant" else f"{speaker} said:\n"
            messages.append({"role": role, "content": prefix + text})

        answer = self.chat(messages)
        body, verdict = split_marker(answer)
        if verdict is None:
            verdict = self.ask_verdict(messages, answer)
        return body, verdict

    def ask_verdict(self, messages: list[dict], answer: str) -> bool | None:
        """Ask only for the verdict when the turn arrived without a marker.

        Small local models argue fine but drop the format instruction often.
        Re-asking for a single token is far cheaper than regenerating the whole
        turn, and temperature 0 keeps it deterministic.
        """
        probe = [
            *messages,
            {"role": "assistant", "content": answer},
            {"role": "user", "content": VERDICT_PROMPT},
        ]
        try:
            reply = self.chat(probe, temperature=0)
        except RuntimeError as exc:
            print(f"  ! {self.name}: verdict probe failed ({exc})", file=sys.stderr)
            return None

        _, verdict = split_marker(reply)
        if verdict is None:
            # The probe answer is a single token, so it may lack the line
            # framing split_marker insists on. Accept it only if unambiguous.
            upper = reply.upper()
            has_c, has_o = "[CONVERGED]" in upper, "[OPEN]" in upper
            if has_c != has_o:
                verdict = has_c
        if verdict is None:
            print(f"  ! {self.name}: no verdict after probe; treating as open", file=sys.stderr)
        return verdict


# Two positions count as a verdict: a line holding nothing else, and the very
# end of the message. Models use both -- some lead with the marker on its own
# line, some end a sentence with it. Anywhere else it is prose quoting the
# marker ("last round I wrote [CONVERGED], however...") and must not be read
# as a verdict.
MARKER_RE = re.compile(r"(?m)^[ \t]*\[(CONVERGED|OPEN)\][ \t]*$")
TRAILING_MARKER_RE = re.compile(r"\[(CONVERGED|OPEN)\]\s*$")


def split_marker(text: str) -> tuple[str, bool | None]:
    """Strip the verdict marker and report it.

    Returns (body, verdict) where verdict is True for [CONVERGED], False for
    [OPEN], and None when the model gave no parseable verdict at all -- the
    caller decides what to do about that, since "no verdict" and "not
    converged" are different situations.
    """
    matches = list(MARKER_RE.finditer(text))
    trailing = TRAILING_MARKER_RE.search(text)
    if trailing and not any(m.start() == trailing.start() for m in matches):
        matches.append(trailing)

    if not matches:
        return text.strip(), None

    last = max(matches, key=lambda m: m.start())
    verdict = last.group(1) == "CONVERGED"

    body = text
    for match in sorted(matches, key=lambda m: m.start(), reverse=True):
        body = body[:match.start()] + body[match.end():]
    return body.strip(), verdict


def ollama_client() -> OpenAI:
    return OpenAI(
        api_key="ollama",  # Ollama ignores this, but the SDK requires a value
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )


def build_agents(local_only: bool = False) -> tuple[Agent, Agent]:
    """Agent A on Gemini and Agent B on Ollama, or both on Ollama.

    The point of running the two sides on different providers is that the
    local one has no quota, so --local-only keeps the project usable once the
    free Gemini tier is spent for the day. Both agents then share one model,
    which makes for a duller debate but a working one.
    """
    load_dotenv()
    local_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    if local_only:
        agent_a = Agent(name="Agent-A (Local)", client=ollama_client(),
                        model=os.getenv("OLLAMA_MODEL_A", local_model))
    else:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            sys.exit("GEMINI_API_KEY is not set. Copy .env.example to .env and fill "
                     "it in, or run with --local-only.")
        agent_a = Agent(
            name="Agent-A (Gemini)",
            client=OpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        )

    agent_b = Agent(name="Agent-B (Local)", client=ollama_client(), model=local_model)
    return agent_a, agent_b


def build_judge(local_only: bool = False) -> Agent:
    """A third opinion, deliberately not one of the debaters.

    Reusing a debater as its own judge invites it to ratify its own position,
    so the judge runs on a different model. It shares Agent A's endpoint only
    because that is the one with more than one model available.
    """
    load_dotenv()
    if local_only:
        # No second local model to fall back on, so the judge shares the
        # debaters' model here. Weaker separation, but still a fresh context
        # with no stake in either position.
        return Agent(name="Judge", client=ollama_client(),
                     model=os.getenv("JUDGE_MODEL_LOCAL",
                                     os.getenv("OLLAMA_MODEL", "qwen2.5:7b")))
    return Agent(
        name="Judge",
        client=OpenAI(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        model=os.getenv("JUDGE_MODEL", "gemini-3.5-flash"),
    )


def judge_convergence(judge: Agent, question: str,
                      transcript: list[tuple[str, str]]) -> tuple[bool, str, bool]:
    """Return (agreed, note, ruled).

    When the judge cannot rule -- it errored, hit a quota, or answered without
    a verdict -- the debate defers to the agents rather than stalling. That is
    a degradation, not a ruling, so `ruled` says which happened; a silent
    "agreed" would look exactly like the judge approving the convergence.
    """
    body = "\n\n".join(f"{speaker}:\n{text}" for speaker, text in transcript)
    try:
        reply = judge.chat(
            [{"role": "system", "content": JUDGE_PROMPT},
             {"role": "user", "content": f"Question: {question}\n\n{body}"}],
            temperature=0,
        )
    except RuntimeError as exc:
        print(f"  ! judge unavailable ({exc}); accepting the agents' own verdict",
              file=sys.stderr, flush=True)
        return True, f"judge unavailable: {exc}", False

    note, verdict = split_marker(reply)
    if verdict is None:
        print("  ! judge gave no verdict; accepting the agents' own verdict",
              file=sys.stderr, flush=True)
        return True, "judge gave no verdict", False
    return verdict, note.strip(), True


STANCE_LINE_RE = re.compile(r"(?mi)^\s*([AB])\s*[:：]\s*(.+?)\s*$")


def split_stances(text: str) -> tuple[str, str] | None:
    """Pull 'A: ...' and 'B: ...' out of the model's reply, or None."""
    found = {label.upper(): body for label, body in STANCE_LINE_RE.findall(text)}
    if "A" in found and "B" in found and found["A"] != found["B"]:
        return found["A"], found["B"]
    return None


def assign_stances(author: Agent, question: str) -> tuple[str, str] | None:
    """Ask for two genuinely opposing opening positions.

    Left to themselves the agents tend to open near the same position and
    spend the debate agreeing, which is the least useful thing two agents can
    do. Starting them apart costs one call and gives the debate something to
    actually resolve.
    """
    try:
        reply = author.chat(
            [{"role": "system", "content": STANCE_SPLIT_PROMPT},
             {"role": "user", "content": question}],
            temperature=0.4,
        )
    except RuntimeError as exc:
        print(f"  ! could not generate stances ({exc}); debating unassigned",
              file=sys.stderr, flush=True)
        return None

    stances = split_stances(reply)
    if stances is None:
        print("  ! stance reply was not two labelled lines; debating unassigned",
              file=sys.stderr, flush=True)
    return stances


def preflight(agents: tuple[Agent, ...]) -> None:
    """Check every agent answers before starting, so a misconfigured model
    fails in two seconds instead of halfway through a paid debate."""
    broken = []
    for agent in agents:
        try:
            # Two attempts, not one: a sleeping Ollama refuses the first
            # connection and is up by the time the retry lands.
            agent.chat([{"role": "user", "content": "Reply with exactly: OK"}],
                       temperature=0, retries=2)
            print(f"  ok   {agent.name}: {agent.model}", flush=True)
        except RuntimeError as exc:
            print(f"  FAIL {agent.name}: {agent.model}\n       {exc}",
                  file=sys.stderr, flush=True)
            broken.append(agent)

    if not broken:
        return

    # A missing local model is the common case and the fixable one, so say
    # what is actually installed instead of only what is missing.
    if any("Local" in a.name for a in broken):
        try:
            import json as _json
            import urllib.request
            base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            tags = base.rsplit("/v1", 1)[0] + "/api/tags"
            with urllib.request.urlopen(tags, timeout=5) as resp:
                names = [m["name"] for m in _json.load(resp).get("models", [])]
            print(f"\nOllama has: {', '.join(names) if names else '(no models)'}"
                  f"\nSet OLLAMA_MODEL in .env to one of them, or run: ollama pull <model>",
                  file=sys.stderr)
        except Exception:
            print("\nIs Ollama running? Start it, then: ollama list", file=sys.stderr)

    sys.exit(1)


def force_utf8_output() -> None:
    """Windows consoles default to a legacy codepage (e.g. cp950), which raises
    UnicodeEncodeError on CJK debate text. Print as UTF-8 instead."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run_debate(question: str, max_rounds: int, min_rounds: int, check: bool = True,
               judge: Agent | None = None, local_only: bool = False,
               stances: tuple[str, str] | bool | None = None) -> dict:
    agent_a, agent_b = build_agents(local_only)
    if check:
        print("Checking both agents are reachable...")
        preflight((agent_a, agent_b))

    if stances is True:
        stances = assign_stances(agent_a, question)
    if stances:
        agent_a.stance, agent_b.stance = stances
        print(f"\nOpening positions\n  A: {agent_a.stance}\n  B: {agent_b.stance}")

    # `transcript` feeds the prompts; `turns` is the record that gets saved,
    # and carries the per-turn verdict so convergence behaviour can be
    # analysed after the fact.
    transcript: list[tuple[str, str]] = []
    turns: list[dict] = []
    rulings: list[dict] = []
    rounds_run = 0
    converged_at: int | None = None
    error: str | None = None

    for rnd in range(1, max_rounds + 1):
        rounds_run = rnd
        print(f"\n{'=' * 60}\nRound {rnd}/{max_rounds}\n{'=' * 60}")

        for agent in (agent_a, agent_b):
            try:
                body, verdict = agent.speak(question, transcript)
            except RuntimeError as exc:
                # Keep what the debate produced so far rather than discarding
                # several rounds of work because one call failed.
                error = str(exc)
                print(f"\n!!! {agent.name} failed; stopping with {len(transcript)} "
                      f"turn(s) kept\n    {exc}", file=sys.stderr, flush=True)
                break
            # No verdict means no verdict -- never let that end the debate.
            agent.converged = verdict is True
            transcript.append((agent.name, body))
            turns.append({"round": rnd, "speaker": agent.name,
                          "verdict": verdict, "text": body})
            flag = {True: " [CONVERGED]", False: " [OPEN]", None: " [NO VERDICT]"}[verdict]
            print(f"\n--- {agent.name}{flag} ---\n{body}")

        if error:
            break

        # Convergence must be mutual and simultaneous, and only after a real
        # exchange has happened -- otherwise one polite agent ends the debate.
        if rnd >= min_rounds and agent_a.converged and agent_b.converged:
            if judge is None:
                converged_at = rnd
                print(f"\n>>> Both agents converged in round {rnd}; ending early.")
                break

            agreed, note, ruled = judge_convergence(judge, question, transcript)
            rulings.append({"round": rnd, "agreed": agreed, "note": note,
                            "ruled": ruled})
            if agreed:
                converged_at = rnd
                why = "and the judge agrees" if ruled else f"but {note}"
                print(f"\n>>> Both agents converged in round {rnd} {why}; ending early.")
                break

            # Put the objection back into the debate rather than silently
            # continuing, so the next round has something to answer.
            print(f"\n>>> Judge overruled the convergence: {note}")
            objection = (f"You both declared you were done, but this is still "
                         f"unresolved:\n{note}\nAddress it directly.")
            transcript.append(("Judge", objection))
            turns.append({"round": rnd, "speaker": "Judge",
                          "verdict": False, "text": objection})
            agent_a.converged = agent_b.converged = False

    if not transcript:
        return {
            "question": question,
            "model_a": agent_a.model,
            "model_b": agent_b.model,
            "stance_a": agent_a.stance,
            "stance_b": agent_b.stance,
            "rounds_run": rounds_run,
            "converged_at": None,
            "transcript": [],
            "judge_rulings": rulings,
            "summary": None,
            "usage": {"agent_a": agent_a.usage(), "agent_b": agent_b.usage()},
            "error": error,
        }

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    body = "\n\n".join(f"{speaker}:\n{text}" for speaker, text in transcript)
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\n{body}"},
    ]
    try:
        summary = agent_a.chat(messages, temperature=0.3)
    except RuntimeError as exc:
        # A debate is expensive; never throw the transcript away because the
        # final summarising call happened to fail. Fall back to Agent B.
        print(f"  ! summary via {agent_a.name} failed ({exc}); trying {agent_b.name}",
              file=sys.stderr)
        try:
            summary = agent_b.chat(messages, temperature=0.3)
        except RuntimeError as exc2:
            print(f"  ! summary failed on both agents ({exc2}); transcript kept", file=sys.stderr)
            summary = None
    print(summary if summary else "(summary unavailable)")

    usage = {"agent_a": agent_a.usage(), "agent_b": agent_b.usage()}
    if judge is not None:
        usage["judge"] = judge.usage()
    tally = "  ".join(f"{who}={u['calls']}" for who, u in usage.items())
    print(f"\nAPI calls this run: {tally}")

    return {
        "question": question,
        "model_a": agent_a.model,
        "model_b": agent_b.model,
        "stance_a": agent_a.stance,
        "stance_b": agent_b.stance,
        "rounds_run": rounds_run,
        "converged_at": converged_at,
        "transcript": turns,
        "judge_rulings": rulings,
        "summary": summary,
        "usage": usage,
        "error": error,
    }


def main() -> None:
    # Do this first, before anything can print: on Windows even a traceback
    # containing CJK text will die on the console's legacy codepage.
    force_utf8_output()

    parser = argparse.ArgumentParser(description="Two AI agents debate a question.")
    parser.add_argument("question", nargs="?", help="the question to debate")
    parser.add_argument("--rounds", type=int, default=4, help="maximum rounds (default: 4)")
    parser.add_argument("--min-rounds", type=int, default=2,
                        help="rounds before convergence may end the debate")
    parser.add_argument("--save", action="store_true", help="write the transcript to transcripts/")
    parser.add_argument("--stances", action="store_true",
                        help="start the agents on opposing positions, generated "
                             "from the question")
    parser.add_argument("--stance-a", help="give Agent A this opening position")
    parser.add_argument("--stance-b", help="give Agent B this opening position")
    parser.add_argument("--local-only", action="store_true",
                        help="run both agents on Ollama; no API key, no quota")
    parser.add_argument("--judge", action="store_true",
                        help="have a third model verify convergence before ending early")
    parser.add_argument("--check", action="store_true",
                        help="only check both agents are reachable, then exit")
    parser.add_argument("--no-check", dest="check_first", action="store_false",
                        help="skip the pre-debate reachability check")
    args = parser.parse_args()

    if args.check:
        force_utf8_output()
        preflight(build_agents(args.local_only))
        print("Both agents reachable.")
        return

    if not args.question:
        parser.error("a question is required (or use --check on its own)")

    if bool(args.stance_a) != bool(args.stance_b):
        parser.error("--stance-a and --stance-b must be given together")
    stances = (args.stance_a, args.stance_b) if args.stance_a else args.stances

    result = run_debate(args.question, args.rounds, args.min_rounds,
                        check=args.check_first,
                        judge=build_judge(args.local_only) if args.judge else None,
                        local_only=args.local_only,
                        stances=stances)

    # Save on request, and always save a partial debate -- the user did not
    # ask for a half-finished run, so they cannot have opted into keeping it.
    if (args.save or result["error"]) and result["transcript"]:
        out_dir = Path("transcripts")
        out_dir.mkdir(exist_ok=True)
        suffix = "-partial" if result["error"] else ""
        path = out_dir / f"debate-{datetime.now():%Y%m%d-%H%M%S}{suffix}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved to {path}")

    if result["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
