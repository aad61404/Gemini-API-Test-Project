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
from dataclasses import dataclass, field
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

SUMMARY_PROMPT = """Below is a debate transcript between two agents.

Write the integrated conclusion, in the language of the original question:
1. What both agents agreed on.
2. What remained disputed, and what the strongest case on each side was.
3. Your integrated answer to the original question.

Do not invent agreement that is not in the transcript."""


@dataclass
class Agent:
    """One debater: a name, a client, a model, and its convergence state."""

    name: str
    client: OpenAI
    model: str
    converged: bool = False
    _history: list[dict] = field(default_factory=list)

    def chat(self, messages: list[dict], temperature: float = 0.7, retries: int = 3) -> str:
        """Call the model, retrying only errors that a retry could actually fix."""
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    timeout=120,
                )
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
                last_error = exc
            wait = 2 ** attempt
            print(f"  ! {self.name} failed ({last_error}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
        raise RuntimeError(f"{self.name} failed after {retries} attempts: {last_error}")

    def speak(self, question: str, transcript: list[tuple[str, str]]) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(name=self.name, converged=CONVERGED)},
            {"role": "user", "content": f"The question under debate:\n\n{question}"},
        ]
        for speaker, text in transcript:
            role = "assistant" if speaker == self.name else "user"
            prefix = "" if role == "assistant" else f"{speaker} said:\n"
            messages.append({"role": role, "content": prefix + text})

        return self.chat(messages)


def split_marker(text: str) -> tuple[str, bool]:
    """Strip the trailing convergence marker; report whether it was [CONVERGED]."""
    converged = bool(re.search(rf"{re.escape(CONVERGED)}\s*$", text))
    body = re.sub(r"(\[CONVERGED\]|\[OPEN\])\s*$", "", text).strip()
    return body, converged


def build_agents() -> tuple[Agent, Agent]:
    load_dotenv()

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        sys.exit("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    agent_a = Agent(
        name="Agent-A (Gemini)",
        client=OpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    )
    agent_b = Agent(
        name="Agent-B (Local)",
        client=OpenAI(
            api_key="ollama",  # Ollama ignores this, but the SDK requires a value
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        ),
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    )
    return agent_a, agent_b


def force_utf8_output() -> None:
    """Windows consoles default to a legacy codepage (e.g. cp950), which raises
    UnicodeEncodeError on CJK debate text. Print as UTF-8 instead."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run_debate(question: str, max_rounds: int, min_rounds: int) -> dict:
    force_utf8_output()
    agent_a, agent_b = build_agents()
    transcript: list[tuple[str, str]] = []
    rounds_run = 0
    converged_at: int | None = None

    for rnd in range(1, max_rounds + 1):
        rounds_run = rnd
        print(f"\n{'=' * 60}\nRound {rnd}/{max_rounds}\n{'=' * 60}")

        for agent in (agent_a, agent_b):
            raw = agent.speak(question, transcript)
            body, agent.converged = split_marker(raw)
            transcript.append((agent.name, body))
            flag = " [CONVERGED]" if agent.converged else ""
            print(f"\n--- {agent.name}{flag} ---\n{body}")

        # Convergence must be mutual and simultaneous, and only after a real
        # exchange has happened -- otherwise one polite agent ends the debate.
        if rnd >= min_rounds and agent_a.converged and agent_b.converged:
            converged_at = rnd
            print(f"\n>>> Both agents converged in round {rnd}; ending early.")
            break

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
        print(f"  ! summary via {agent_a.name} failed ({exc}); trying {agent_b.name}", file=sys.stderr)
        try:
            summary = agent_b.chat(messages, temperature=0.3)
        except RuntimeError as exc2:
            print(f"  ! summary failed on both agents ({exc2}); transcript kept", file=sys.stderr)
            summary = None
    print(summary if summary else "(summary unavailable)")

    return {
        "question": question,
        "model_a": agent_a.model,
        "model_b": agent_b.model,
        "rounds_run": rounds_run,
        "converged_at": converged_at,
        "transcript": [{"speaker": s, "text": t} for s, t in transcript],
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Two AI agents debate a question.")
    parser.add_argument("question", help="the question to debate")
    parser.add_argument("--rounds", type=int, default=4, help="maximum rounds (default: 4)")
    parser.add_argument("--min-rounds", type=int, default=2, help="rounds before convergence may end the debate")
    parser.add_argument("--save", action="store_true", help="write the transcript to transcripts/")
    args = parser.parse_args()

    result = run_debate(args.question, args.rounds, args.min_rounds)

    if args.save:
        out_dir = Path("transcripts")
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"debate-{datetime.now():%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
