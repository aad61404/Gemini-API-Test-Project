"""Measure how reliably an agent reports convergence.

The debate loop trusts each agent to say whether a disagreement is settled.
That trust is only worth what the model's verdicts are worth, so this measures
it directly: a fixed transcript where one objection is plainly unanswered, run
N times. The correct verdict is [OPEN] every single time, so any [CONVERGED]
is a counted error and the result is a straight error rate.

Needs Ollama running (and a key only for --agent a).

    python bench_verdict.py --runs 8
    python bench_verdict.py --runs 8 --mode probe
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

import local_debate as ld

QUESTION = "小型新創團隊應該先寫單元測試，還是先把功能做出來？"

# Agent-B objects; Agent-A then restates its position without answering the
# objection. Nothing here is resolved, so [OPEN] is the only correct verdict.
TRANSCRIPT = [
    ("Agent-A (Gemini)",
     "先把功能做出來。新創最大的風險是做出沒人要的東西，"
     "測試是在為可能被丟棄的程式碼買保險。"),
    ("Agent-B (Local)",
     "我同意速度重要，但你沒有回答：付款與計費邏輯一旦出錯就是真金白銀的損失，"
     "這部分為什麼也能不測？"),
    ("Agent-A (Gemini)",
     "總之早期速度優先，團隊成長後再補測試。"),
]


def build_messages(agent: ld.Agent) -> list[dict]:
    messages = [
        {"role": "system",
         "content": ld.SYSTEM_PROMPT.format(name=agent.name, converged=ld.CONVERGED)},
        {"role": "user", "content": f"The question under debate:\n\n{QUESTION}"},
    ]
    for speaker, text in TRANSCRIPT:
        role = "assistant" if speaker == agent.name else "user"
        prefix = "" if role == "assistant" else f"{speaker} said:\n"
        messages.append({"role": role, "content": prefix + text})
    return messages


def main() -> None:
    ld.force_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=8, help="samples per mode (default: 8)")
    parser.add_argument("--agent", choices=["a", "b"], default="b",
                        help="which debater to measure (default: b, the local one)")
    parser.add_argument("--mode", choices=["marker", "probe", "both"], default="both",
                        help="verdict from the turn itself, from a follow-up, or compare")
    args = parser.parse_args()

    load_dotenv()
    agent_a, agent_b = ld.build_agents()
    under_test = agent_a if args.agent == "a" else agent_b
    print(f"model: {under_test.model}   runs: {args.runs}   "
          f"(correct verdict is always [OPEN])\n")

    modes = ["marker", "probe"] if args.mode == "both" else [args.mode]
    for mode in modes:
        verdicts = []
        for _ in range(args.runs):
            agent = ld.Agent(under_test.name, under_test.client, under_test.model)
            if mode == "marker":
                _, verdict = agent.speak(QUESTION, TRANSCRIPT)
            else:
                messages = build_messages(agent)
                verdict = agent.ask_verdict(messages, agent.chat(messages))
            verdicts.append(verdict)

        wrong = sum(1 for v in verdicts if v is True)
        missing = sum(1 for v in verdicts if v is None)
        shown = ["CONVERGED" if v else "OPEN" if v is False else "none" for v in verdicts]
        print(f"{mode:7} wrongly converged {wrong}/{args.runs}, "
              f"no verdict {missing}/{args.runs}")
        print(f"        {', '.join(shown)}")


if __name__ == "__main__":
    main()
