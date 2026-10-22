"""Measure how reliably an agent reports convergence.

The debate loop trusts each agent to say whether a disagreement is settled.
That trust is only worth what the model's verdicts are worth, so this measures
it directly against two fixed transcripts whose correct verdict is known:

    open      one objection is plainly unanswered -> [OPEN] every time
    resolved  the objection was answered and accepted -> [CONVERGED] every time

Both cases are needed. Scoring only the "open" case rewards a model that
always says [OPEN], which is not accuracy, just pessimism -- the "resolved"
case is what catches that.

Needs Ollama running (and a key for --agent a or --mode judge).

    python bench_verdict.py --runs 8
    python bench_verdict.py --runs 8 --mode all --case both
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

import local_debate as ld

QUESTION = "小型新創團隊應該先寫單元測試，還是先把功能做出來？"

# Agent-B objects; Agent-A then restates its position without answering the
# objection. Nothing here is resolved, so [OPEN] is the only correct verdict.
OPEN_CASE = [
    ("Agent-A (Gemini)",
     "先把功能做出來。新創最大的風險是做出沒人要的東西，"
     "測試是在為可能被丟棄的程式碼買保險。"),
    ("Agent-B (Local)",
     "我同意速度重要，但你沒有回答：付款與計費邏輯一旦出錯就是真金白銀的損失，"
     "這部分為什麼也能不測？"),
    ("Agent-A (Gemini)",
     "總之早期速度優先，團隊成長後再補測試。"),
]

# Same opening and same objection, but this time Agent-A answers it, concedes
# a concrete rule, and Agent-B accepts that rule. [CONVERGED] is correct here.
RESOLVED_CASE = [
    ("Agent-A (Gemini)",
     "先把功能做出來。新創最大的風險是做出沒人要的東西，"
     "測試是在為可能被丟棄的程式碼買保險。"),
    ("Agent-B (Local)",
     "我同意速度重要，但你沒有回答：付款與計費邏輯一旦出錯就是真金白銀的損失，"
     "這部分為什麼也能不測？"),
    ("Agent-A (Gemini)",
     "你說得對，我的說法太寬了。修正為一條具體規則："
     "會動到錢、權限或資料刪除的程式碼一律先寫測試；其餘探索性功能先做再說。"
     "理由是前者的錯誤無法用「之後再修」收回。"),
    ("Agent-B (Local)",
     "我接受這條規則，它正好回答了我的異議，界線也夠明確可以直接執行。"
     "我沒有其他未解決的反對意見。"),
]

CASES = {"open": (OPEN_CASE, False), "resolved": (RESOLVED_CASE, True)}


def build_messages(agent: ld.Agent, transcript: list[tuple[str, str]]) -> list[dict]:
    messages = [
        {"role": "system",
         "content": ld.SYSTEM_PROMPT.format(name=agent.name, converged=ld.CONVERGED)},
        {"role": "user", "content": f"The question under debate:\n\n{QUESTION}"},
    ]
    for speaker, text in transcript:
        role = "assistant" if speaker == agent.name else "user"
        prefix = "" if role == "assistant" else f"{speaker} said:\n"
        messages.append({"role": role, "content": prefix + text})
    return messages


def sample(mode: str, under_test: ld.Agent, judge: ld.Agent,
           transcript: list[tuple[str, str]]) -> bool | None:
    if mode == "judge":
        agreed, _note, ruled = ld.judge_convergence(judge, QUESTION, transcript)
        # A judge that could not rule did not answer; do not score it as one.
        return agreed if ruled else None
    agent = ld.Agent(under_test.name, under_test.client, under_test.model)
    if mode == "marker":
        _, verdict = agent.speak(QUESTION, transcript)
        return verdict
    messages = build_messages(agent, transcript)
    return agent.ask_verdict(messages, agent.chat(messages))


def main() -> None:
    ld.force_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=8, help="samples per mode (default: 8)")
    parser.add_argument("--agent", choices=["a", "b"], default="b",
                        help="which debater to measure (default: b, the local one)")
    parser.add_argument("--mode", choices=["marker", "probe", "judge", "both", "all"],
                        default="both",
                        help="verdict from the turn itself, from a follow-up, "
                             "from the --judge model, or compare")
    parser.add_argument("--case", choices=["open", "resolved", "both"], default="both",
                        help="which fixed transcript to score (default: both)")
    args = parser.parse_args()

    load_dotenv()
    agent_a, agent_b = ld.build_agents()
    under_test = agent_a if args.agent == "a" else agent_b
    judge = ld.build_judge()
    print(f"debater: {under_test.model}   judge: {judge.model}   runs: {args.runs}\n")

    modes = {"both": ["marker", "probe"],
             "all": ["marker", "probe", "judge"]}.get(args.mode, [args.mode])
    cases = ["open", "resolved"] if args.case == "both" else [args.case]

    for case in cases:
        transcript, correct = CASES[case]
        label = "CONVERGED" if correct else "OPEN"
        print(f"case '{case}' -- correct verdict is always [{label}]")
        for mode in modes:
            verdicts = [sample(mode, under_test, judge, transcript)
                        for _ in range(args.runs)]
            # A missing verdict is not a wrong answer, but it is not a right
            # one either; count it separately rather than hiding it in either.
            wrong = sum(1 for v in verdicts if v is not None and v is not correct)
            missing = sum(1 for v in verdicts if v is None)
            shown = ["CONVERGED" if v else "OPEN" if v is False else "none"
                     for v in verdicts]
            print(f"  {mode:7} wrong {wrong}/{args.runs}, no verdict {missing}/{args.runs}")
            print(f"          {', '.join(shown)}")
        print()


if __name__ == "__main__":
    main()
