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

from dotenv import load_dotenv

import local_debate as ld

# Three topics, so a result is not an artefact of one question's wording.
# Each has an `open` transcript (an objection is restated past, never
# answered) and a `resolved` one (the same objection is answered with a
# concrete rule the other side accepts). Same opening in both, so the only
# difference is whether the disagreement was actually settled.
TOPICS = {
    "tests": {
        "question": "小型新創團隊應該先寫單元測試，還是先把功能做出來？",
        "opening": [
            ("Agent-A (Gemini)",
             "先把功能做出來。新創最大的風險是做出沒人要的東西，"
             "測試是在為可能被丟棄的程式碼買保險。"),
            ("Agent-B (Local)",
             "我同意速度重要，但你沒有回答：付款與計費邏輯一旦出錯就是真金白銀的損失，"
             "這部分為什麼也能不測？"),
        ],
        "dodge": ("Agent-A (Gemini)", "總之早期速度優先，團隊成長後再補測試。"),
        "answer": ("Agent-A (Gemini)",
                   "你說得對，我的說法太寬了。修正為一條具體規則："
                   "會動到錢、權限或資料刪除的程式碼一律先寫測試；其餘探索性功能先做再說。"
                   "理由是前者的錯誤無法用「之後再修」收回。"),
        "accept": ("Agent-B (Local)",
                   "我接受這條規則，它正好回答了我的異議，界線也夠明確可以直接執行。"
                   "我沒有其他未解決的反對意見。"),
    },
    "hiring": {
        "question": "工程團隊面試該不該考白板演算法題？",
        "opening": [
            ("Agent-A (Gemini)",
             "該考。白板題能在一小時內看出候選人的問題拆解能力，"
             "而且題目一致，不同面試官之間比較得起來。"),
            ("Agent-B (Local)",
             "一致性我同意，但你沒有回答：這類題目對有照顧責任、沒時間刷題的資深工程師"
             "明顯不利，等於用準備時間篩人而不是用能力篩人。"),
        ],
        "dodge": ("Agent-A (Gemini)", "總之標準化的測驗還是目前最可行的方式。"),
        "answer": ("Agent-A (Gemini)",
                   "這點我讓步。改成：題目限定在不需要事前背誦的範圍，"
                   "並且允許候選人改用自己帶來的程式碼來替代，兩條路給同樣的評分權重。"
                   "這樣保留了可比較性，也不再獎勵刷題時間。"),
        "accept": ("Agent-B (Local)",
                   "這個做法回應了我的反對點，而且可以直接寫進面試流程。"
                   "我沒有其他未解決的異議。"),
    },
    "monorepo": {
        "question": "中型團隊該用 monorepo 還是多個 repo？",
        "opening": [
            ("Agent-A (Gemini)",
             "用 monorepo。跨專案的改動可以一次 commit 完成，"
             "不會出現版本對不起來、要追三個 PR 才看得懂一個變更的情況。"),
            ("Agent-B (Local)",
             "原子性改動確實是優點，但你沒有回答：CI 時間會隨 repo 成長而爆炸，"
             "改一行文件也要等整包跑完，這個成本誰吸收？"),
        ],
        "dodge": ("Agent-A (Gemini)", "整體來說 monorepo 的協作效益還是大於成本。"),
        "answer": ("Agent-A (Gemini)",
                   "這是真的問題，我修正主張：monorepo 的前提是先有依路徑觸發的 CI，"
                   "沒有這個就不要合。工具不到位時，多 repo 反而是對的。"),
        "accept": ("Agent-B (Local)",
                   "把工具當成前提條件，我接受 —— 這正好處理了我擔心的成本問題，"
                   "而且是可以驗證的條件。我沒有其他異議。"),
    },
}


def case(topic: str, kind: str) -> tuple[list[tuple[str, str]], bool]:
    """Return (transcript, correct_verdict) for one topic and case."""
    t = TOPICS[topic]
    if kind == "open":
        return [*t["opening"], t["dodge"]], False
    return [*t["opening"], t["answer"], t["accept"]], True


def build_messages(agent: ld.Agent, question: str,
                   transcript: list[tuple[str, str]]) -> list[dict]:
    messages = [
        {"role": "system",
         "content": ld.SYSTEM_PROMPT.format(name=agent.name, converged=ld.CONVERGED)},
        {"role": "user", "content": f"The question under debate:\n\n{question}"},
    ]
    for speaker, text in transcript:
        role = "assistant" if speaker == agent.name else "user"
        prefix = "" if role == "assistant" else f"{speaker} said:\n"
        messages.append({"role": role, "content": prefix + text})
    return messages


def sample(mode: str, under_test: ld.Agent, judge: ld.Agent,
           question: str, transcript: list[tuple[str, str]]) -> bool | None:
    if mode == "judge":
        agreed, _note, ruled = ld.judge_convergence(judge, question, transcript)
        # A judge that could not rule did not answer; do not score it as one.
        return agreed if ruled else None
    agent = ld.Agent(under_test.name, under_test.client, under_test.model)
    if mode == "marker":
        _, verdict = agent.speak(question, transcript)
        return verdict
    messages = build_messages(agent, question, transcript)
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
    parser.add_argument("--topic", choices=[*TOPICS, "all"], default="all",
                        help="which question to score (default: all three)")
    args = parser.parse_args()

    load_dotenv()
    agent_a, agent_b = ld.build_agents()
    under_test = agent_a if args.agent == "a" else agent_b
    judge = ld.build_judge()
    print(f"debater: {under_test.model}   judge: {judge.model}   runs: {args.runs}\n")

    modes = {"both": ["marker", "probe"],
             "all": ["marker", "probe", "judge"]}.get(args.mode, [args.mode])
    kinds = ["open", "resolved"] if args.case == "both" else [args.case]
    topics = list(TOPICS) if args.topic == "all" else [args.topic]

    totals: dict[str, list[int]] = {mode: [0, 0, 0] for mode in modes}
    for topic in topics:
        question = TOPICS[topic]["question"]
        for kind in kinds:
            transcript, correct = case(topic, kind)
            label = "CONVERGED" if correct else "OPEN"
            print(f"{topic}/{kind} -- correct verdict is always [{label}]")
            for mode in modes:
                verdicts = [sample(mode, under_test, judge, question, transcript)
                            for _ in range(args.runs)]
                # A missing verdict is not a wrong answer, but it is not a
                # right one either; count it apart rather than hide it in one.
                wrong = sum(1 for v in verdicts if v is not None and v is not correct)
                missing = sum(1 for v in verdicts if v is None)
                totals[mode][0] += wrong
                totals[mode][1] += missing
                totals[mode][2] += args.runs
                shown = ["CONVERGED" if v else "OPEN" if v is False else "none"
                         for v in verdicts]
                print(f"  {mode:7} wrong {wrong}/{args.runs}, "
                      f"no verdict {missing}/{args.runs}")
                print(f"          {', '.join(shown)}")
            print()

    if len(topics) * len(kinds) > 1:
        print("pooled across topics and cases:")
        for mode, (wrong, missing, total) in totals.items():
            print(f"  {mode:7} wrong {wrong}/{total}, no verdict {missing}/{total}")
        print()


if __name__ == "__main__":
    main()
