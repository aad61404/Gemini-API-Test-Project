"""Render a saved debate as readable Markdown.

The JSON in transcripts/ is the record; this is how you actually read it.

    python show_debate.py                      # the most recent debate
    python show_debate.py --stats              # verdicts only, no prose
    python show_debate.py transcripts/x.json -o debate.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

VERDICT_LABEL = {True: "CONVERGED", False: "OPEN", None: "no verdict"}


def latest_transcript(directory: Path) -> Path:
    files = sorted(directory.glob("*.json"))
    if not files:
        sys.exit(f"No transcripts in {directory}/ -- run a debate with --save first.")
    return max(files, key=lambda p: p.stat().st_mtime)


def render(data: dict) -> str:
    out = [f"# {data['question']}", ""]

    ended = (f"converged in round {data['converged_at']}" if data.get("converged_at")
             else f"ran the full {data['rounds_run']} round(s) without converging")
    out += [
        f"- **Agent A**: `{data['model_a']}`",
        f"- **Agent B**: `{data['model_b']}`",
        f"- **Outcome**: {ended}",
    ]
    if data.get("error"):
        out.append(f"- **Stopped early**: {data['error']}")
    if data.get("usage"):
        calls = "  ".join(f"{who}={u['calls']}" for who, u in data["usage"].items())
        out.append(f"- **API calls**: {calls}")
    out.append("")

    if data.get("stance_a") or data.get("stance_b"):
        out += ["**Opening positions**", "",
                f"- A: {data.get('stance_a') or '(unassigned)'}",
                f"- B: {data.get('stance_b') or '(unassigned)'}", ""]

    for ruling in data.get("judge_rulings") or []:
        if not ruling.get("ruled", True):
            verdict = "could not rule"
        else:
            verdict = "agreed" if ruling["agreed"] else "overruled"
        out.append(f"> **Judge, round {ruling['round']}** — {verdict}: {ruling['note']}")
    if data.get("judge_rulings"):
        out.append("")

    current_round = None
    for turn in data["transcript"]:
        if turn.get("round") != current_round:
            current_round = turn.get("round")
            out += [f"## Round {current_round}", ""]
        label = VERDICT_LABEL[turn.get("verdict")]
        out += [f"### {turn['speaker']} — _{label}_", "", turn["text"], ""]

    if data.get("summary"):
        out += ["## Summary", "", data["summary"], ""]
    return "\n".join(out)


def render_stats(data: dict) -> str:
    """Just the verdicts -- how each side behaved, without reading the prose."""
    out = [f"{data['question']}",
           f"  A: {data['model_a']}   B: {data['model_b']}",
           f"  rounds: {data['rounds_run']}   converged_at: {data['converged_at']}", ""]

    per_speaker: dict[str, Counter] = {}
    for turn in data["transcript"]:
        per_speaker.setdefault(turn["speaker"], Counter())[VERDICT_LABEL[turn.get("verdict")]] += 1

    for speaker, counts in per_speaker.items():
        tally = ", ".join(f"{label} {n}" for label, n in counts.most_common())
        out.append(f"  {speaker:20} {tally}")

    sequence = [f"r{t['round']}:{t['speaker'].split()[0]}="
                f"{VERDICT_LABEL[t.get('verdict')][:4]}" for t in data["transcript"]]
    out += ["", "  " + "  ".join(sequence)]

    # Each turn resends the whole debate so far, so the prompt grows every
    # round. Worth seeing: it is the cost that scales with --rounds.
    sized = [t for t in data["transcript"] if t.get("prompt_tokens")]
    if sized:
        out += ["", "  prompt tokens per turn:"]
        for turn in sized:
            bar = "#" * max(1, round(turn["prompt_tokens"] / 100))
            out.append(f"    r{turn['round']} {turn['speaker']:17} "
                       f"{turn['prompt_tokens']:6} {bar}")
    return "\n".join(out)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", type=Path,
                        help="transcript JSON (default: the most recent one)")
    parser.add_argument("-o", "--output", type=Path, help="write to this file instead of stdout")
    parser.add_argument("--stats", action="store_true",
                        help="show only the verdict pattern, not the arguments")
    args = parser.parse_args()

    path = args.path or latest_transcript(Path("transcripts"))
    data = json.loads(path.read_text(encoding="utf-8"))
    text = render_stats(data) if args.stats else render(data)

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
