"""Rewrite the results table in README.md from results/summary.json, so the
README numbers always come from logged rollout data.

Usage (client venv):  python scripts/readme_table.py
"""

import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
START = "<!-- RESULTS_TABLE_START -->"
END = "<!-- RESULTS_TABLE_END -->"


def build_table(summary):
    lines = [
        "| prompt | group | kind | success |",
        "| --- | --- | --- | --- |",
    ]
    for entry in summary["prompts"].values():
        lines.append(
            '| "{}" | {} | {} | {}/{} |'.format(
                entry["text"],
                entry["group"].replace("_", "-"),
                entry["kind"],
                entry["num_successes"],
                entry["num_episodes"],
            )
        )
    lines.append("")
    lines.append(
        "Generated {} from `results/summary.json`.".format(summary["generated_at"][:10])
    )
    return "\n".join(lines)


def main():
    with open(str(REPO_ROOT / "results" / "summary.json")) as f:
        summary = json.load(f)
    readme_path = REPO_ROOT / "README.md"
    readme = readme_path.read_text()
    if START not in readme or END not in readme:
        raise SystemExit("README.md is missing the results table markers")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    readme_path.write_text(head + START + "\n" + build_table(summary) + "\n" + END + tail)
    print("README.md results table updated")


if __name__ == "__main__":
    main()
