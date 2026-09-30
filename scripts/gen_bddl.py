"""Generate BDDL task files for the novel prompts.

Each generated file copies the libero_goal scene definition verbatim from the
anchor task's BDDL file (shipped with the LIBERO package) and changes only:

  - (:language ...)        -> the novel instruction text
  - (:obj_of_interest ...) -> the objects named in the new goal
  - (:goal ...)            -> the new goal predicate

Success is therefore still judged by LIBERO's own BDDL goal evaluator, in the
exact same scene as the in-distribution tasks. openpi is not modified.

Run inside the client venv:  python scripts/gen_bddl.py
"""

import pathlib
import re

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "bddl"

# Known fixtures in the libero_goal scene (goal terms resolve to these when a
# region suffix such as flat_stove_1_cook_region is stripped).
FIXTURES = ["wooden_cabinet_1", "flat_stove_1", "wine_rack_1", "main_table"]
OBJECTS = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1"]


def _find_block(text, name):
    """Return (start, end) spanning the s-expression block `(:name ...)`."""
    start = text.index("(:" + name)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return start, i + 1
    raise ValueError("unbalanced parens in block " + name)


def _replace_block(text, name, replacement):
    start, end = _find_block(text, name)
    return text[:start] + replacement + text[end:]


def _objects_of_interest(goal):
    """Objects/fixtures mentioned in the goal, region suffixes stripped."""
    terms = set(re.findall(r"[a-z_0-9]+_\d+(?:_[a-z_]+_region)?", goal))
    result = []
    for term in sorted(terms):
        base = term
        for fixture in FIXTURES:
            if term.startswith(fixture):
                base = fixture
                break
        if base in OBJECTS + FIXTURES and base != "main_table" and base not in result:
            result.append(base)
    return result


def anchor_bddl_path(anchor_task):
    from libero.libero import get_libero_path

    return (
        pathlib.Path(get_libero_path("bddl_files"))
        / "libero_goal"
        / (anchor_task + ".bddl")
    )


def generate(template_text, language, goal):
    text = template_text
    text = _replace_block(text, "language", "(:language {})".format(language))
    objs = "\n    ".join(_objects_of_interest(goal))
    text = _replace_block(
        text, "obj_of_interest", "(:obj_of_interest\n    {}\n  )".format(objs)
    )
    if not goal.strip().startswith("(And"):
        goal = "(And {})".format(goal)
    text = _replace_block(text, "goal", "(:goal\n    {}\n  )".format(goal))
    return text


def main():
    spec = yaml.safe_load((REPO_ROOT / "prompts.yaml").read_text())
    template = anchor_bddl_path(spec["anchor_task"]).read_text()
    OUT_DIR.mkdir(exist_ok=True)

    for prompt in spec["prompts"]:
        kind, _, name = prompt["bddl"].partition(":")
        if kind != "custom":
            continue
        out_path = OUT_DIR / (name + ".bddl")
        out_path.write_text(generate(template, prompt["text"], prompt["goal"]))
        print("wrote", out_path)


if __name__ == "__main__":
    main()
