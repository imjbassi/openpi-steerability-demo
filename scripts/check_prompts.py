"""Verify the prompt-group claims in prompts.yaml against LIBERO itself.

The pi05_libero checkpoint was fine-tuned on the converted
libero_spatial + libero_object + libero_goal + libero_10 suites, with the
task language strings used verbatim as prompts (openpi's LIBERO data config
sets prompt_from_task=True). So:

  - every in_distribution prompt must equal one of those 40 task strings
  - every novel prompt must equal none of them

Exits non-zero if any claim is wrong. Run inside the client venv:
  python scripts/check_prompts.py
"""

import pathlib
import sys

import yaml

TRAINING_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def training_languages():
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    languages = {}
    for suite_name in TRAINING_SUITES:
        suite = benchmark_dict[suite_name]()
        for task_id in range(suite.n_tasks):
            languages[str(suite.get_task(task_id).language)] = suite_name
    return languages


def main():
    spec = yaml.safe_load((REPO_ROOT / "prompts.yaml").read_text())
    languages = training_languages()
    print("{} training task strings across {}".format(len(languages), TRAINING_SUITES))

    failures = 0
    for prompt in spec["prompts"]:
        text, group = prompt["text"], prompt["group"]
        trained_in = languages.get(text)
        if group == "in_distribution":
            ok = trained_in is not None
            detail = "training task of {}".format(trained_in) if ok else "NOT in training set"
        else:
            ok = trained_in is None
            detail = "not in training set" if ok else "IS a training task of {}".format(trained_in)
        failures += 0 if ok else 1
        print(
            "{} {:<28} {:<16} {!r}: {}".format(
                "PASS" if ok else "FAIL", prompt["id"], group, text, detail
            )
        )

    if failures:
        print("\n{} prompt group claim(s) are wrong".format(failures))
        sys.exit(1)
    print("\nall prompt group claims verified")


if __name__ == "__main__":
    main()
