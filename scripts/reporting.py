"""Small reporting invariants, testable without rendering dependencies."""


def episode_zero(entry):
    matches = [episode for episode in entry["episodes"]
               if type(episode.get("episode")) is int and episode["episode"] == 0]
    if len(matches) != 1:
        raise ValueError("{}: expected exactly one episode 0; refusing to substitute another episode".format(
            entry.get("text", "prompt")))
    return matches[0]


def video_groups(summary):
    entries = list(summary["prompts"].values())
    groups = tuple([entry for entry in entries if entry["group"] == group]
                   for group in ("in_distribution", "novel"))
    if any(not 1 <= len(group) <= 4 for group in groups):
        raise ValueError("summary.json must contain 1–4 prompts in each group; no panels are silently dropped")
    for group in groups:
        for entry in group:
            episode_zero(entry)
    return groups
