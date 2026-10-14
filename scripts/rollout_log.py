"""Dependency-free validation shared by resume and aggregation."""

import json


def read_records(path):
    if not path.exists():
        return []
    records, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected an episode object")
                if not isinstance(record.get("prompt_id"), str) or not record["prompt_id"]:
                    raise ValueError("missing prompt_id")
                episode = record.get("episode")
                if type(episode) is not int or episode < 0:
                    raise ValueError("episode must be a nonnegative integer")
                if type(record.get("success")) is not bool:
                    raise ValueError("success must be a boolean")
                key = (record["prompt_id"], episode)
                if key in seen:
                    raise ValueError("duplicate episode {}".format(key))
            except ValueError as error:
                raise ValueError("{}:{}: {}. Repair the log before resuming or reporting.".format(
                    path, line_number, error
                )) from error
            seen.add(key)
            records.append(record)
    return records


def positive_episode_count(value):
    count = int(value)
    if count <= 0:
        raise ValueError("episode count must be positive")
    return count
