import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rollout_log import read_records, positive_episode_count


class RolloutLogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "log.jsonl"
        self.record = {"prompt_id": "bowl", "episode": 0, "success": True}

    def write(self, records):
        self.path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    def test_missing_and_blank_log(self):
        self.assertEqual(read_records(self.path), [])
        self.path.write_text("\n\n", encoding="utf-8")
        self.assertEqual(read_records(self.path), [])

    def test_distinct_prompts_can_share_episode_index(self):
        records = [self.record, dict(self.record, prompt_id="stove")]
        self.write(records)
        self.assertEqual(read_records(self.path), records)

    def test_duplicate_results_cannot_inflate_summary(self):
        self.write([self.record, dict(self.record, success=False)])
        with self.assertRaisesRegex(ValueError, r"log.jsonl:2: duplicate"):
            read_records(self.path)

    def test_truncated_tail_is_not_silently_ignored(self):
        self.write([self.record])
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write('\n{"prompt_id":')
        with self.assertRaisesRegex(ValueError, r"log.jsonl:2:"):
            read_records(self.path)

    def test_invalid_record_types(self):
        for record in [[], dict(self.record, episode=True), dict(self.record, episode=-1),
                       dict(self.record, success="false"), dict(self.record, prompt_id="")]:
            with self.subTest(record=record):
                self.write([record])
                with self.assertRaises(ValueError):
                    read_records(self.path)

    def test_episode_override_must_be_positive(self):
        for value in ["0", "-2", "bad"]:
            with self.assertRaises(ValueError):
                positive_episode_count(value)
        self.assertEqual(positive_episode_count("2"), 2)

    def test_published_log_has_unique_valid_records(self):
        records = read_records(Path(__file__).resolve().parents[1] / "results" / "log.jsonl")
        self.assertEqual(len(records), 80)


if __name__ == "__main__":
    unittest.main()
