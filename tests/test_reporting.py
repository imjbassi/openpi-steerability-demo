import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reporting import episode_zero, video_groups
import readme_table


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
        self.entry = next(iter(self.summary["prompts"].values()))

    def test_reordered_episodes_keep_zero_video_and_outcome(self):
        original = copy.deepcopy(self.entry["episodes"][0])
        self.entry["episodes"].reverse()
        self.assertEqual(episode_zero(self.entry), original)

    def test_no_substitution_for_missing_zero(self):
        self.entry["episodes"] = self.entry["episodes"][1:]
        with self.assertRaisesRegex(ValueError, "episode 0"):
            video_groups(self.summary)

    def test_duplicate_zero_is_ambiguous(self):
        self.entry["episodes"].append(copy.deepcopy(self.entry["episodes"][0]))
        with self.assertRaises(ValueError):
            episode_zero(self.entry)

    def test_extra_prompts_are_not_silently_dropped(self):
        self.summary["prompts"]["extra"] = copy.deepcopy(self.entry)
        with self.assertRaisesRegex(ValueError, "silently dropped"):
            video_groups(self.summary)

    def test_published_video_groups_remain_four_each(self):
        self.assertEqual([len(group) for group in video_groups(self.summary)], [4, 4])

    def test_mock_table_is_visibly_labeled(self):
        self.summary["server_metadata"] = {"mock": True}
        self.assertIn("MOCK SERVER DATA", readme_table.build_table(self.summary))

    def test_mock_summary_cannot_overwrite_readme(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            self.summary["server_metadata"] = {"mock": True}
            (root / "results" / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
            readme = root / "README.md"
            readme.write_text("original results", encoding="utf-8")
            with patch.object(readme_table, "REPO_ROOT", root), self.assertRaises(SystemExit):
                readme_table.main()
            self.assertEqual(readme.read_text(encoding="utf-8"), "original results")

    def test_published_table_is_unchanged(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        table = readme.split(readme_table.START)[1].split(readme_table.END)[0].strip()
        self.assertEqual(readme_table.build_table(self.summary), table)


if __name__ == "__main__":
    unittest.main()
