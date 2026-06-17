import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from ghc_api.token_usage_reporter import ANONYMOUS_USER_ID, get_token_usage_overview


class TokenUsageOverviewUserBackcompatTests(unittest.TestCase):
    def test_old_jsonl_without_user_id_is_anonymous(self) -> None:
        now_ts = int(time.time())
        with tempfile.TemporaryDirectory() as tmpdir:
            usage_file = Path(tmpdir) / "token_usage.jl"
            usage_file.write_text(
                "".join(
                    [
                        json.dumps({
                            "timestamp": now_ts - 120,
                            "models": [
                                {"model": "gpt-5", "request_count": 1, "input_tokens": 2, "output_tokens": 3}
                            ],
                        })
                        + "\n",
                        json.dumps({
                            "timestamp": now_ts - 60,
                            "user_id": "user-1",
                            "models": [
                                {"model": "gpt-5", "request_count": 2, "input_tokens": 4, "output_tokens": 6}
                            ],
                        })
                        + "\n",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch("ghc_api.token_usage_reporter._resolve_usage_files", return_value=[("local", usage_file)]):
                overview = get_token_usage_overview("all")
                filtered = get_token_usage_overview("all", user_filter="user-1")

        self.assertIn(ANONYMOUS_USER_ID, overview["users"])
        self.assertIn("user-1", overview["users"])
        self.assertEqual(overview["totals"]["request_count"], 3)
        self.assertEqual(overview["totals"]["total_tokens"], 15)
        self.assertEqual(
            sorted({row["user_id"] for row in overview["user_rows"]}),
            [ANONYMOUS_USER_ID, "user-1"],
        )

        self.assertEqual(filtered["user_filter"], "user-1")
        self.assertEqual(filtered["totals"]["request_count"], 2)
        self.assertEqual(filtered["totals"]["total_tokens"], 10)
        self.assertTrue(all(row["user_id"] == "user-1" for row in filtered["user_rows"]))


if __name__ == "__main__":
    unittest.main()
