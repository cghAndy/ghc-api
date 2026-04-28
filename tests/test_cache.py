import unittest
from datetime import datetime
from unittest import mock

from flask import Flask

from ghc_api.cache import RequestCache
from ghc_api.routes import dashboard as dashboard_routes


class RequestCacheTimestampTests(unittest.TestCase):
    def test_start_request_uses_unix_timestamp(self) -> None:
        cache = RequestCache()

        cache.start_request("req-1", {"model": "gpt-5", "endpoint": "/v1/chat/completions"})

        item = cache.get_request("req-1")
        self.assertIsNotNone(item)
        self.assertIsInstance(item["timestamp"], int)
        self.assertGreater(item["timestamp"], 0)

    def test_import_request_converts_iso_timestamp_to_unix(self) -> None:
        cache = RequestCache()
        iso_timestamp = "2026-03-22T10:20:30"

        cache.import_request({
            "id": "req-1",
            "timestamp": iso_timestamp,
            "model": "gpt-5",
            "endpoint": "/v1/chat/completions",
        })

        item = cache.get_request("req-1")
        self.assertIsNotNone(item)
        self.assertEqual(
            item["timestamp"],
            int(datetime.fromisoformat(iso_timestamp).timestamp()),
        )

    def test_import_request_keeps_numeric_timestamp(self) -> None:
        cache = RequestCache()

        cache.import_request({
            "id": "req-1",
            "timestamp": 1711111111,
            "model": "gpt-5",
            "endpoint": "/v1/chat/completions",
        })

        item = cache.get_request("req-1")
        self.assertIsNotNone(item)
        self.assertEqual(item["timestamp"], 1711111111)


class DashboardRequestTimestampTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self.app.register_blueprint(dashboard_routes.dashboard_bp)
        self.client = self.app.test_client()

    def test_request_routes_return_unix_timestamp_for_imported_legacy_record(self) -> None:
        cache = RequestCache()
        iso_timestamp = "2026-03-22T10:20:30"
        expected_timestamp = int(datetime.fromisoformat(iso_timestamp).timestamp())
        cache.import_request({
            "id": "req-1",
            "timestamp": iso_timestamp,
            "request_body": {"messages": []},
            "response_body": {"id": "resp-1"},
            "model": "gpt-5",
            "endpoint": "/v1/chat/completions",
        })

        with mock.patch.object(dashboard_routes, "cache", cache):
            detail_response = self.client.get("/api/request/req-1")
            list_response = self.client.get("/api/requests")

        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(list_response.status_code, 200)

        detail_data = detail_response.get_json()
        self.assertEqual(detail_data["timestamp"], expected_timestamp)
        self.assertIsInstance(detail_data["timestamp"], int)

        list_data = list_response.get_json()
        self.assertEqual(list_data["items"][0]["timestamp"], expected_timestamp)
        self.assertIsInstance(list_data["items"][0]["timestamp"], int)


class RequestCacheMemoryEvictionTests(unittest.TestCase):
    def _make_body(self, size_bytes: int) -> str:
        """Return a string of approximately `size_bytes` UTF-8 bytes."""
        return "x" * size_bytes

    def test_size_based_eviction_removes_old_entries(self) -> None:
        """Inserting entries totalling >1 MB should evict the oldest ones."""
        cache = RequestCache(max_size_mb=1, max_entries=10000)
        body = self._make_body(300 * 1024)  # 300 KB each

        for i in range(5):  # 5 * 300 KB = 1500 KB > 1 MB limit
            cache.start_request(f"req-{i}", {
                "model": "gpt-5",
                "endpoint": "/v1/chat/completions",
                "request_body": body,
            })

        self.assertLessEqual(cache.current_size_bytes, cache.max_size_bytes)
        # Oldest entries should have been evicted
        self.assertIsNone(cache.get_request("req-0"))

    def test_size_based_eviction_current_size_stays_within_limit(self) -> None:
        """current_size_bytes must never exceed max_size_bytes after eviction."""
        cache = RequestCache(max_size_mb=1, max_entries=10000)
        body = self._make_body(200 * 1024)  # 200 KB each

        for i in range(10):
            cache.start_request(f"req-{i}", {
                "model": "gpt-5",
                "endpoint": "/v1/chat/completions",
                "request_body": body,
            })

        self.assertLessEqual(cache.current_size_bytes, cache.max_size_bytes)

    def test_max_entries_safety_cap(self) -> None:
        """max_entries should limit the cache even when memory is not exceeded."""
        cache = RequestCache(max_size_mb=10000, max_entries=3)

        for i in range(5):
            cache.start_request(f"req-{i}", {
                "model": "gpt-5",
                "endpoint": "/v1/chat/completions",
            })

        self.assertEqual(cache.get_total_count(), 3)
        # Oldest entries evicted
        self.assertIsNone(cache.get_request("req-0"))
        self.assertIsNone(cache.get_request("req-1"))
        self.assertIsNotNone(cache.get_request("req-2"))
        self.assertIsNotNone(cache.get_request("req-3"))
        self.assertIsNotNone(cache.get_request("req-4"))

    def test_complete_request_updates_size_and_triggers_eviction(self) -> None:
        """Completing a request with a large response body should update size and evict if needed."""
        cache = RequestCache(max_size_mb=1, max_entries=10000)
        small_body = self._make_body(10 * 1024)  # 10 KB

        # Insert several small entries
        for i in range(3):
            cache.start_request(f"req-{i}", {
                "model": "gpt-5",
                "endpoint": "/v1/chat/completions",
                "request_body": small_body,
            })

        size_before = cache.current_size_bytes

        # Complete req-2 with a large response body (900 KB)
        large_response = self._make_body(900 * 1024)
        cache.complete_request("req-2", {
            "response_body": large_response,
            "status_code": 200,
        })

        # Size should have increased by the delta
        self.assertGreater(cache.current_size_bytes, size_before)
        # And must not exceed the limit
        self.assertLessEqual(cache.current_size_bytes, cache.max_size_bytes)

    def test_update_request_state_adjusts_size(self) -> None:
        """update_request_state with new body content should adjust current_size_bytes."""
        cache = RequestCache(max_size_mb=10, max_entries=10000)
        small_body = self._make_body(1024)  # 1 KB

        cache.start_request("req-1", {
            "model": "gpt-5",
            "endpoint": "/v1/chat/completions",
            "request_body": small_body,
        })
        size_after_start = cache.current_size_bytes

        large_body = self._make_body(100 * 1024)  # 100 KB
        cache.update_request_state("req-1", RequestCache.STATE_RECEIVING,
                                   response_body=large_body)

        self.assertGreater(cache.current_size_bytes, size_after_start)

    def test_entry_sizes_and_current_size_bytes_consistent(self) -> None:
        """entry_sizes values must sum to current_size_bytes."""
        cache = RequestCache(max_size_mb=10, max_entries=10000)

        for i in range(5):
            cache.start_request(f"req-{i}", {
                "model": "gpt-5",
                "endpoint": "/v1/chat/completions",
                "request_body": self._make_body((i + 1) * 1024),
            })

        self.assertEqual(cache.current_size_bytes, sum(cache.entry_sizes.values()))

    def test_get_stats_includes_cache_size_fields(self) -> None:
        """get_stats() should include the new cache size fields."""
        cache = RequestCache(max_size_mb=200, max_entries=10000)
        cache.start_request("req-1", {"model": "gpt-5", "endpoint": "/v1/chat"})

        stats = cache.get_stats()
        self.assertIn("cache_size_bytes", stats)
        self.assertIn("cache_size_mb", stats)
        self.assertIn("cache_max_size_mb", stats)
        self.assertIn("cache_entry_count", stats)
        self.assertIn("cache_max_entries", stats)
        self.assertEqual(stats["cache_max_size_mb"], 200)
        self.assertEqual(stats["cache_max_entries"], 10000)
        self.assertEqual(stats["cache_entry_count"], 1)


if __name__ == "__main__":
    unittest.main()
