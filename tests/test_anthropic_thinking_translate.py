import unittest

import ghc_api.state
from ghc_api.routes.anthropic import (
    apply_effort_policy,
    translate_thinking_enabled_to_adaptive,
)


FAKE_MODELS = {"data": [
    {"id": "claude-opus-4.8", "capabilities": {"supports": {"reasoning_effort": ["low", "medium", "high", "xhigh", "max"]}}},
    {"id": "claude-opus-4.6", "capabilities": {"supports": {"reasoning_effort": ["low", "medium", "high", "max"]}}},
    {"id": "gpt-5.4", "capabilities": {"supports": {"reasoning_effort": ["none", "low", "medium", "high", "xhigh"]}}},
    {"id": "claude-sonnet-4.5", "capabilities": {"supports": {}}},  # old protocol, no reasoning_effort
]}


def _enabled_payload(model, budget_tokens=8000, max_tokens=16000, with_effort=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
    }
    if with_effort is not None:
        payload["output_config"] = {"effort": with_effort}
    return payload


class TranslateThinkingEnabledTest(unittest.TestCase):
    def setUp(self):
        self._saved_models = ghc_api.state.state.models
        ghc_api.state.state.models = FAKE_MODELS

    def tearDown(self):
        ghc_api.state.state.models = self._saved_models

    def test_new_protocol_model_translates_thinking_type(self):
        result = translate_thinking_enabled_to_adaptive(
            _enabled_payload("claude-opus-4.8", budget_tokens=8000), "claude-opus-4.8"
        )
        self.assertEqual(result["thinking"], {"type": "adaptive"})

    def test_budget_lt_4096_maps_to_low(self):
        result = translate_thinking_enabled_to_adaptive(
            _enabled_payload("claude-opus-4.8", budget_tokens=2048, max_tokens=16000),
            "claude-opus-4.8",
        )
        self.assertEqual(result["output_config"]["effort"], "low")

    def test_budget_lt_16384_maps_to_medium(self):
        result = translate_thinking_enabled_to_adaptive(
            _enabled_payload("claude-opus-4.8", budget_tokens=8000), "claude-opus-4.8"
        )
        self.assertEqual(result["output_config"]["effort"], "medium")

    def test_budget_ge_16384_maps_to_high(self):
        result = translate_thinking_enabled_to_adaptive(
            _enabled_payload("claude-opus-4.8", budget_tokens=32000, max_tokens=64000),
            "claude-opus-4.8",
        )
        self.assertEqual(result["output_config"]["effort"], "high")

    def test_missing_budget_tokens_defaults_to_medium(self):
        payload = {
            "model": "claude-opus-4.8",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16000,
            "thinking": {"type": "enabled"},  # no budget_tokens
        }
        result = translate_thinking_enabled_to_adaptive(payload, "claude-opus-4.8")
        self.assertEqual(result["output_config"]["effort"], "medium")
        self.assertEqual(result["thinking"], {"type": "adaptive"})

    def test_client_supplied_effort_preserved(self):
        result = translate_thinking_enabled_to_adaptive(
            _enabled_payload("claude-opus-4.8", budget_tokens=8000, with_effort="max"),
            "claude-opus-4.8",
        )
        self.assertEqual(result["thinking"], {"type": "adaptive"})
        self.assertEqual(result["output_config"]["effort"], "max")

    def test_old_protocol_model_passes_through(self):
        payload = _enabled_payload("claude-sonnet-4.5", budget_tokens=8000)
        result = translate_thinking_enabled_to_adaptive(payload, "claude-sonnet-4.5")
        self.assertIs(result, payload)
        self.assertEqual(result["thinking"], {"type": "enabled", "budget_tokens": 8000})
        self.assertNotIn("output_config", result)

    def test_unknown_model_passes_through(self):
        payload = _enabled_payload("never-heard-of-this-model", budget_tokens=8000)
        result = translate_thinking_enabled_to_adaptive(payload, "never-heard-of-this-model")
        self.assertIs(result, payload)

    def test_thinking_already_adaptive_passes_through(self):
        payload = {
            "model": "claude-opus-4.8",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16000,
            "thinking": {"type": "adaptive"},
        }
        result = translate_thinking_enabled_to_adaptive(payload, "claude-opus-4.8")
        self.assertIs(result, payload)

    def test_no_thinking_passes_through(self):
        payload = {
            "model": "claude-opus-4.8",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16000,
        }
        result = translate_thinking_enabled_to_adaptive(payload, "claude-opus-4.8")
        self.assertIs(result, payload)

    def test_max_tokens_bumped_when_budget_exceeds(self):
        # client set max_tokens=4096 with budget_tokens=8000 (the original code's
        # adjust_max_tokens_for_thinking would have bumped this; we preserve that)
        result = translate_thinking_enabled_to_adaptive(
            _enabled_payload("claude-opus-4.8", budget_tokens=8000, max_tokens=4096),
            "claude-opus-4.8",
        )
        self.assertEqual(result["max_tokens"], 16000)  # 8000 + min(16384, 8000)

    def test_max_tokens_not_touched_when_adequate(self):
        result = translate_thinking_enabled_to_adaptive(
            _enabled_payload("claude-opus-4.8", budget_tokens=8000, max_tokens=32000),
            "claude-opus-4.8",
        )
        self.assertEqual(result["max_tokens"], 32000)

    def test_preserves_other_output_config_keys(self):
        payload = _enabled_payload("claude-opus-4.8", budget_tokens=8000)
        payload["output_config"] = {"extra": "keep"}
        result = translate_thinking_enabled_to_adaptive(payload, "claude-opus-4.8")
        self.assertEqual(result["output_config"], {"effort": "medium", "extra": "keep"})


class TranslateThenEffortGateTest(unittest.TestCase):
    """End-to-end: translation followed by apply_effort_policy."""

    def setUp(self):
        self._saved_models = ghc_api.state.state.models
        ghc_api.state.state.models = FAKE_MODELS

    def tearDown(self):
        ghc_api.state.state.models = self._saved_models

    def test_mapped_effort_survives_when_supported(self):
        payload = _enabled_payload("claude-opus-4.8", budget_tokens=8000)
        translated = translate_thinking_enabled_to_adaptive(payload, "claude-opus-4.8")
        final = apply_effort_policy(translated, "claude-opus-4.8")
        self.assertEqual(final["thinking"], {"type": "adaptive"})
        self.assertEqual(final["output_config"], {"effort": "medium"})

    def test_unsupported_client_effort_dropped_after_translation(self):
        # User-supplied "max" on gpt-5.4 (which only has up to xhigh) — survives
        # translation (preserved), then dropped by apply_effort_policy.
        payload = _enabled_payload("gpt-5.4", budget_tokens=8000, with_effort="max")
        translated = translate_thinking_enabled_to_adaptive(payload, "gpt-5.4")
        self.assertEqual(translated["output_config"]["effort"], "max")
        final = apply_effort_policy(translated, "gpt-5.4")
        self.assertNotIn("output_config", final)
        # thinking is still adaptive — bare adaptive should still be accepted
        self.assertEqual(final["thinking"], {"type": "adaptive"})


if __name__ == "__main__":
    unittest.main()
