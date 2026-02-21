
# To test : export LLM_ADAPTER_ALLOWED_MODELS="openai:gpt-4o-mini"
# python -c "from llm_adapter.llm_adapter import LLMAdapter; llm=LLMAdapter(); print(llm.allowed_model_keys)"
# python -m unittest tests.test_registry -v


import unittest
# Environment-fallback test for allowed model keys:
import os


class TestAllowedModelKeys(unittest.TestCase):
    def test_allowlist_requires_registry_key(self):
        from llm_adapter.llm_adapter import LLMAdapter, LLMError

        llm = LLMAdapter(allowed_model_keys={"openai:gpt-4o-mini"})

        with self.assertRaises(LLMError) as ctx:
            llm._lookup_model_info_from_registry("gpt-4o-mini")  # provider-native name
        err = ctx.exception
        self.assertEqual(getattr(err, "kind", None), "config")
        self.assertEqual(getattr(err, "code", None), "model_key_required")

    def test_allowlist_allows_explicit_allowed_key(self):
        from llm_adapter.llm_adapter import LLMAdapter

        llm = LLMAdapter(allowed_model_keys={"openai:gpt-4o-mini"})
        mi = llm._lookup_model_info_from_registry("openai:gpt-4o-mini")
        self.assertIsNotNone(mi)

    def test_allowlist_blocks_other_existing_keys(self):
        from llm_adapter.llm_adapter import LLMAdapter, LLMError

        llm = LLMAdapter(allowed_model_keys={"openai:gpt-4o-mini"})

        # This key should exist in your default registry (you showed it earlier).
        with self.assertRaises(LLMError) as ctx:
            llm._lookup_model_info_from_registry("openai:embed_small")
        err = ctx.exception
        self.assertEqual(getattr(err, "kind", None), "config")
        self.assertEqual(getattr(err, "code", None), "model_not_allowed")





class TestAllowedModelKeysEnvFallback(unittest.TestCase):
    def test_env_fallback_sets_allowlist(self):
        from llm_adapter.llm_adapter import LLMAdapter

        old = os.environ.get("LLM_ADAPTER_ALLOWED_MODELS")
        try:
            os.environ["LLM_ADAPTER_ALLOWED_MODELS"] = "openai:gpt-4o-mini, openai:embed_small"
            llm = LLMAdapter()
            self.assertEqual(
                llm.allowed_model_keys,
                {"openai:gpt-4o-mini", "openai:embed_small"},
            )
        finally:
            if old is None:
                os.environ.pop("LLM_ADAPTER_ALLOWED_MODELS", None)
            else:
                os.environ["LLM_ADAPTER_ALLOWED_MODELS"] = old

