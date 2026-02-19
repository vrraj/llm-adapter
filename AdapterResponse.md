# AdapterResponse Structure and Serialization

## Overview

When using the demo UI or API endpoints, responses are wrapped in an `AdapterResponse` structure that provides both raw and transformed data for maximum flexibility.

## AdapterResponse Format

```json
{
  "output_text": "The model's response text",
  "model": "models/gemini-3-flash-preview",
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 11,
    "total_tokens": 264,
    "reasoning_tokens": 245,
    "cached_tokens": 0
  },
  "metadata": {
    "provider": "gemini",
    "model": "models/gemini-3-flash-preview",
    "model_key": "gemini:native-sdk-3-flash-preview"
  },
  "adapter_response": {
    // Standardized, provider-agnostic format
    "output_text": "The model's response text",
    "usage": { /* normalized usage metrics */ },
    "model": "models/gemini-3-flash-preview"
  },
  "model_response": {
    // Raw provider response (JSON-serialized)
    "candidates": [ /* raw SDK objects */ ],
    "usageMetadata": { /* raw usage in provider format */ }
  },
  "status": "incomplete",
  "finish_reason": "FinishReason.MAX_TOKENS"
}
```

## Key Fields

| Field | Type | Description |
|-------|------|-------------|
| **`adapter_response`** | Object | Provider-agnostic, standardized format for consistent consumption |
| **`model_response`** | Object | Raw provider response (JSON-serialized version of SDK objects) |
| **`usage`** | Object | Normalized usage metrics across all providers |
| **`raw`** | Object | In embedding responses, contains the original provider response |
| **`output_text`** | String | Extracted response text (convenience field) |
| **`metadata`** | Object | Provider, model, and routing information |
| **`status`** | String | Response status (e.g., "complete", "incomplete") |
| **`finish_reason`** | String | Why the generation stopped |

## Serialization Details

**Important**: The `model_response` field contains JSON-serialized data, not raw Python objects. This serialization process:

1. **Converts binary data to Base64**: Binary fields like `thought_signature` are encoded as Base64 strings
2. **Preserves original field names**: Provider SDK field names (often CamelCase) are maintained
3. **Handles complex objects**: Nested SDK objects are recursively serialized to JSON dictionaries

### Example: Gemini SDK Field Name Mapping

Raw SDK objects may have different field naming conventions:

```python
# Raw SDK response (debug view)
usage_metadata=GenerateContentResponseUsageMetadata(
  prompt_token_count=8,        # Python SDK documentation shows snake_case
  candidates_token_count=11,
  thoughts_token_count=245,
  total_token_count=264
)

# JSON-serialized model_response
"usageMetadata": {
  "promptTokenCount": 8,        // Actual SDK field names are CamelCase
  "candidatesTokenCount": 11,
  "thoughtsTokenCount": 245,
  "totalTokenCount": 264
}
```

### Why This Happens

- **SDK objects** use internal field names (often CamelCase like `promptTokenCount`)
- **Python documentation** may show snake_case aliases (`prompt_token_count`)
- **JSON serialization** preserves the actual field names from the SDK objects
- **Usage extraction functions** handle both naming conventions via helper functions

## Usage Normalization

The adapter normalizes usage metrics across providers:

| Provider Raw Field | Normalized Field | Description |
|-------------------|------------------|-------------|
| `prompt_token_count` / `promptTokenCount` | `prompt_tokens` | Input tokens |
| `candidates_token_count` / `completion_tokens` | `completion_tokens` | Output tokens |
| `total_token_count` / `total_tokens` | `total_tokens` | Total tokens |
| `thoughts_token_count` / `reasoning_tokens` | `reasoning_tokens` | Reasoning tokens |
| `cached_content_token_count` / `cached_tokens` | `cached_tokens` | Cached tokens |

This normalization ensures consistent usage reporting regardless of the underlying provider's field naming conventions.

## Binary Data Handling

Binary fields in SDK responses are automatically converted to Base64 strings during JSON serialization:

```python
# Raw SDK response
thought_signature=b"\x12\xb9\x07\n\xb6\x07\x01\xbe>\xf6\xfbV\xc6\x8ff\xb1YUe\xefh\xd3!\x90U\x1f\x8a\xac\xdb\x9a\x16|\xcd\xf6\xbc1\xa3mk\x1b='\xd1\xbd\xa2\x85\xfe\xff\x10?N\xfbv\x02\x1f\xb5\x7f\xadS\xf0\xbe\xa9$\x1e\xd3a\xfe\x98\x90\n\x9c\xed'O\xbb\x06\xabZ?D\xf0U*m\xee\xd3\x0czi\xa39\x95\xe3\xec\x9eK\x96..."

# JSON-serialized response
"thoughtSignature": "EsQHCsEHAb4-9vs935PsDSiyGQWuPntIMzLk4EViWMCgB85L0aqcI8lk0PxCJ3qgzyQqc0fmO-9NJ0_p9FSvikv49CNSPnX6Irudx6dMlhBfZnIrH2Hx5fkiRtMa8jo-K8LsXG2kJicLA0essvpMTctj98xdZbCEiZVfJjQ5wcvhc0OxRvYyaYR7svZ7O4Hulvc5KSif7Qo-e4-_wcTzpSn1uHNlIJJ3JpBfI6mhkTDHv8NqNouuChMldM0srxB4hjWlOyg2wztFXhy5mlYRsSG5cw-riQjv5oHlRmgAeYJaQsqije7tOAeS1Mspq6VEs4JIcsekv4OuLhdUIS5CxlFF9JXS5SS7QSBEpPmJIim5waieIU1aPtGLcOevK0B7O3tl51awlwiPzEO2ZNYZZk6euRbXZGJJZ4yle5U5ABxU1RqiMV_8dk7_h9rxdtyGV82zsdNdgS6M26_NskE_j2bCB_GbpTPIc-LycAbD1LB_pU4wfKuhE_7Cdo2GccHaG6s_NqDVcN6bI6F2gPspwBurg8KCUosRUcpDcwb8nplGCKriyrWoW0lVmNvqdB-Xr5TEMQvW0zypA9Or39rKmok-RSd2vOs4VZzP82_71DwB6OSB5jYbZvVkK_LVMWhQpnWFmRuPd0hNfo6TLhB-4JDyOcDoGOkwJUkQ8dY5N8ahEQ4W99dCVEAanz_Z-UBQf32OniRo78IrsaSamtHBFV32ND_8xfF1yFqAGoNh9nruCpk3LRzdmBYtdc3VihKtrhCkbHhTDN8B9JdHYNNj1Rk_WKecqk1BQ3mxvkjVz2sLwtEcv3ZGCxn_b0t3N8_O8MdXmiTEipuxmGcry-9Ke_3NHZeLCArv4tex4iP3mp_dn8nfMAeLh9RGH_bqWzSOXOS5J1YW9LSucXHOPgZjGQHndA05UlDRmaUOPt9ETwRnviacxUKYx5wurTS1UiLEB4tDJISNVepIAAGnDym8efsxFTAfV1oqF-kwtT7rT85Q0JmdXBCs3GufjE0-skmEikVTqA_CqCNdKSWMMsNOBSaqh1EP2ECCwh_-TtaDYeNfVf4OHuIYhlo9uYhgcJWLs95abBTjg02YJlmioV5JyR9N-LWpR2FyKv4_vlXVW8uE-ZshkE9n3fdyEMHyWUwyVVtTnd4GPdfW55WYdp-_1EW_CoHkAeAw_eT1euzj7_h1iHVOX9skbYUeOyPTgvoL1eOv8zyuKFjrKgO9-lTacuDKws4HZp-a-HSzwl5AG3eTh-PWfAfBV8xwfoJWXipHu_Vj4qaDw=="
```

## Provider-Specific Examples

### OpenAI Chat Completions

```json
{
  "model_response": {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "Hello! How can I help you today?"
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 9,
      "completion_tokens": 9,
      "total_tokens": 18
    }
  }
}
```

### Gemini Native SDK

```json
{
  "model_response": {
    "candidates": [
      {
        "content": {
          "parts": [
            {
              "text": "Hello! How can I help you today?",
              "thoughtSignature": "Base64-encoded-binary-data..."
            }
          ],
          "role": "model"
        },
        "finishReason": "STOP",
        "index": 0
      }
    ],
    "usageMetadata": {
      "promptTokenCount": 9,
      "candidatesTokenCount": 9,
      "totalTokenCount": 18
    }
  }
}
```

## Usage in Code

### Accessing Raw vs Normalized Data

```python
# Using the adapter_response (normalized)
result = llm_adapter.build_llm_result_from_response(resp)
print(result["text"])                    # Normalized text
print(result["usage"])                   # Normalized usage

# Accessing model_response directly (raw)
if hasattr(resp, 'model_response'):
    raw_usage = resp.model_response.get("usage", {})
    raw_candidates = resp.model_response.get("candidates", [])
```

### Debugging with Raw Responses

The `model_response` field is valuable for debugging because it contains the exact response from the underlying provider SDK, including all provider-specific fields and metadata.

## Best Practices

1. **Use `adapter_response`** for application logic (provider-agnostic)
2. **Use `model_response`** for debugging and provider-specific features
3. **Use normalized `usage`** for cost calculation and analytics
4. **Handle missing fields gracefully** - not all providers support all features
5. **Check provider** in `metadata` to understand response format differences

## Migration Notes

When migrating from direct SDK usage to the adapter:

- Replace direct field access with normalized equivalents
- Update usage calculations to use normalized fields
- Add error handling for missing optional fields
- Consider using `build_llm_result_from_response()` for maximum compatibility
