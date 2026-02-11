# Embedding Magnitude Metadata

The standalone llm-adapter now supports magnitude metadata for embedding responses, providing transparency into the embedding normalization process.

## Features

### **Magnitude Tracking**
- **Original Magnitude**: Stores the L2 norm before normalization
- **Normalization Flag**: Indicates if the embedding was normalized
- **Provider Information**: Tracks which provider generated the embedding

### **Supported Providers**
- **OpenAI**: Magnitude calculation (embeddings are pre-normalized)
- **Gemini**: Magnitude tracking + optional normalization
- **Future-Proof**: Automatically supports any provider that adds magnitude metadata

## Usage

### **Basic Usage**
```python
from llm_adapter import llm_adapter

# OpenAI embeddings (magnitude calculated automatically)
result = llm_adapter.embeddings.create(
    provider="openai",
    model="text-embedding-3-small", 
    input="Your text here"
)

# Access magnitude metadata
for item in result.data:
    magnitude = getattr(item, 'magnitude', None)
    normalized = getattr(item, 'normalized', False)
    provider = getattr(item, 'provider', 'unknown')
    
    print(f"Magnitude: {magnitude}")
    print(f"Normalized: {normalized}")
    print(f"Provider: {provider}")
```

### **Gemini with Normalization**
```python
# Gemini embeddings with normalization
result = llm_adapter.embeddings.create(
    provider="gemini",
    model="gemini-embedding-001",
    input="Your text here",
    normalize_embedding=True  # Enable normalization
)

for item in result.data:
    magnitude = getattr(item, 'magnitude', None)  # Original magnitude
    normalized = getattr(item, 'normalized', False)  # True if normalized
    provider = getattr(item, 'provider', 'gemini_adapter')
    
    print(f"Original magnitude: {magnitude}")
    print(f"Normalized: {normalized}")
```

## Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `magnitude` | `float` or `None` | L2 norm before normalization |
| `normalized` | `bool` | Whether embedding was normalized |
| `provider` | `str` | Provider that generated the embedding |

## Benefits

### **Debugging & Quality Control**
- Verify normalization worked correctly
- Detect anomalies in embedding generation
- Monitor magnitude distributions

### **Provider Comparison**
- Compare magnitude characteristics across providers
- Track normalization behavior
- Analyze provider-specific patterns

### **Future Compatibility**
- Automatically captures metadata when providers add it
- Backward compatible with existing code
- No breaking changes to current API

## Testing

Run the test script to verify functionality:

```bash
# Set your API keys
export OPENAI_API_KEY=your_openai_key
export GEMINI_API_KEY=your_gemini_key

# Run the test
python test_magnitude_metadata.py
```

## Implementation Notes

- **Backward Compatible**: Existing code continues to work unchanged
- **Safe Access**: Use `getattr()` to safely access metadata fields
- **Error Handling**: Graceful fallback if magnitude calculation fails
- **Performance**: Minimal overhead for magnitude calculations

## Integration Examples

### **Vector Database Storage**
```python
# Store magnitude with vectors in your database
for item in result.data:
    vector_data = {
        "vector": item.embedding,
        "magnitude": getattr(item, 'magnitude', None),
        "normalized": getattr(item, 'normalized', False),
        "provider": getattr(item, 'provider', 'unknown')
    }
    # Store in your vector database...
```

### **Quality Monitoring**
```python
# Monitor magnitude distributions
magnitudes = [getattr(item, 'magnitude', None) for item in result.data]
avg_magnitude = sum(m for m in magnitudes if m) / len(magnitudes)
print(f"Average magnitude: {avg_magnitude}")
```
