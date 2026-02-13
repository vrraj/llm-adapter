"""Test package imports and basic functionality for CI validation."""

def test_package_import():
    """Test that package can be imported correctly."""
    try:
        import llm_adapter
        print("✅ Package imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_llm_adapter_import():
    """Test that main llm_adapter module can be imported."""
    try:
        from llm_adapter import llm_adapter
        print("✅ llm_adapter module imports successfully")
        return True
    except ImportError as e:
        print(f"❌ llm_adapter import failed: {e}")
        return False

def test_modelspec_import():
    """Test that ModelSpec can be imported and instantiated."""
    try:
        from llm_adapter import ModelSpec
        spec = ModelSpec(provider="openai", model="gpt-4o-mini")
        print("✅ ModelSpec imports and creates successfully")
        return True
    except Exception as e:
        print(f"❌ ModelSpec test failed: {e}")
        return False

def test_registry_import():
    """Test that model registry can be accessed."""
    try:
        from llm_adapter import model_registry
        # Test that registry has entries
        if hasattr(model_registry, 'REGISTRY'):
            print(f"✅ Model registry accessible with {len(model_registry.REGISTRY)} models")
            return True
        else:
            print("❌ Model registry missing REGISTRY attribute")
            return False
    except Exception as e:
        print(f"❌ Model registry test failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    success &= test_package_import()
    success &= test_llm_adapter_import()
    success &= test_modelspec_import()
    success &= test_registry_import()
    
    if success:
        print("✅ All import tests passed")
        exit(0)
    else:
        print("❌ Some import tests failed")
        exit(1)
