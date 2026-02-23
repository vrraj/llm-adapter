# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release preparation
- Provider and model inference system for `normalize_adapter_response()`
- Enhanced parameter mapping for Gemini reasoning models
- Improved UI documentation and API call signatures
- Complete project metadata and URLs

### Changed
- Removed redundant `build_llm_result()` wrapper method
- Updated backend to use `normalize_adapter_response()` directly
- Simplified API structure and documentation

### Fixed
- Gemini parameter mapping issues with `thinking_budget`
- Provider resolution for registry-based model keys
- UI format request accuracy and educational value

## [0.1.0] - 2025-02-12

### Added
- Initial release of llm-adapter
- Unified API for LLM generation and embeddings
- Model Registry-driven provider resolution and parameter mapping
- FastAPI demo with interactive UI
- Support for OpenAI and Gemini providers
- Streaming support at library level
- Embedding magnitude metadata tracking
- Pricing metadata helpers
- Comprehensive documentation and examples
