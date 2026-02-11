from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Iterable, Optional


def _infer_provider_from_model_key(model_key: str) -> str:
    mk = (model_key or "").strip()
    if ":" in mk:
        return mk.split(":", 1)[0].strip().lower() or "openai"
    return "openai"


def _get_event_type(ev: Any) -> Optional[str]:
    if ev is None:
        return None
    if isinstance(ev, dict):
        t = ev.get("type") or ev.get("event")
        return str(t) if t is not None else None
    t = getattr(ev, "type", None)
    if t is None:
        t = getattr(ev, "event", None)
    return str(t) if t is not None else None


def _get_event_delta(ev: Any) -> Optional[str]:
    if ev is None:
        return None
    if isinstance(ev, dict):
        d = ev.get("delta")
        return str(d) if d is not None else None
    d = getattr(ev, "delta", None)
    if d is not None:
        return str(d)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Test llm_adapter.create(stream=True) via CLI")
    parser.add_argument("--model-key", required=True, help='Model key or model name (e.g. "openai:fast" or "gemini:native-sdk-3-flash-preview")')
    parser.add_argument("--prompt", default=None, help="Prompt to send (if omitted, read from stdin)")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--reasoning-effort", default=None, help="Reasoning effort (none/minimal/low/medium/high)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p")
    parser.add_argument("--provider", default=None, help="Override provider (openai/gemini). If omitted, inferred from model-key prefix.")
    args = parser.parse_args()

    try:
        from llm_adapter import llm_adapter  # type: ignore[import]
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Could not import llm_adapter: {e}")
        sys.exit(1)

    provider = (args.provider or _infer_provider_from_model_key(args.model_key)).strip().lower()

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] OPENAI_API_KEY is not set; OpenAI streaming will fail.")
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        print("[WARNING] GEMINI_API_KEY is not set; Gemini streaming will fail.")

    prompt = args.prompt
    if prompt is None:
        prompt = sys.stdin.read().strip()
    prompt = (prompt or "").strip()
    if not prompt:
        print("[ERROR] Empty prompt. Provide --prompt or pipe text into stdin.")
        sys.exit(1)

    kwargs: dict[str, Any] = {}
    if args.max_output_tokens is not None:
        kwargs["max_output_tokens"] = int(args.max_output_tokens)
    if args.reasoning_effort is not None:
        kwargs["reasoning_effort"] = str(args.reasoning_effort)
    if args.temperature is not None:
        kwargs["temperature"] = float(args.temperature)
    if args.top_p is not None:
        kwargs["top_p"] = float(args.top_p)

    print("=== llm_adapter.create(stream=True) ===")
    print(f"provider: {provider}")
    print(f"model:    {args.model_key}")
    print(f"prompt:   {prompt}")
    if kwargs:
        print(f"kwargs:   {kwargs}")
    print("----------------------------------------")

    try:
        resp = llm_adapter.create(
            provider=provider,
            model=args.model_key,
            input=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs,
        )
    except Exception as e:
        print(f"[ERROR] llm_adapter.create failed: {e}")
        sys.exit(1)

    # Streaming: resp is expected to be an iterable of events.
    if not isinstance(resp, Iterable) or isinstance(resp, (str, bytes)):
        print("[ERROR] Expected a streaming iterable response but got a non-iterable result.")
        try:
            print(repr(resp))
        except Exception:
            pass
        sys.exit(1)

    collected: list[str] = []

    try:
        for ev in resp:
            et = _get_event_type(ev)
            if et == "response.output_text.delta":
                delta = _get_event_delta(ev) or ""
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    collected.append(delta)
                continue

            if et == "response.output_text.done":
                break

            # Unknown event type: ignore by default.
            continue
    except Exception as e:
        print(f"\n[ERROR] Streaming iteration failed: {e}")
        sys.exit(1)

    final_text = "".join(collected).strip()
    print("\n\n=== Final (collected) ===")
    print(final_text or "<no text collected>")


if __name__ == "__main__":  # pragma: no cover
    main()
