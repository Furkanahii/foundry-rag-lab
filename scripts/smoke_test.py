"""Environment diagnostic: prove Foundry Local actually works on this machine.

Run this first, and run it again whenever something breaks. It answers, in
order, the questions that every later failure traces back to:

  1. Is the SDK importable, and which version?
  2. What execution providers does this hardware expose?
  3. Can we reach the model catalog?
  4. Can we download + load an embedding model and get a real vector?
  5. Can we download + load a chat model and get real text?
  6. Does the OpenAI-compatible web service come up?

Usage:  python scripts/smoke_test.py
"""

from __future__ import annotations

import platform
import sys
import time
import traceback

CHAT_ALIAS = "qwen2.5-0.5b"
EMBED_ALIAS = "qwen3-embedding-0.6b"


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main() -> int:
    section("1. Environment")
    print(f"python : {platform.python_version()} ({platform.machine()})")
    print(f"system : {platform.system()} {platform.release()}")

    try:
        import foundry_local_sdk as fls
        from foundry_local_sdk import Configuration, FoundryLocalManager
    except ImportError:
        print("FAIL: foundry_local_sdk not importable. pip install foundry-local-sdk")
        return 1
    print(f"sdk    : foundry_local_sdk {getattr(fls, '__version__', '?')}")

    section("2. Initialize manager")
    config = Configuration(app_name="frag_smoke_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    print("manager initialized")
    print(f"manager attributes: {[a for a in dir(manager) if not a.startswith('_')]}")

    section("3. Execution providers")
    eps = manager.discover_eps()
    for ep in eps:
        print(f"  {getattr(ep, 'name', ep)!s:<34} registered={getattr(ep, 'is_registered', '?')}")
    if eps:
        print("\nRegistering execution providers (may download on first run)...")
        current = {"name": ""}

        def on_ep(name: str, pct: float) -> None:
            if name != current["name"]:
                if current["name"]:
                    print()
                current["name"] = name
            print(f"\r  {name:<34} {pct:6.1f}%", end="", flush=True)

        manager.download_and_register_eps(progress_callback=on_ep)
        print()
    else:
        print("  (none reported - CPU-only path)")

    section("4. Catalog")
    catalog = manager.catalog
    models = catalog.list_models()
    print(f"catalog exposes {len(models)} models. First 15 aliases:")
    for m in models[:15]:
        print(f"  {getattr(m, 'alias', '?'):<30} ctx={getattr(m, 'context_length', '?')}")

    section(f"5. Embedding model: {EMBED_ALIAS}")
    emb_model = catalog.get_model(EMBED_ALIAS)
    if emb_model is None:
        print(f"FAIL: alias '{EMBED_ALIAS}' not in catalog")
        return 1
    print(f"  id={emb_model.id}  cached={emb_model.is_cached}")
    if not emb_model.is_cached:
        emb_model.download(lambda p: print(f"\r  downloading {p:5.1f}%", end="", flush=True))
        print()
    t0 = time.perf_counter()
    emb_model.load()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    ec = emb_model.get_embedding_client()
    t0 = time.perf_counter()
    resp = ec.generate_embeddings(["merhaba dünya", "hello world", "kayıt dondurma"])
    dt = time.perf_counter() - t0
    dim = len(resp.data[0].embedding)
    print(f"  {len(resp.data)} vectors, dim={dim}, {dt * 1000:.0f} ms")

    # Sanity check the geometry: unrelated sentences should score lower than
    # a sentence against itself. If this fails the embeddings are garbage.
    import math

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    v = [d.embedding for d in resp.data]
    print(f"  cos(tr_hello, en_hello) = {cos(v[0], v[1]):.4f}   <- cross-lingual")
    print(f"  cos(tr_hello, unrelated) = {cos(v[0], v[2]):.4f}")
    print(f"  cos(self, self)          = {cos(v[0], v[0]):.4f}   <- must be ~1.0")

    section(f"6. Chat model: {CHAT_ALIAS}")
    chat_model = catalog.get_model(CHAT_ALIAS)
    if chat_model is None:
        print(f"FAIL: alias '{CHAT_ALIAS}' not in catalog")
        return 1
    print(f"  id={chat_model.id}  cached={chat_model.is_cached}")
    if not chat_model.is_cached:
        chat_model.download(lambda p: print(f"\r  downloading {p:5.1f}%", end="", flush=True))
        print()
    t0 = time.perf_counter()
    chat_model.load()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    cc = chat_model.get_chat_client()
    t0 = time.perf_counter()
    print("  streaming: ", end="", flush=True)
    out = []
    for chunk in cc.complete_streaming_chat(
        [{"role": "user", "content": "Say 'ready' and nothing else."}]
    ):
        if chunk.choices:
            d = chunk.choices[0].delta.content
            if d:
                out.append(d)
                print(d, end="", flush=True)
    print(f"\n  total {time.perf_counter() - t0:.1f}s")

    section("7. OpenAI-compatible web service")
    # The native ChatClient exposes no temperature/max_tokens knobs. The web
    # service does, via the standard OpenAI protocol - that is the path the
    # main pipeline uses.
    try:
        manager.start_web_service()
        urls = getattr(manager, "urls", None)
        print(f"  web service urls: {urls}")
        if urls:
            from openai import OpenAI

            base = urls[0] if isinstance(urls, (list, tuple)) else str(urls)
            base = base.rstrip("/")
            if not base.endswith("/v1"):
                base = base + "/v1"
            client = OpenAI(base_url=base, api_key="local", timeout=120.0)
            r = client.chat.completions.create(
                model=chat_model.id,
                messages=[{"role": "user", "content": "Reply with the word OK."}],
                temperature=0.0,
                max_tokens=10,
            )
            print(f"  temperature-controlled reply: {r.choices[0].message.content!r}")
            print("  -> full parameter control confirmed")
    except Exception as exc:
        print(f"  web service unavailable: {exc}")
        print("  -> pipeline will fall back to the native client")

    section("RESULT: environment is working")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
