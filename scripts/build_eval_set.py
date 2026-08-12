"""Generate the evaluation set from the indexed corpus, once, and save it.

Run this before any benchmark. The set must stay fixed afterwards: regenerating
between two configurations would change the questions, and the comparison would
then measure the eval set rather than the system.

Usage:
    PYTHONPATH=src python scripts/build_eval_set.py [--n 40] [--paraphrase]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")

from frag.config import RunConfig  # noqa: E402
from frag.evaluation.qagen import (  # noqa: E402
    build_unanswerable,
    eval_set_stats,
    generate_questions,
    paraphrase_questions,
    save_eval_set,
)
from frag.evaluation.stats import required_sample_size  # noqa: E402
from frag.index.store import HybridStore  # noqa: E402
from frag.runtime.foundry import FoundryRuntime  # noqa: E402

DEFAULT_INDEX = "data/index/bogazici.db"
DEFAULT_OUT = "data/eval/eval_set.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40,
                        help="answerable questions to generate")
    parser.add_argument("--paraphrase", action="store_true",
                        help="also produce a colloquial variant of each question")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--model", default=None,
                        help="override the chat model used for generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    cfg = RunConfig()
    if args.model:
        cfg.runtime.chat_model = args.model

    print(f"model: {cfg.runtime.chat_model}")
    runtime = FoundryRuntime(cfg.runtime)
    store = HybridStore(args.index)
    print(f"index: {store.count_chunks()} chunks\n")

    started = time.perf_counter()
    print(f"Generating {args.n} answerable questions...")
    questions = generate_questions(runtime, store, n_questions=args.n)

    if args.paraphrase and questions:
        print(f"\nParaphrasing {len(questions)} questions...")
        questions.extend(paraphrase_questions(runtime, questions))

    questions.extend(build_unanswerable())

    # An eval set is the reference point for every result already on disk.
    # Overwriting it in place would make the stored benchmarks uninterpretable -
    # their numbers would refer to questions no longer anywhere on the machine.
    out_path = Path(args.out)
    if out_path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = out_path.with_name(f"{out_path.stem}.{stamp}.bak.json")
        shutil.copy2(out_path, backup)
        print(f"\nprevious eval set backed up to {backup}")

    save_eval_set(questions, args.out)
    elapsed = time.perf_counter() - started

    print(f"\n=== eval set saved to {args.out} ({elapsed:.0f}s) ===")
    stats = eval_set_stats(questions)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Power: the reason this set was rebuilt. Reported here so an underpowered
    # set is caught now, not after a benchmark sweep has been run on it.
    n_generated = stats["variants"].get("generated", 0)
    print(f"\nPower with n={n_generated} per variant:")
    for d, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        need = required_sample_size(d)
        ok = "sufficient" if n_generated >= need else f"UNDERPOWERED (need {need})"
        print(f"  {label:<7} effect (d={d}): {ok}")

    overlap = stats["mean_gold_overlap"]
    if "generated" in overlap and "paraphrased" in overlap:
        drop = overlap["generated"] - overlap["paraphrased"]
        print(
            f"\nLexical overlap with gold chunk: generated {overlap['generated']:.3f} "
            f"-> paraphrased {overlap['paraphrased']:.3f} (drop {drop:+.3f})"
        )
        if drop < 0.02:
            print("  WARNING: the paraphrase pass barely changed wording. The two")
            print("  variants will not separate lexical from semantic retrieval.")

    print("\n--- sample ---")
    for q in questions[:5]:
        print(f"  [{q.variant}] {q.question}")
        if q.gold_chunk_ids:
            print(f"      gold: {q.gold_chunk_ids[0]}")

    store.close()
    runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
