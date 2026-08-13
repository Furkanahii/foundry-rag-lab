"""Measure whether the paraphrases still mean what their originals meant.

The paraphrase pass exists to break vocabulary overlap without changing the
question. Only the first half of that is enforced by the generator: the echo
retry checks that the wording *changed*, and nothing checks that the meaning
*stayed*. That gap matters, because the two failures look identical in the
benchmark. A configuration scoring lower on the paraphrased set could be losing
its lexical crutch - the effect we want to measure - or it could be answering a
question whose gold label is now wrong.

Inspection of the first rebuilt set found real drift: "akademik yarıyıl
kayıtları tamamlanmamış" came back as "yarıyıl notlarını vermediği", which is
registration turning into grades, and one item flipped a positive question to a
negative one. Eyeballing 60 items does not scale and does not produce a number,
so this script measures it with the embedding model the pipeline already loads.

Cosine similarity between the two questions is the right instrument here, and
notably *not* the right instrument for the lexical overlap in `qagen.py` - there
we deliberately measure what BM25 sees. Here we want the opposite: a measure
that ignores wording and responds to meaning.

Usage:
    PYTHONPATH=src python scripts/check_paraphrase_fidelity.py
    PYTHONPATH=src python scripts/check_paraphrase_fidelity.py --annotate
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from frag.config import RunConfig  # noqa: E402
from frag.evaluation.qagen import load_eval_set, save_eval_set  # noqa: E402
from frag.runtime.foundry import FoundryRuntime  # noqa: E402

DEFAULT_EVAL = "data/eval/eval_set.json"

# Below this cosine the pair is reported as suspect. Not a hard filter: the
# threshold is a reading aid, and the honest response to a drifted item is to
# report how many there are, not to quietly delete the inconvenient ones.
SUSPECT_BELOW = 0.80


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=DEFAULT_EVAL)
    parser.add_argument("--threshold", type=float, default=SUSPECT_BELOW)
    parser.add_argument("--annotate", action="store_true",
                        help="write the similarity into each item's meta and save")
    parser.add_argument("--show", type=int, default=10,
                        help="how many of the worst pairs to print")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    questions = load_eval_set(args.eval)
    pairs = [q for q in questions if q.variant == "paraphrased" and q.meta.get("original")]
    if not pairs:
        print(f"No paraphrased questions with originals in {args.eval}")
        return 1

    cfg = RunConfig()
    runtime = FoundryRuntime(cfg.runtime)

    # Embedded in one batch each: the runtime normalises to unit length, so the
    # dot product is the cosine.
    originals = runtime.embed([q.meta["original"] for q in pairs])
    rewrites = runtime.embed([q.question for q in pairs])
    similarity = np.sum(originals * rewrites, axis=1)

    print(f"\n=== paraphrase fidelity ({len(pairs)} pairs) ===")
    print(f"  mean   {similarity.mean():.4f}")
    print(f"  median {np.median(similarity):.4f}")
    print(f"  min    {similarity.min():.4f}")
    print(f"  p10    {np.percentile(similarity, 10):.4f}")

    suspect = int((similarity < args.threshold).sum())
    print(f"\n  below {args.threshold}: {suspect}/{len(pairs)} "
          f"({100 * suspect / len(pairs):.0f}%)")

    order = np.argsort(similarity)
    print(f"\n--- {min(args.show, len(pairs))} lowest-similarity pairs ---")
    for idx in order[: args.show]:
        q = pairs[idx]
        print(f"\n  cos={similarity[idx]:.3f}")
        print(f"  ORJ: {q.meta['original']}")
        print(f"  PAR: {q.question}")

    if args.annotate:
        for q, value in zip(pairs, similarity):
            q.meta["semantic_similarity"] = round(float(value), 4)
        save_eval_set(questions, args.eval)
        print(f"\nAnnotated {len(pairs)} items in {args.eval}")

    print(
        "\nReading: a low-similarity pair means the paraphrase drifted, so its\n"
        "gold label may no longer be correct. Those items depress the paraphrased\n"
        "variant for a reason unrelated to vocabulary, which makes the measured\n"
        "wording gap an *upper* bound on the true lexical advantage."
    )

    runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
