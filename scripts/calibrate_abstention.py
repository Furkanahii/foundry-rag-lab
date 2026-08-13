"""Pick the abstention threshold from data instead of guessing it.

`GenerationConfig.abstention_threshold` shipped as 0.0, which is not a threshold
but a placeholder meaning "never refuse". Refusing is the single most important
safety behaviour this system has - a fabricated regulation is worse than no
answer - so the number has to come from somewhere defensible.

## Which signal is being calibrated - and why it is not the obvious one

The obvious candidate is the pipeline's fused score for the top hit, and it is
the wrong one for a structural reason worth stating before any number appears.
`weighted_fusion` min-max normalises each arm *within the candidates retrieved
for that query* (see `index/fusion.py`). The best candidate therefore scores 1.0
in its arm no matter how good it actually is: the score says "this is the best
of what I found", never "what I found is any good". A quantity that cannot
express "the corpus has nothing" cannot drive a decision about whether the
corpus has anything. Measured here, it reaches AUC 0.727 and hands one
unanswerable question a perfect 1.000.

The raw scores survive on `FusedHit.dense_score` and `.lexical_score` and are
not normalised, so this script scores all three candidate signals and calibrates
whichever separates the classes best:

  * fused top score - the tempting default, kept in the comparison so its
    weakness is demonstrated rather than asserted
  * best raw dense cosine - absolute, comparable across queries
  * best raw BM25 score - absolute, but scaled by query length and IDF

Caveat on the raw signals: they are read off the top_k hits that survived
fusion, not off the full candidate lists, so a query whose best dense candidate
was ranked out of the final five contributes a slightly pessimistic value. With
alpha=0.5 the top dense candidate carries 0.5 of fused score on its own and is
almost always retained.

Retrieval-only by design: the score exists before generation runs, so 130
questions cost seconds rather than the two hours generation would need. The
threshold is a property of retrieval; the generator only consumes it.

## What the output means

  * **F1-optimal threshold** balances the two errors. Reported with its full
    confusion matrix so the trade can be read directly rather than trusted.
  * **A recall-weighted alternative** is also printed, because F1 treats the
    two errors as equally bad and this system should not. Refusing an
    answerable question annoys a user; answering an unanswerable one invents a
    rule about someone's education. The deployment default follows the second
    reading, and the script prints both so the choice is visible.
  * **ECE / MCE / Brier** say whether the score behaves like a probability at
    all. A threshold can work even when calibration is poor - it only needs
    monotonicity - but a badly calibrated score cannot be shown to a user as a
    confidence percentage.

Usage:
    PYTHONPATH=src python scripts/calibrate_abstention.py
    PYTHONPATH=src python scripts/calibrate_abstention.py --variant paraphrased
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from frag.config import RunConfig  # noqa: E402
from frag.evaluation.qagen import load_eval_set  # noqa: E402
from frag.evaluation.stats import (  # noqa: E402
    expected_calibration_error,
    find_best_threshold,
)
from frag.pipeline import RagPipeline  # noqa: E402
from frag.runtime.foundry import FoundryRuntime  # noqa: E402

DEFAULT_INDEX = "data/index/bogazici.db"
DEFAULT_EVAL = "data/eval/eval_set.json"
DEFAULT_OUT = "data/results/abstention_calibration.json"


def confusion_line(stats: dict[str, float]) -> str:
    return (
        f"    eşik      {stats['threshold']:.6f}\n"
        f"    F1        {stats['f1']:.4f}   doğruluk {stats['accuracy']:.4f}\n"
        f"    kesinlik  {stats['precision']:.4f}   duyarlılık {stats['recall']:.4f}\n"
        f"    TP {int(stats['tp'])}  FP {int(stats['fp'])}  "
        f"FN {int(stats['fn'])}  TN {int(stats['tn'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--eval", default=DEFAULT_EVAL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--variant", default="paraphrased",
                        help="which wording variant supplies the positives "
                             "(default: paraphrased, the realistic one)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    questions = load_eval_set(args.eval)
    positives = [q for q in questions if q.answerable and q.variant == args.variant]
    negatives = [q for q in questions if not q.answerable]
    if not positives or not negatives:
        print(f"Need both answerable ({args.variant}) and unanswerable questions.")
        return 1

    # The positives come from one wording variant only. Pooling both would count
    # each underlying item twice and make the negative class look half as
    # frequent as it is, which shifts the threshold for a purely bookkeeping
    # reason. The paraphrased set is the default because the threshold has to
    # hold for questions as users write them.
    print(f"positives: {len(positives)} ({args.variant})   negatives: {len(negatives)}")

    cfg = RunConfig()
    print(f"config: fusion={cfg.retrieval.fusion} alpha={cfg.retrieval.alpha} "
          f"top_k={cfg.retrieval.top_k}\n")

    runtime = FoundryRuntime(cfg.runtime)
    pipeline = RagPipeline(cfg, args.index, runtime=runtime)

    signals: dict[str, list[float]] = {"fused": [], "dense_raw": [], "lexical_raw": []}
    labels: list[bool] = []
    started = time.perf_counter()

    for i, q in enumerate(positives + negatives, start=1):
        result = pipeline.retrieve(q.question)
        dense_values = [h.dense_score for h in result.hits if h.dense_score is not None]
        lexical_values = [
            h.lexical_score for h in result.hits if h.lexical_score is not None
        ]
        signals["fused"].append(float(result.top_score))
        signals["dense_raw"].append(float(max(dense_values)) if dense_values else 0.0)
        signals["lexical_raw"].append(
            float(max(lexical_values)) if lexical_values else 0.0
        )
        labels.append(q.answerable)
        if i % 25 == 0:
            print(f"  {i}/{len(positives) + len(negatives)}...", flush=True)

    elapsed = time.perf_counter() - started
    y = np.asarray(labels, dtype=bool)
    print(f"\nretrieved {len(labels)} queries in {elapsed:.1f}s")

    def auc_of(values: list[float]) -> float:
        """Threshold-free separability: P(random positive > random negative).

        Computed as the Mann-Whitney statistic. It answers "does this score
        carry the signal at all" without depending on where a cut is placed,
        which is the question to settle before choosing between signals.
        """
        arr = np.asarray(values)
        pos, neg = arr[y], arr[~y]
        if not len(pos) or not len(neg):
            return float("nan")
        wins = float(np.sum(pos[:, None] > neg[None, :]))
        ties = float(np.sum(pos[:, None] == neg[None, :]))
        return (wins + 0.5 * ties) / (len(pos) * len(neg))

    print("\n--- aday sinyaller (AUC: 0.5 = ayırt edemiyor, 1.0 = kusursuz) ---")
    aucs = {name: auc_of(values) for name, values in signals.items()}
    for name, value in sorted(aucs.items(), key=lambda kv: -kv[1]):
        arr = np.asarray(signals[name])
        print(f"  {name:<12} AUC {value:.4f}   "
              f"cevaplanabilir ort {arr[y].mean():.4f}  "
              f"cevaplanamaz ort {arr[~y].mean():.4f}  "
              f"cevaplanamaz maks {arr[~y].max():.4f}")

    best_signal = max(aucs, key=lambda k: aucs[k])
    scores = signals[best_signal]
    s = np.asarray(scores)
    auc = aucs[best_signal]
    print(f"\n  seçilen sinyal: {best_signal} (AUC {auc:.4f})")
    if best_signal != "fused":
        print("  Füzyon skoru sorgu içinde normalize edildiği için mutlak bir")
        print("  güven ifade edemiyor; ham skor bu yüzden daha iyi ayırıyor.")

    print("\n--- F1-optimal eşik ---")
    f1_threshold, f1_stats = find_best_threshold(scores, labels, objective="f1")
    print(confusion_line(f1_stats))

    print("\n--- doğruluk-optimal eşik ---")
    acc_threshold, acc_stats = find_best_threshold(scores, labels, objective="accuracy")
    print(confusion_line(acc_stats))

    # Safety-weighted choice: the strictest threshold that still answers at
    # least this fraction of answerable questions. Fabricating a regulation is
    # the error we are willing to pay for, so we buy fewer of them with
    # unnecessary refusals - but not at the cost of a system that refuses
    # everything, which would score well on the negatives and be useless.
    min_recall = 0.90
    candidates = np.unique(s)
    safe_threshold = float(candidates[0])
    safe_stats = f1_stats
    for threshold in candidates:
        answered = s >= threshold
        tp = int(np.sum(answered & y))
        fn = int(np.sum(~answered & y))
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if recall < min_recall:
            break
        fp = int(np.sum(answered & ~y))
        tn = int(np.sum(~answered & ~y))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        safe_threshold = float(threshold)
        safe_stats = {
            "threshold": safe_threshold,
            "f1": (2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0),
            "accuracy": (tp + tn) / len(s),
            "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    print(f"\n--- güvenlik ağırlıklı eşik (duyarlılık >= {min_recall:.0%}) ---")
    print(confusion_line(safe_stats))

    # ECE compares a score against an accuracy, so it only means anything when
    # the score already lives on [0, 1]. A BM25 score does not: run it through
    # the bins and every value lands above the last edge, which reports a
    # flattering ECE of 0.0000 next to a Brier score of 152 - two numbers that
    # cannot both be true, and a good illustration of why a metric should be
    # refused rather than reported when its precondition fails.
    print("\n--- kalibrasyon ---")
    bounded = bool(s.min() >= 0.0 and s.max() <= 1.0)
    if bounded:
        calibration = expected_calibration_error(scores, labels, n_bins=10)
        print(f"    {calibration}")
        print("    ECE, skorun bir olasılık gibi okunup okunamayacağını söyler.")
        print("    Eşik için monotonluk yeterli; kullanıcıya yüzde göstermek için değil.")
    else:
        calibration = None
        print(f"    ATLANDI: '{best_signal}' [0,1] aralığında değil "
              f"(gözlenen {s.min():.3f}–{s.max():.3f}).")
        print("    ECE bir skoru doğrulukla karşılaştırır; sınırsız bir skorda")
        print("    tanımsızdır. Eşik yine de geçerli, çünkü eşik yalnızca")
        print("    monotonluk gerektirir — skorun olasılık olmasını değil.")

    payload = {
        "variant": args.variant,
        "n_positive": int(y.sum()),
        "n_negative": int((~y).sum()),
        "config": cfg.to_dict(),
        "signal": best_signal,
        "auc_by_signal": aucs,
        "auc": auc,
        "f1_optimal": f1_stats,
        "accuracy_optimal": acc_stats,
        "safety_weighted": safe_stats,
        "min_recall": min_recall,
        "calibration": (
            {"ece": calibration.ece, "mce": calibration.mce,
             "brier": calibration.brier}
            if calibration is not None else
            {"skipped": "signal is not bounded to [0, 1]"}
        ),
        "scores": scores,
        "all_signals": signals,
        "labels": [bool(v) for v in labels],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved to {out}")

    print(
        f"\nÖnerilen: generation.abstention_threshold = {safe_stats['threshold']:.6f}\n"
        f"Bu eşik {len(negatives)} negatif örnek üzerinde kalibre edildi; küçük bir\n"
        f"negatif sınıf, eşiğin kendisini gürültülü yapar. Negatif sayısını\n"
        f"artırmak, bu sayıyı iyileştirmenin en ucuz yolu."
    )

    pipeline.close()
    runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
