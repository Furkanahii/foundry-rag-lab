"""Which local chat model is actually usable for grounded Turkish answers?

The project plan suggests picking a small model "so students get quick
feedback". On an English corpus that advice is fine. On Turkish it is not: the
0.5B model collapses into repetition loops, which no amount of prompt
engineering fixes.

This script turns model choice into a measurement instead of a guess. For each
candidate it runs the same grounded-QA prompts and reports:

  * load time and answer latency (the cost)
  * a repetition score - the fraction of generated 5-grams that are duplicates
    (the failure mode we actually observed)
  * whether the answer contains the key fact from the context (crude grounding
    check via required keywords)

Repetition score is the interesting one. Degenerate decoding shows up as
n-gram reuse long before a human calls the text "bad", so it detects the
failure automatically across dozens of runs.

Usage:  PYTHONPATH=src python scripts/compare_chat_models.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

sys.path.insert(0, "src")

from frag.config import RuntimeConfig  # noqa: E402
from frag.runtime.foundry import FoundryRuntime  # noqa: E402

CANDIDATES = [
    "qwen2.5-0.5b",
    "qwen2.5-1.5b",
    "phi-3.5-mini",
    "qwen2.5-7b",
]

CONTEXT = (
    "[1] Kayıt dondurma başvurusu, en geç ders ekleme-bırakma döneminin "
    "sonuna kadar öğrenci işleri müdürlüğüne yapılır.\n"
    "[2] Kayıt dondurma süresi en fazla iki yarıyıldır ve bu süre azami "
    "öğrenim süresine dahil edilmez.\n"
    "[3] Mezuniyet için gereken toplam kredi 240 AKTS'dir."
)

CASES = [
    {
        "question": "Kayıt dondurma başvurusu en geç ne zaman yapılır?",
        "must_contain": ["ders ekleme", "bırakma"],
    },
    {
        "question": "Kayıt dondurma en fazla kaç yarıyıl olabilir?",
        "must_contain": ["iki", "2"],
    },
    {
        "question": "Yurt başvurusu nasıl yapılır?",  # not in context
        "must_contain": [],  # correct behaviour is refusal
        "expect_refusal": True,
    },
]

SYSTEM = (
    "Aşağıdaki bağlamı kullanarak soruyu Türkçe yanıtla. "
    "Yanıtında kullandığın kaynağı [1], [2] gibi numaralarla belirt. "
    "Bağlamda cevap yoksa 'Bu bilgi verilen belgelerde yok.' de ve başka bir şey ekleme.\n\n"
    f"Bağlam:\n{CONTEXT}"
)

REFUSAL_MARKERS = ["belgelerde yok", "bilgi yok", "bulunmuyor", "mevcut değil"]


def repetition_score(text: str, n: int = 5) -> float:
    """Fraction of n-grams that are repeats. 0.0 = no repetition.

    Degenerate decoding loops produce the same n-gram many times. Measuring
    duplicate rate catches that automatically, well before the text reads as
    obviously broken to a human skimming output.
    """
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


@dataclass
class Result:
    model: str
    load_s: float
    mean_latency_ms: float
    mean_repetition: float
    grounding_hits: int
    refused_correctly: bool
    sample: str


def evaluate(alias: str) -> Result | None:
    cfg = RuntimeConfig(chat_model=alias)
    rt = FoundryRuntime(cfg)
    try:
        t0 = time.perf_counter()
        rt.ensure_model(alias)
        load_s = time.perf_counter() - t0
    except Exception as exc:
        print(f"  !! {alias}: could not load ({exc})")
        return None

    latencies: list[float] = []
    repetitions: list[float] = []
    hits = 0
    refused_correctly = False
    sample = ""

    for case in CASES:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": case["question"]},
        ]
        t0 = time.perf_counter()
        try:
            answer = rt.chat(messages, temperature=0.1, max_tokens=250)
        except Exception as exc:
            print(f"  !! {alias}: generation failed ({exc})")
            rt.close()
            return None
        latencies.append((time.perf_counter() - t0) * 1000)

        low = answer.lower()
        repetitions.append(repetition_score(answer))

        if case.get("expect_refusal"):
            refused_correctly = any(m in low for m in REFUSAL_MARKERS)
        elif case["must_contain"]:
            if any(k.lower() in low for k in case["must_contain"]):
                hits += 1

        if not sample:
            sample = answer.strip().replace("\n", " ")[:150]

    rt.close()
    return Result(
        model=alias,
        load_s=load_s,
        mean_latency_ms=sum(latencies) / len(latencies),
        mean_repetition=sum(repetitions) / len(repetitions),
        grounding_hits=hits,
        refused_correctly=refused_correctly,
        sample=sample,
    )


def main() -> int:
    results: list[Result] = []
    for alias in CANDIDATES:
        print(f"\n### {alias}")
        r = evaluate(alias)
        if r is None:
            continue
        results.append(r)
        print(f"  load {r.load_s:6.1f}s | latency {r.mean_latency_ms:7.0f}ms | "
              f"repetition {r.mean_repetition:.3f} | grounded {r.grounding_hits}/2 | "
              f"refusal {'OK' if r.refused_correctly else 'FAIL'}")
        print(f"  sample: {r.sample}")

    print("\n" + "=" * 92)
    print(f"{'model':<18}{'load(s)':>9}{'latency(ms)':>13}{'repetition':>12}"
          f"{'grounded':>10}{'refusal':>9}")
    print("=" * 92)
    for r in results:
        print(f"{r.model:<18}{r.load_s:>9.1f}{r.mean_latency_ms:>13.0f}"
              f"{r.mean_repetition:>12.3f}{r.grounding_hits:>8}/2"
              f"{'OK' if r.refused_correctly else 'FAIL':>9}")
    print("=" * 92)
    print("\nrepetition > 0.15 indicates degenerate decoding - the model is unusable")
    print("regardless of how fast it is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
