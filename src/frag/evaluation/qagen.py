"""Building the evaluation set from the corpus itself.

Why not write the questions by hand: this project already learned that lesson
the expensive way. A hand-written question about "kayıt dondurma" returned
nothing useful, and the reason turned out to be that the phrase does not appear
anywhere in the corpus - the regulations say "izinli sayılma". The question was
testing the author's vocabulary, not the system.

Generating questions *from* chunks fixes that by construction, and gives us
something hand-labelling cannot: the gold chunk id is known exactly, because we
generated the question from that chunk. No annotation, no ambiguity about which
passage "should" have been retrieved.

## The circularity problem, and what we do about it

Questions generated from a chunk tend to reuse that chunk's wording, which
flatters lexical retrieval - BM25 looks great when the query is a paraphrase of
the document. Two mitigations:

  1. The prompt explicitly asks for the question a *student* would ask, in
     everyday language, not the regulation's own phrasing.
  2. `paraphrase_questions()` runs a second pass that rewrites questions into
     colloquial Turkish, and the harness reports metrics on both sets. A large
     gap between them is itself a finding: it quantifies how much of the
     measured retrieval quality is vocabulary overlap rather than understanding.

The mitigations are not taken on faith. `question_gold_overlap()` measures, for
every item, the Jaccard overlap between the question's lexical tokens and its
gold chunk's - computed with the *same* stemming the BM25 arm uses, so it is
literally the signal that arm scores on. The first benchmark run of this project
was discarded because that number was never checked: the generated questions
shared so much wording with their sources that BM25 reached 0.86 nDCG against
dense retrieval's 0.58, and the sweep was measuring the eval set's construction
rather than the pipeline. The number is now printed at build time and stored per
question, so the same mistake announces itself.

## Unanswerable questions

Half of abstention evaluation needs questions the corpus provably cannot answer.
These are generated separately, on plausible-but-absent topics, so that a system
which refuses everything and a system which answers everything both score badly.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..index.store import HybridStore
from ..ingest.normalize import lexical_tokens
from ..runtime.foundry import FoundryRuntime

logger = logging.getLogger(__name__)


@dataclass
class EvalQuestion:
    """One evaluation item with its ground truth."""

    question: str
    gold_chunk_ids: list[str]
    answerable: bool
    source_doc: str = ""
    gold_answer: str = ""
    variant: str = "generated"     # "generated" | "paraphrased" | "unanswerable"
    meta: dict[str, Any] = field(default_factory=dict)


_GEN_SYSTEM = """Sana bir üniversite yönetmeliğinden bir bölüm verilecek.

Görevin: Bu bölümün YANITLADIĞI, bir öğrencinin gerçekten soracağı TEK bir soru yaz.

Kurallar:
- Soru günlük Türkçe olsun, yönetmelik dilini kopyalama.
- Soru SADECE bu bölümdeki bilgiyle yanıtlanabilmeli.
- "Bu maddede", "yukarıdaki bölümde" gibi ifadeler kullanma; soru tek başına anlaşılmalı.
- SADECE soruyu yaz. Açıklama, numara, tırnak ekleme."""

# The first version of this prompt ("bir öğrencinin WhatsApp'ta soracağı gibi
# yeniden yaz") failed on measurement, not on inspection: 42 of 60 outputs came
# back token-identical to the input even after three retries at rising
# temperature. A model asked to reword a sentence that is already a grammatical
# question has no gradient to follow unless it is told *what* to change. The
# rewrite below therefore names the operation (swap the content words), forbids
# the failure mode explicitly, and shows it happening three times - few-shot
# examples do the work that adjectives like "günlük" could not.
_PARAPHRASE_SYSTEM = """Sana resmî dilde yazılmış bir soru verilecek. Onu, konuyu bilmeyen bir öğrencinin günlük konuşma diliyle soracağı hâle çevir.

Kurallar:
- Anlam birebir aynı kalsın.
- **Orijinal sorunun kelimelerini kullanma.** Her resmî terimi günlük karşılığıyla değiştir. Aynı cümleyi geri vermek yanlış cevaptır.
- Cümle kuruluşunu da değiştir; sadece kelime değiştirmek yetmez.
- SADECE yeni soruyu yaz. Açıklama, numara, tırnak ekleme.

Örnekler:

Soru: Öğrencinin kayıt yenileme işlemini hangi süre içinde tamamlaması gerekir?
Yeni soru: Derslere yazılmak için kaç günüm var?

Soru: Disiplin cezası alan bir öğrencinin itiraz hakkı var mıdır?
Yeni soru: Ceza yersem buna karşı çıkabilir miyim?

Soru: Lisansüstü programlarda azami öğrenim süresi ne kadardır?
Yeni soru: Yüksek lisans en fazla kaç yıl sürebiliyor?"""

_UNANSWERABLE_SYSTEM = """Bir üniversitenin öğrenci yönetmelikleri hakkında SORULABİLECEK ama bu konularda BİLGİ İÇERMEYEN sorular üret.

Konular yönetmelikle ilgili görünmeli ama şu alanlardan olmalı: yemekhane menüsü, otopark ücreti, spor salonu saatleri, kampüs wifi şifresi, mezuniyet töreni kıyafeti, staj bulma tavsiyeleri.

Her satıra bir soru yaz. Numara, tire, açıklama ekleme."""


def _clean_question(raw: str) -> str:
    """Strip the decorations small models add despite being told not to."""
    text = raw.strip()
    # Leading list markers and numbering.
    text = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", text)
    # Surrounding quotes.
    text = text.strip().strip('"').strip("'").strip()
    # Keep only the first line; models often continue with commentary.
    text = text.split("\n")[0].strip()
    # Drop a leading label like "Soru:".
    text = re.sub(r"^(soru|question)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _is_usable_question(text: str) -> bool:
    """Reject degenerate output before it pollutes the eval set.

    A bad eval question is worse than a missing one: it becomes permanent noise
    in every future comparison, and unlike a bad answer it never announces
    itself.
    """
    if len(text) < 15 or len(text) > 250:
        return False
    if "?" not in text:
        return False
    # References to "this section" mean the question is not self-contained.
    for phrase in ("bu maddede", "bu bölümde", "yukarıda", "aşağıda", "metinde"):
        if phrase in text.lower():
            return False
    return True


def question_gold_overlap(question: str, gold_text: str, stem: bool = True) -> float:
    """Fraction of a question's lexical tokens that also occur in its gold chunk.

    Deliberately computed with `lexical_tokens`, the same folding and stemming
    the BM25 arm applies at index and query time. That makes this number the
    lexical signal itself rather than a proxy for it: an item scoring 0.5 has
    half its stemmed vocabulary sitting in the passage it is supposed to find,
    and BM25 will locate it without any understanding of the question.

    ## Why containment and not Jaccard

    Jaccard was tried first and had to be replaced, which is worth recording
    because the failure is not obvious from the formula. A question holds ~10
    stemmed tokens; a 900-character chunk holds ~120. Their union is therefore
    dominated by the chunk, and the ratio is squeezed into a narrow band near
    the bottom of the scale no matter how much copying happened. Measured on
    this corpus's 60 generated questions, Jaccard reported 0.101 where
    containment reported 0.448 - and the paraphrased set moved by 0.007 under
    Jaccard versus 0.041 under containment. The first reading suggests nothing
    is happening; the second shows a real effect four times larger than the
    noise floor.

    Containment also matches what BM25 actually does. A query term contributes
    score when it appears in the document; terms in the document that are absent
    from the query contribute nothing. Dividing by the union charges the metric
    for chunk length, which is a property of chunking, not of question quality.
    """
    q = set(lexical_tokens(question, stem=stem))
    g = set(lexical_tokens(gold_text, stem=stem))
    if not q or not g:
        return 0.0
    return len(q & g) / len(q)


def generate_questions(
    runtime: FoundryRuntime,
    store: HybridStore,
    n_questions: int = 40,
    min_chunk_chars: int = 300,
    seed: int = 20260801,
    progress: bool = True,
    max_attempts_factor: float = 2.5,
) -> list[EvalQuestion]:
    """Generate answerable questions from randomly sampled chunks.

    `n_questions` is a target of *usable* questions, not of attempts. The
    distinction decides whether the resulting benchmark is powered: rejection
    rates vary by model (phi-3.5-mini produced malformed Turkish questions where
    qwen2.5-7b does not), so sampling exactly n chunks silently yields whatever
    survives - the discarded first run of this project asked for 30 and
    benchmarked on 15, far below the 34 a medium effect needs. We therefore keep
    drawing from the shuffled pool until the target is met, capped at
    `max_attempts_factor * n` model calls so a badly behaved model fails loudly
    with a short set rather than looping over the whole corpus.

    Chunks shorter than `min_chunk_chars` are skipped: they rarely contain a
    complete rule, so a question generated from one tends to be unanswerable
    even from its own source - which would put a permanently unreachable item
    into the gold set.
    """
    rows = store.conn.execute(
        "SELECT chunk_id, doc_id, text FROM chunks WHERE length(text) >= ?",
        (min_chunk_chars,),
    ).fetchall()
    if not rows:
        raise ValueError("No chunks long enough to generate questions from")

    rng = random.Random(seed)
    # Shuffle the whole eligible pool once, then walk it. Sampling n up front
    # would leave no replacements for rejected items; walking a fixed shuffled
    # order keeps the draw reproducible from `seed` regardless of how many
    # rejections happen along the way.
    pool = list(rows)
    rng.shuffle(pool)
    max_attempts = min(len(pool), int(n_questions * max_attempts_factor))

    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    attempts = 0

    for row in pool:
        if len(questions) >= n_questions or attempts >= max_attempts:
            break
        attempts += 1
        passage = re.sub(r"\[\[page:\d+\]\]\s*", "", row["text"]).strip()[:1500]
        try:
            raw = runtime.chat(
                [
                    {"role": "system", "content": _GEN_SYSTEM},
                    {"role": "user", "content": passage},
                ],
                temperature=0.7,   # some diversity; 0.0 produces near-identical phrasing
                max_tokens=90,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Question generation failed for %s: %s", row["chunk_id"], exc)
            continue

        text = _clean_question(raw)
        if not _is_usable_question(text):
            logger.debug("Rejected generated question: %r", text[:80])
            continue

        # Two chunks covering the same rule can yield the same question with
        # different gold ids. Keeping both would make one of them unanswerable
        # by construction: whichever chunk is not its label counts as a miss no
        # matter which one retrieval returns.
        key = " ".join(lexical_tokens(text))
        if key in seen:
            logger.debug("Rejected duplicate question: %r", text[:80])
            continue
        seen.add(key)

        questions.append(
            EvalQuestion(
                question=text,
                gold_chunk_ids=[row["chunk_id"]],
                answerable=True,
                source_doc=row["doc_id"],
                gold_answer=passage[:400],
                variant="generated",
                meta={"gold_overlap": round(question_gold_overlap(text, passage), 4)},
            )
        )
        if progress and len(questions) % 5 == 0:
            print(f"  generated {len(questions)}/{n_questions} "
                  f"({attempts} attempts)...", flush=True)

    if len(questions) < n_questions:
        logger.warning(
            "Asked for %d questions, got %d usable in %d attempts. The benchmark "
            "will be underpowered; try a stronger generation model.",
            n_questions, len(questions), attempts,
        )
    logger.info("Generated %d usable questions from %d attempts", len(questions), attempts)
    return questions


def paraphrase_questions(
    runtime: FoundryRuntime,
    questions: Sequence[EvalQuestion],
    progress: bool = True,
    max_retries: int = 2,
    echo_threshold: float = 0.8,
) -> list[EvalQuestion]:
    """Rewrite questions colloquially, keeping the same gold chunks.

    The gold labels transfer unchanged because paraphrasing preserves meaning -
    the same passage still answers it. Comparing metrics on the original versus
    the paraphrased set isolates how much retrieval depends on shared wording.

    Two properties are enforced here, both in service of that comparison.

    **The output stays one-to-one with the input.** A dropped item would make
    the two sets different question populations, and the original-vs-paraphrased
    difference could then be explained by which questions were dropped rather
    than by wording. Keeping the alignment makes the two runs *paired*, so the
    difference can be tested per question with the same machinery used for
    configurations. An item that survives no attempt is kept in its original
    wording and flagged, never silently removed.

    **Echoes are retried.** A small model asked to reword often returns the
    input nearly verbatim. Such an item contributes a paraphrased score that is
    really an original score, biasing the measured gap towards zero - the
    direction that would let us wrongly conclude the lexical advantage is
    harmless. Retries use a higher temperature because a greedy decode has
    already demonstrated it lands on the input.
    """
    out: list[EvalQuestion] = []
    echoes = 0

    for i, q in enumerate(questions, start=1):
        original_tokens = set(lexical_tokens(q.question))
        best: str | None = None
        reworded = False

        for attempt in range(max_retries + 1):
            try:
                raw = runtime.chat(
                    [
                        {"role": "system", "content": _PARAPHRASE_SYSTEM},
                        # Framed as an instruction rather than passed bare: a
                        # lone question in the user turn invites the model to
                        # answer it or repeat it, which is what the first run
                        # did 42 times out of 60.
                        {"role": "user",
                         "content": f"Soru: {q.question}\nYeni soru:"},
                    ],
                    temperature=0.7 + 0.15 * attempt,
                    max_tokens=90,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Paraphrase failed: %s", exc)
                continue

            text = _clean_question(raw)
            if not _is_usable_question(text):
                continue

            tokens = set(lexical_tokens(text))
            similarity = (
                len(original_tokens & tokens) / len(original_tokens | tokens)
                if original_tokens | tokens else 1.0
            )
            best = text
            if similarity <= echo_threshold:
                reworded = True
                break
            logger.debug("Echo (%.2f) on attempt %d: %r", similarity, attempt + 1, text[:60])

        # `best` is the last usable candidate, or None if nothing validated -
        # in which case we fall back to the original wording to keep the sets
        # aligned. Either way the item counts as un-reworded.
        text = best if best is not None else q.question
        if not reworded:
            echoes += 1

        out.append(
            EvalQuestion(
                question=text,
                gold_chunk_ids=list(q.gold_chunk_ids),
                answerable=True,
                source_doc=q.source_doc,
                gold_answer=q.gold_answer,
                variant="paraphrased",
                meta={
                    "original": q.question,
                    "gold_overlap": round(
                        question_gold_overlap(text, q.gold_answer), 4
                    ),
                    "echo": not reworded,
                },
            )
        )
        if progress and i % 5 == 0:
            print(f"  paraphrased {i}/{len(questions)}...", flush=True)

    if echoes:
        logger.warning(
            "%d/%d paraphrases stayed close to the original after retries; "
            "the measured wording gap is a lower bound.", echoes, len(questions),
        )
    return out


# Fixed unanswerable questions. Deliberately hand-written rather than generated:
# we must be *certain* the corpus cannot answer them, and a generated question
# might accidentally land on a topic the regulations do cover - silently
# corrupting the abstention metric.
FIXED_UNANSWERABLE = [
    "Yemekhanede bugün ne var",
    "Kampüs wifi şifresi nedir",
    "Otopark ücreti aylık kaç lira",
    "Spor salonu hafta sonu kaça kadar açık",
    "Mezuniyet töreninde ne giymeliyim",
    "Staj bulmak için hangi siteye bakmalıyım",
    "Kütüphanede kaç kişilik çalışma odası var",
    "Servis otobüsü saatleri nedir",
    "Mars'ta yaşam var mı",
    "Python'da liste nasıl sıralanır",
]


def build_unanswerable(n: int | None = None) -> list[EvalQuestion]:
    items = FIXED_UNANSWERABLE if n is None else FIXED_UNANSWERABLE[:n]
    return [
        EvalQuestion(question=q, gold_chunk_ids=[], answerable=False,
                     variant="unanswerable")
        for q in items
    ]


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def save_eval_set(questions: Sequence[EvalQuestion], path: str | Path) -> None:
    """Persist the eval set.

    Generating it costs many model calls, and - more importantly - it must stay
    *fixed* across configurations. Regenerating between two runs would change
    the questions and the comparison would measure the eval set, not the system.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(q) for q in questions], fh, ensure_ascii=False, indent=2)
    logger.info("Saved %d questions to %s", len(questions), path)


def load_eval_set(path: str | Path) -> list[EvalQuestion]:
    with open(path, "r", encoding="utf-8") as fh:
        return [EvalQuestion(**item) for item in json.load(fh)]


def eval_set_stats(questions: Sequence[EvalQuestion]) -> dict[str, Any]:
    """Descriptive statistics, including the ones that decide validity.

    `mean_gold_overlap` per variant is the number to read before trusting any
    benchmark built on this set. If the generated and paraphrased variants show
    the same overlap, the paraphrase pass did nothing and the two runs will not
    separate lexical retrieval from semantic retrieval.
    """
    from collections import Counter

    variants = Counter(q.variant for q in questions)
    docs = Counter(q.source_doc for q in questions if q.source_doc)

    overlap_by_variant: dict[str, float] = {}
    for variant in variants:
        values = [
            q.meta["gold_overlap"] for q in questions
            if q.variant == variant and "gold_overlap" in q.meta
        ]
        if values:
            overlap_by_variant[variant] = round(sum(values) / len(values), 4)

    return {
        "n_total": len(questions),
        "n_answerable": sum(1 for q in questions if q.answerable),
        "n_unanswerable": sum(1 for q in questions if not q.answerable),
        "variants": dict(variants),
        "documents_covered": len(docs),
        "mean_gold_overlap": overlap_by_variant,
        "echoed_paraphrases": sum(
            1 for q in questions if q.variant == "paraphrased" and q.meta.get("echo")
        ),
        "mean_question_chars": (
            round(sum(len(q.question) for q in questions) / len(questions), 1)
            if questions else 0.0
        ),
    }
