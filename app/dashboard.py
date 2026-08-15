"""Streamlit lab dashboard.

Two things this is *not*: a chat toy, and a wrapper that hides the pipeline.
The point is the opposite - every number the system used to pick its answer is
on screen. You can see that a chunk arrived at rank 2 because BM25 ranked it
first while dense retrieval never returned it, and you can flip fusion to
dense-only and watch it disappear.

That transparency is what makes it a demo of a *system* rather than a demo of a
language model.

Run:  PYTHONPATH=src ./.venv/bin/streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frag.config import RunConfig  # noqa: E402
from frag.pipeline import RagPipeline  # noqa: E402
from frag.runtime.foundry import FoundryRuntime  # noqa: E402

INDEX_PATH = ROOT / "data" / "index" / "bogazici.db"

st.set_page_config(page_title="Foundry RAG Lab", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Foundry Local başlatılıyor...")
def get_runtime(chat_model: str, embedding_model: str) -> FoundryRuntime:
    """One runtime per model pair, cached across reruns.

    Streamlit re-executes the whole script on every interaction. Without this
    cache each keystroke would reload multi-gigabyte models, which takes minutes
    - `cache_resource` is what makes the UI usable at all.
    """
    cfg = RunConfig()
    cfg.runtime.chat_model = chat_model
    cfg.runtime.embedding_model = embedding_model
    return FoundryRuntime(cfg.runtime)


def build_config(**overrides) -> RunConfig:
    cfg = RunConfig()
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        if field:
            setattr(getattr(cfg, section), field, value)
    return cfg


# ---------------------------------------------------------------------------
# sidebar: the experiment controls
# ---------------------------------------------------------------------------

st.sidebar.title("Yapılandırma")

chat_model = st.sidebar.selectbox(
    "Üretim modeli",
    ["phi-3.5-mini", "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-7b"],
    index=0,
    help="Ölçüm: 0.5b uyduruyor, 1.5b dejenere, 7b kaliteli ama ~24 sn.",
)

st.sidebar.subheader("Erişim")
# Every widget below starts on the value `RunConfig` ships, so what the demo
# shows on open is the configuration the benchmark actually selected. Hardcoded
# indices drift the moment a default changes - which is exactly what happened
# once here, leaving the dashboard opening on RRF and a 2400-character context
# after both had been measured and replaced.
_defaults = RunConfig()
_fusion_options = ["rrf", "weighted", "dense_only", "lexical_only"]
fusion = st.sidebar.selectbox(
    "Füzyon", _fusion_options,
    index=_fusion_options.index(_defaults.retrieval.fusion),
    help="Ölçüm: weighted-0.5, Holm düzeltmesinden sonra ayakta kalan tek kazanç.",
)
alpha = st.sidebar.slider(
    "alpha (weighted)", 0.0, 1.0, _defaults.retrieval.alpha, 0.05,
    help="1.0 = sadece yoğun, 0.0 = sadece BM25",
    disabled=(fusion != "weighted"),
)
rrf_k = st.sidebar.slider("RRF k", 1, 120, _defaults.retrieval.rrf_k, 1,
                          disabled=(fusion != "rrf"))
top_k = st.sidebar.slider("top_k", 1, 12, _defaults.retrieval.top_k, 1)
dense_k = st.sidebar.slider("dense aday sayısı", 5, 60,
                            _defaults.retrieval.dense_top_k, 5)
lexical_k = st.sidebar.slider("BM25 aday sayısı", 5, 60,
                              _defaults.retrieval.lexical_top_k, 5)

st.sidebar.subheader("İyileştirmeler")
lexical_stem = st.sidebar.checkbox(
    "Türkçe gövdeleme", _defaults.retrieval.lexical_stem,
    help="Ölçüm: projenin en büyük tek etkisi (+0.21 nDCG, p=0.0014).",
)
query_instruction = st.sidebar.checkbox("Sorgu talimat öneki", False)
use_mmr = st.sidebar.checkbox("MMR çeşitlilik", False)
mmr_lambda = st.sidebar.slider("MMR lambda", 0.0, 1.0, 0.7, 0.05, disabled=not use_mmr)
rerank = st.sidebar.selectbox("Yeniden sıralama", ["none", "llm"], index=0)

st.sidebar.subheader("Üretim")
enable_abstention = st.sidebar.checkbox(
    "Çekimserlik açık", _defaults.generation.enable_abstention
)
st.sidebar.caption(
    f"Sinyal: `{_defaults.generation.abstention_signal}` "
    f"(AUC 0.843; füzyon skoru 0.727 ile kaybediyor — sorgu içinde "
    f"normalize edildiği için mutlak güven ifade edemiyor)."
)
threshold = st.sidebar.number_input(
    "Çekimserlik eşiği", value=_defaults.generation.abstention_threshold,
    step=0.1, format="%.5f",
    disabled=not enable_abstention,
    help="scripts/calibrate_abstention.py ile kalibre edildi: duyarlılığı "
         "%90'da tutan en katı eşik. Birim, seçilen sinyale bağlı (BM25 skoru).",
)
temperature = st.sidebar.slider(
    "Sıcaklık", 0.0, 1.0, _defaults.generation.temperature, 0.05
)
# Answerer's own default, not a widget-local guess: 1600 is the measured
# budget, and the slider previously opened at 2400 - a value the ablation
# showed already degrades instruction-following.
max_context = st.sidebar.slider(
    "Bağlam bütçesi (karakter)", 800, 8000, 1600, 200,
    help="Ölçüm: 1200 kr'de 2/3 doğru reddetme, 4000 kr'de 0/3. "
         "Uzun bağlam talimat takibini bozuyor ve gecikmeyi büyütüyor.",
)

cfg = build_config(**{
    "runtime.chat_model": chat_model,
    "retrieval.fusion": fusion,
    "retrieval.alpha": alpha,
    "retrieval.rrf_k": rrf_k,
    "retrieval.top_k": top_k,
    "retrieval.dense_top_k": dense_k,
    "retrieval.lexical_top_k": lexical_k,
    "retrieval.lexical_stem": lexical_stem,
    "retrieval.query_instruction": query_instruction,
    "retrieval.use_mmr": use_mmr,
    "retrieval.mmr_lambda": mmr_lambda,
    "retrieval.rerank": rerank,
    "generation.enable_abstention": enable_abstention,
    "generation.abstention_threshold": threshold,
    "generation.temperature": temperature,
})

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

st.title("Foundry RAG Lab")
st.caption(
    "Tamamen cihaz üzerinde çalışan hibrit RAG. "
    "Boğaziçi Üniversitesi yönetmelikleri · Foundry Local · internet yok."
)

if not INDEX_PATH.exists():
    st.error(f"İndeks bulunamadı: {INDEX_PATH}\n\nÖnce indeksi kurun.")
    st.stop()

runtime = get_runtime(chat_model, cfg.runtime.embedding_model)
pipeline = RagPipeline(cfg, INDEX_PATH, runtime=runtime)
pipeline.answerer.max_context_chars = max_context

stats = pipeline.store.stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Doküman", stats["n_documents"])
c2.metric("Chunk", stats["n_chunks"])
c3.metric("Vektör boyutu", stats["embedding_dim"])
c4.metric("İndeks", f"{stats['size_mb']} MB")

tab_ask, tab_compare, tab_about = st.tabs(["Soru sor", "Karşılaştır", "Sistem"])

# ---- ask -------------------------------------------------------------------

with tab_ask:
    question = st.text_input(
        "Soru", placeholder="burs başvurusu için gereken not ortalaması nedir"
    )
    go = st.button("Sor", type="primary")

    if go and question.strip():
        with st.spinner("Erişim..."):
            t0 = time.perf_counter()
            result = pipeline.retrieve(question)
            retrieval_ms = (time.perf_counter() - t0) * 1000

        left, right = st.columns([3, 2])

        with left:
            st.subheader("Cevap")
            if not result.hits:
                st.warning("Hiçbir pasaj bulunamadı.")
            else:
                placeholder = st.empty()
                streamed = ""
                t0 = time.perf_counter()
                for delta in pipeline.answerer.answer_stream(question, result):
                    streamed += delta
                    placeholder.markdown(streamed + "▌")
                placeholder.markdown(streamed)
                generation_ms = (time.perf_counter() - t0) * 1000

                m1, m2, m3 = st.columns(3)
                m1.metric("Erişim", f"{retrieval_ms:.0f} ms")
                m2.metric("Üretim", f"{generation_ms/1000:.1f} s")
                m3.metric("En yüksek skor", f"{result.top_score:.5f}")

        with right:
            st.subheader("Aşamalar")
            st.json({
                "yoğun aday": result.n_dense,
                "BM25 aday": result.n_lexical,
                "füzyon sonrası": result.n_fused,
                "yeniden sıralandı": result.reranked,
                "MMR": result.mmr_applied,
                **{k: round(v, 1) for k, v in result.stage_ms.items()},
            })

            # The refusal decision, shown as an inequality rather than a verdict:
            # a demo that says "refused" teaches nothing, while one that shows
            # 4.54 < 6.87 explains itself and makes the threshold arguable.
            st.subheader("Çekimserlik kararı")
            score = pipeline.answerer.abstention_score(result)
            if not enable_abstention:
                st.caption("Kapalı — sistem her durumda cevap üretiyor.")
            else:
                comparison = "<" if score < threshold else "≥"
                st.metric(
                    f"{cfg.generation.abstention_signal}",
                    f"{score:.3f} {comparison} {threshold:.3f}",
                    delta="reddedildi" if score < threshold else "cevaplandı",
                    delta_color="inverse" if score < threshold else "normal",
                )

        st.subheader("Getirilen pasajlar — neden bunlar?")
        rows = []
        for i, hit in enumerate(result.hits, start=1):
            rows.append({
                "#": i,
                "kaynak": hit.hit.citation_label(),
                "yoğun sıra": hit.dense_rank,
                "yoğun skor": round(hit.dense_score, 4) if hit.dense_score is not None else None,
                "BM25 sıra": hit.lexical_rank,
                "BM25 skor": round(hit.lexical_score, 4) if hit.lexical_score is not None else None,
                "füzyon": round(hit.fused_score, 5),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for i, hit in enumerate(result.hits, start=1):
            with st.expander(f"[{i}] {hit.hit.citation_label()} — {hit.explain()}"):
                st.write(hit.hit.display_text())

# ---- compare ---------------------------------------------------------------

with tab_compare:
    st.subheader("Aynı soru, farklı erişim stratejileri")
    st.caption(
        "Tek soruda strateji karşılaştırması sezgi verir ama kanıt değildir. "
        "Kanıt için eval seti üzerinde eşleştirilmiş permütasyon testi gerekir "
        "(evaluation/runner.py)."
    )
    cmp_question = st.text_input("Karşılaştırılacak soru", key="cmp")
    if st.button("Karşılaştır") and cmp_question.strip():
        strategies = ["dense_only", "lexical_only", "rrf", "weighted"]
        table = []
        for strategy in strategies:
            variant = build_config(**{
                "retrieval.fusion": strategy,
                "retrieval.top_k": top_k,
                "retrieval.dense_top_k": dense_k,
                "retrieval.lexical_top_k": lexical_k,
                "retrieval.lexical_stem": lexical_stem,
            })
            vp = RagPipeline(variant, INDEX_PATH, runtime=runtime)
            r = vp.retrieve(cmp_question)
            for rank, hit in enumerate(r.hits[:3], start=1):
                table.append({
                    "strateji": strategy, "sıra": rank,
                    "kaynak": hit.hit.citation_label(),
                    "metin": hit.hit.display_text()[:90],
                })
            vp.close()
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

# ---- about -----------------------------------------------------------------

with tab_about:
    st.subheader("Sistem durumu")
    st.json(pipeline.describe())
    st.subheader("Gecikme telemetrisi")
    summary = pipeline.telemetry.summary()
    if summary:
        st.dataframe(
            pd.DataFrame(summary).T.round(1), use_container_width=True
        )
    else:
        st.caption("Henüz ölçüm yok — bir soru sorun.")
