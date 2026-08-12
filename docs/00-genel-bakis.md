# Teori defteri

Bu klasör, koddaki her önemli kararın *neden* öyle olduğunu açıklar. Kod
"nasıl"ı, buradaki dokümanlar "neden"i anlatır.

| Doküman | Konu | Bağlı kod |
|---|---|---|
| [01 — Vektörler ve benzerlik](01-vektorler-ve-benzerlik.md) | Kosinüs, normalizasyon, boyutun laneti, Johnson–Lindenstrauss, asimetrik gömme | `runtime/foundry.py`, `index/store.py` |
| [02 — Chunking](02-chunking.md) | Dört strateji, örtüşme garantisi, yüzdelik eşik, Türkçe cümle bölme | `ingest/chunkers.py`, `ingest/normalize.py` |
| [03 — BM25 ve hibrit](03-bm25-ve-hibrit.md) | IDF'nin log-olasılık türevi, doygunluk, RRF, Borda/Arrow bağlantısı, MMR | `index/store.py`, `index/fusion.py` |
| [06 — Değerlendirmenin istatistiği](06-istatistik-degerlendirme.md) | nDCG'nin log indirimi, bootstrap, eşleştirilmiş permütasyon, Holm, güç, ECE | `evaluation/` |
| [07 — Sunum anlatımı](07-sunum-anlatimi.md) | Projeyi sözlü anlatma metni, demo koreografisi, beklenen sorular | — |

---

## Projenin tek cümlelik tezi

> Bir RAG sisteminin her tasarım kararı ölçülebilir; ölçülmeyen karar bir
> tercih değil, bir varsayımdır.

Bu, referans yaklaşımdan farkımızın tamamı. Proje planı "top-2 chunk al" der;
biz "kaç chunk alacağımızı eval seti üzerinde permütasyon testiyle belirledik"
deriz.

---

## Ölçümden çıkan kararlar (özet)

Hiçbiri literatürden alınmadı; hepsi bu korpusta, bu makinede ölçüldü.

| Karar | Ölçüm | Doküman |
|---|---|---|
| Vektörleri birim normalize et | L2 sıralaması = kosinüs sıralaması | 01 |
| Mutlak benzerlik eşiği kullanma | Alakasız kısa metinler 0.62 alıyor | 01 |
| `min_chunk_chars` alt sınırı koy | Kısa metinde geometri dejenere | 01, 02 |
| `MADDE` sınırlarında böl | Türk yönetmelik yapısı | 02 |
| Türkçe `lower()` yazma | `IŞIK → işik` bozuk | 03 |
| Gövdeleme uygula | `kayıtların` ↔ `kayıt` eşleşiyor | 03 |
| Skor toplama, sıra kullan (RRF) | BM25 0.998 vs dense 0.368, aynı belge | 03 |
| Bağlam bütçesini 1600 kr'de tut | 4000 kr'de talimat takibi çöküyor | README |
| phi-3.5-mini seç | 0.5b uyduruyor, 1.5b dejenere, 7b 24 sn | README |
| Eval setini korpustan üret | "kayıt dondurma" korpusta yok | 06 |
| Holm düzeltmesi uygula | 4 "keşif" → 1 gerçek | 06 |

---

## Bilinen sınırlar

Dürüstlük, projenin bir parçası. Bunlar raporda açıkça yazılacak:

1. **Gövdeleme türetim eklerini yakalayamıyor.** `dondurulma` ≠ `dondurma`.
   Gerçek çözüm morfolojik çözümleyici; kapsam dışı bırakıldı.
2. **Prompt deneyinin hücre başına 5 sorusu var.** Türkçe'deki monoton trend
   ikna edici, ama tek tek hücre karşılaştırmaları bu örneklemle kanıt değil.
3. **Cross-encoder reranker yok.** Foundry Local kataloğunda bulunmuyor;
   listwise LLM reranking ikame olarak kullanıldı.
4. **Sorgu–belge kelime uyumsuzluğu çözülmedi.** Tespit edildi ve ölçüldü,
   ama sorgu genişletme uygulanmadı.
5. **Gömme yavaş:** 1.2 chunk/sn. Parametre taramasında chunking değişirse
   yeniden indeksleme pahalı.
