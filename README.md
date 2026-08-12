# Foundry RAG Lab

Tamamen cihaz üzerinde çalışan, hibrit erişimli ve **istatistiksel olarak
doğrulanan** bir Türkçe RAG sistemi. Microsoft Foundry Local üzerine kurulu;
hiçbir aşamada internet veya bulut hesabı gerekmiyor.

Bu bir RAG *demosu* değil, bir RAG *laboratuvarı*: her tasarım kararı ölçümle
gerekçelendiriliyor, ve sistemin kendisi o ölçümü yapacak araçları içeriyor.

---

## Neden bu mimari

Referans yaklaşım (Microsoft Learn tutorial'ı ve proje planı) şudur: metni
parçala, göm, SQLite'a JSON olarak yaz, Python'da tek tek kosinüs hesapla,
en iyi 2 parçayı modele ver. Bu, sekiz cümlelik bir örnek için çalışır.

Gerçek bir Türkçe yönetmelik korpusunda ölçtüğümüz sorunlar:

| Sorun | Ölçüm | Çözüm |
|---|---|---|
| Kısa metinlerde gömme geometrisi bozuluyor | alakasız kısa cümleler 0.62 kosinüs | `min_chunk_chars` alt sınırı |
| Tam eşleşme gereken sorgular (ders kodu) | dense marj 0.368 vs 0.304 | BM25 kolu + RRF füzyonu |
| `MADDE` 9 dokümanın hepsinde, 195 kez | IDF = log(N/df) = 0 | BM25 tek başına yetersiz, hibrit şart |
| Türkçe `lower()` bozuk | `IŞIK → işik` | dile özgü normalizasyon |
| Çekim ekleri eşleşmiyor | `kayıtların` vs `kayıt` | Türkçe gövdeleme |
| PDF kelimeleri kırıyor | 21 gerçek kopukluk | sözlük kanıtlı onarım |
| Kullanıcı ile belge farklı kelime kullanıyor | "kayıt dondurma" korpusta **yok**, terim "izinli sayılma" | sorgu genişletme + korpustan üretilen eval seti |
| Uzun bağlam talimat takibini bozuyor | 1200 kr'de 2/3 doğru reddetme → 4000 kr'de **0/3** | bağlam bütçesi 1600 kr; geniş getir, dar besle |

### Bağlam bütçesi: sezginin tersi

`scripts/prompt_ablation.py`, phi-3.5-mini üzerinde bağlam boyutunu taradı:

| bağlam | doğru reddetme | doğru cevap | gecikme |
|---|---|---|---|
| **1200 kr** | **2/3** | **2/2** | 38.6 s |
| 2400 kr | 1/3 | 1/2 | 59.6 s |
| 4000 kr | 0/3 | 0/2 | 57.2 s |
| 6000 kr | 0/3 | 1/2 | 93.9 s |

Tek yönlü çöküş. Çoğu RAG anlatısı "daha çok bağlam daha iyi cevap" varsayar;
küçük bir modelde bağlam bir kaynak değil **maliyettir** — grounding kuralları
sistem mesajının başında durur ve sonrasında gelen her şeyle seyrelir.

Bu, erişimle üretim arasında kasıtlı bir gerilim yaratır: `top_k=5` kalır çünkü
recall@5 ölçtüğümüz şeydir, ama bağlam bütçesi sadece ilk ~2 chunk'ın modele
ulaşmasını sağlar. **Geniş getir, dar besle.**

---

## Kurulum

Apple Silicon Mac veya Windows. **Önemli:** `foundry-local-sdk` paketinin arm64
sürümü (1.2.x) native runtime'ı da getirir; Homebrew kurulumuna gerek yoktur.
x86_64 üzerinde PyPI yalnızca 0.5.x sunar ve API'si tamamen farklıdır — Anaconda
Python'u genelde x86_64'tür, bu yüzden native bir Python şart.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Ortamı doğrula (modelleri indirir, ilk çalıştırma uzun sürer):

```bash
PYTHONPATH=src ./.venv/bin/python scripts/smoke_test.py
```

---

## Mimari

```
data/corpus/*.pdf,*.html
        │
        ▼
  ingest/loaders.py      PDF/HTML/MD okuma, tekrarlayan başlık temizliği,
        │                sözlük kanıtlı kelime kopukluğu onarımı
        ▼
  ingest/normalize.py    Türkçe küçük harf (i/İ/ı/I), NFC, cümle bölme
        │                (kısaltma + sıra sayısı farkındalıklı), gövdeleme
        ▼
  ingest/chunkers.py     fixed | recursive | sentence_window | semantic
        │                (recursive: "MADDE 7" sınırlarını tanır)
        ▼
  runtime/foundry.py     Foundry Local: qwen3-embedding-0.6b (dim 1024)
        │                + phi-3.5-mini (üretim)
        ▼
  index/store.py         TEK SQLite dosyası:
        │                  chunk_vec  → sqlite-vec (yoğun, C'de)
        │                  chunks_fts → FTS5 BM25 (sözcüksel, C'de)
        ▼
  index/fusion.py        RRF | weighted | dense_only | lexical_only + MMR
        ▼
  (sırada) retrieve/ → generate/ → evaluation/
```

Sunucu yok, Docker yok. İndeks tek bir `.db` dosyası — USB'ye kopyalanır, çalışır.

---

## Kullanım

```bash
# İndeksi kur
PYTHONPATH=src ./.venv/bin/python -c "
from frag.config import RunConfig
from frag.runtime.foundry import FoundryRuntime
from frag.index.builder import build_index
cfg = RunConfig(); rt = FoundryRuntime(cfg.runtime)
build_index(cfg, rt, 'data/corpus', 'data/index/bogazici.db', progress=print)
"
```

Model seçimini yeniden ölçmek için:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/compare_chat_models.py
```

---

## Ölçülmüş model seçimi

Apple M2 / WebGPU, grounded Türkçe soru-cevap:

| model | gecikme | tekrar skoru | reddetme | karar |
|---|---|---|---|---|
| qwen2.5-0.5b | 660 ms | 0.000 | ✗ uyduruyor | güvensiz |
| qwen2.5-1.5b | 10.8 s | **0.315** | ✗ | dejenere |
| **phi-3.5-mini** | 7.2 s | 0.000 | ✓ | **varsayılan** |
| qwen2.5-7b | 23.8 s | 0.000 | ✓ | en kaliteli, çok yavaş |

Not: 0.5b modeli tekrar skorunda 0.000 aldı ama içeriği uydurdu. Tekrar metriği
dejenerasyonu yakalar, **halüsinasyonu yakalamaz** — bu yüzden değerlendirme
katmanında ayrı bir faithfulness yargıcı gerekiyor.

---

## Korpus

Boğaziçi Üniversitesi'nin kamuya açık yönetmelik ve yönergeleri
(`bogazici.edu.tr`): kayıt, burs, özel öğrenci, disiplin, konut tahsis,
sertifika programları, lisans ve lisansüstü eğitim-öğretim yönetmelikleri.

9 doküman · 214.982 karakter · 347 chunk · 4.63 MB indeks

---

## Teori defteri

Koddaki her kararın matematiksel gerekçesi [docs/](docs/00-genel-bakis.md)
altında:

- [01 — Vektörler ve benzerlik](docs/01-vektorler-ve-benzerlik.md)
- [02 — Chunking](docs/02-chunking.md)
- [03 — BM25 ve hibrit](docs/03-bm25-ve-hibrit.md)
- [06 — Değerlendirmenin istatistiği](docs/06-istatistik-degerlendirme.md)

## Değerlendirme

```bash
# Eval setini korpustan üret (bir kez, sonra sabit kalır)
PYTHONPATH=src ./.venv/bin/python scripts/build_eval_set.py --n 60 --paraphrase --model qwen2.5-7b

# Konfigürasyon taraması + eşleştirilmiş permütasyon testi + Holm düzeltmesi
PYTHONPATH=src ./.venv/bin/python scripts/run_benchmark.py
```

### Bir benchmark'ı ne geçerli kılar

Bu projenin ilk ölçüm koşusu çöpe gitti. Nedenleri
[data/results/archive-n15-invalid/NOTE.md](data/results/archive-n15-invalid/NOTE.md)
altında duruyor, çünkü bir ölçümün neden yorumlanamaz hale geldiği ölçümün
kendisi kadar öğretici. Üç şey düzeltildi ve üçü de artık kodun içinde:

**Güç.** `--n` artık *kullanılabilir* soru hedefi, örneklenen chunk sayısı değil.
Eskiden 30 istenip 15 elde ediliyordu ve bunu hiçbir şey söylemiyordu. Eşleştirilmiş
tasarımda %80 güçle d=0.5'i görmek 32 soru ister; hem `build_eval_set.py` hem
`run_benchmark.py` bunu çıktısında raporluyor.

**Kelime örtüşmesi.** Sorular chunk'lardan üretiliyor, dolayısıyla chunk'ın söz
varlığını taşıyorlar — BM25'in puanladığı şeyin ta kendisi. Artık her soru için
gold chunk ile Jaccard örtüşmesi kaydediliyor, **BM25 kolunun kullandığı aynı
gövdelemeyle** hesaplanarak, yani bir vekil değil sinyalin kendisi.

**Varyantların ayrılması.** Tarama iki kez koşuyor: üretilmiş sorular ve onların
günlük dile çevrilmiş parafrazları. İki set gold chunk'a göre eşleştirilmiş
olduğundan aradaki düşüş permütasyon testiyle ölçülebiliyor. Bu düşüş, bir
konfigürasyonun puanının ne kadarının pasajı *bulmaktan*, ne kadarının kelime
*eşleştirmekten* geldiğini söyleyen sayı. Parafrazlanmış sütun, kullanıcının
gerçekten yazdığı sorulara verilen yanıtın tahminidir.

## Dashboard

```bash
PYTHONPATH=src ./.venv/bin/streamlit run app/dashboard.py
```

## Durum

Tamamlanan: runtime, telemetri, ingestion, chunking, hibrit indeks, füzyon.
Sırada: reranking + abstention, değerlendirme harness'ı (nDCG/MRR/Recall,
bootstrap güven aralıkları, permütasyon testi, ECE kalibrasyonu), Streamlit
dashboard, teori dokümanları.
