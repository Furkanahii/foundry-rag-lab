# Foundry RAG Lab

[![tests](https://github.com/Furkanahii/foundry-rag-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Furkanahii/foundry-rag-lab/actions/workflows/tests.yml)

Tamamen cihaz üzerinde çalışan, hibrit erişimli ve **istatistiksel olarak
doğrulanan** bir Türkçe RAG sistemi. Microsoft Foundry Local üzerine kurulu;
hiçbir aşamada internet veya bulut hesabı gerekmiyor.

Bu bir RAG *demosu* değil, bir RAG *laboratuvarı*: her tasarım kararı ölçümle
gerekçelendiriliyor, ve sistemin kendisi o ölçümü yapacak araçları içeriyor.

## Hızlı başlangıç

```bash
/opt/homebrew/bin/python3.12 -m venv .venv        # x86_64 Python OLMAZ, bkz. docs/07
./.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src ./.venv/bin/python scripts/smoke_test.py    # modelleri indirir
PYTHONPATH=src ./.venv/bin/streamlit run app/dashboard.py
```

İndeks (`data/index/bogazici.db`) depoda hazır geliyor, yani ilk çalıştırma için
yeniden gömme gerekmiyor. Korpusu değiştirirsen:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/build_index.py --force
```

### Komutlar

| script | ne yapar |
|---|---|
| `build_index.py` | Korpustan hibrit indeksi kurar (dense + BM25) |
| `build_eval_set.py` | Değerlendirme setini korpustan üretir |
| `run_benchmark.py` | 11 konfigürasyonu iki kelime varyantında karşılaştırır |
| `calibrate_abstention.py` | Çekimserlik sinyalini ve eşiğini veriden türetir |
| `check_paraphrase_fidelity.py` | Parafrazların anlamı koruduğunu ölçer |
| `compare_chat_models.py` | Üretim modellerini gecikme/tekrar/reddetme ile karşılaştırır |
| `prompt_ablation.py` | Bağlam bütçesini tarar |
| `smoke_test.py` | Ortamı ve modelleri doğrular |

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

## Kurulum notu: mimari tuzağı

Apple Silicon Mac veya Windows. Kurulum komutları yukarıda; buradaki tek konu
kolay kaçırılan bir tuzak.

`foundry-local-sdk` **mimariye göre iki farklı kütüphane** kuruyor ve ikisinin
API'si ortak değil:

| mimari | sürüm | ne geliyor |
|---|---|---|
| arm64 (Apple Silicon) | 1.2.x | native runtime dahil; Homebrew kurulumuna gerek yok |
| x86_64 | 0.5.x | ince HTTP istemcisi, tamamen farklı çağrılar |

Bu proje 1.x API'sini hedefliyor. Anaconda Python'u genelde x86_64 olduğu için
sanal ortamı `/opt/homebrew/bin/python3.12` ile kurmak şart. `requirements.txt`
tabanı bu yüzden `>=1.2`: x86_64 üzerinde pip çözümlemeyi **başarısız kılar**,
ki doğru davranış budur — 0.5.x kurulsaydı gereksinimi karşılar ve ilk çağrıda
kırılırdı.

Ayrıntı ve diğer ortam tuzakları: [docs/07](docs/07-model-secimi.md).

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
# İndeksi kur (chunking veya gömme modeli değiştiyse --force gerekir)
PYTHONPATH=src ./.venv/bin/python scripts/build_index.py

# Farklı bir chunking stratejisi dene
PYTHONPATH=src ./.venv/bin/python scripts/build_index.py --strategy sentence_window --force

# Model seçimini yeniden ölç
PYTHONPATH=src ./.venv/bin/python scripts/compare_chat_models.py
```

`RunConfig` chunking ve gömme modelini geri kalan ayarlardan **ayrı** hashliyor
(`index_fingerprint`). Erişim ve üretim ayarları sorgu anında uygulandığı için
yeniden gömme gerektirmiyor; çoğu deney 20 dakikalık bir yeniden indekslemeden
2 saniyelik bir config yüklemesine iniyor. `build_index.py` bu parmak izini
kontrol edip indeks güncelse hiç çalışmıyor.

**Uyarı:** indeksi yeniden kurmak chunk id'lerini değiştirir, dolayısıyla
`data/eval/eval_set.json` içindeki gold etiketleri geçersizleşir. Script bunu
hatırlatıyor; eval setini yeniden üretmeden koşulan bir benchmark anlamsızdır.

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
- [04 — Üretim ve grounding](docs/04-uretim-ve-grounding.md)
- [05 — Kalibrasyon ve çekimserlik](docs/05-kalibrasyon-abstention.md)
- [06 — Değerlendirmenin istatistiği](docs/06-istatistik-degerlendirme.md)
- [07 — Model seçimi](docs/07-model-secimi.md)
- [08 — Sunum anlatımı](docs/08-sunum-anlatimi.md) — projeyi sözlü anlatmak için
  dakika dakika metin, demo koreografisi ve beklenen sorular

## Değerlendirme

```bash
# Eval setini korpustan üret (bir kez, sonra sabit kalır)
PYTHONPATH=src ./.venv/bin/python scripts/build_eval_set.py --n 60 --paraphrase --model qwen2.5-7b

# Konfigürasyon taraması + eşleştirilmiş permütasyon testi + Holm düzeltmesi
PYTHONPATH=src ./.venv/bin/python scripts/run_benchmark.py
```

### Sonuçlar

60 üretilmiş soru + 60 parafraz + 52 cevaplanamaz soru. Her konfigürasyon her
iki kelime varyantında ayrı koşuldu; p değerleri eşleştirilmiş permütasyon
testinden, anlamlılık Holm-Bonferroni sonrası.

| konfigürasyon | üretilmiş | parafraz | temele karşı (üretilmiş) | (parafraz) |
|---|---|---|---|---|
| baseline-dense (referans) | 0.5796 | 0.5158 | — | — |
| **weighted-0.5** | **0.7370** | **0.6295** | **+0.157 (p=0.0006) ✓** | **+0.114 (p=0.0002) ✓** |
| weighted-0.7 | 0.7001 | 0.5915 | +0.121 (p=0.0005) ✓ | +0.076 (p=0.0011) ✓ |
| lexical-only | 0.7040 | 0.5577 | +0.124 (p=0.065) | +0.042 (p=0.47) |
| rrf | 0.6817 | 0.6130 | +0.102 (p=0.026) | +0.097 (p=0.0088) |
| rrf-instruct | 0.6871 | 0.6233 | +0.108 (p=0.052) | +0.108 (p=0.028) |
| lexical-nostem | 0.4912 | 0.3362 | −0.088 (p=0.23) | −0.180 (p=0.0091) |
| rrf-mmr | 0.5481 | 0.4915 | −0.032 (p=0.21) | −0.024 (p=0.38) |

✓ = Holm düzeltmesinden sonra ayakta kalan. Metrik nDCG@5.

**Üç bulgu:**

**1. Ağırlıklı füzyon, düzeltmeden sonra ayakta kalan tek erişim kazancı** — ve
her iki varyantta da. RRF hiçbir varyantta geçemiyor. Bu, projenin varsayılanını
`rrf`'ten `weighted` (alpha=0.5) değiştirdi.

**2. Türkçe gövdeleme en büyük tek etki.** Planlı karşılaştırma (taramanın
parçası değil, önceden kurulmuş hipotez):

| hipotez | varyant | fark | p | d |
|---|---|---|---|---|
| Türkçe gövdeleme, BM25 kolunda | üretilmiş | +0.213 | 0.0014 | 0.45 |
| Türkçe gövdeleme, BM25 kolunda | parafraz | +0.222 | 0.0004 | 0.45 |
| hibrit füzyon, saf yoğun aramaya karşı | üretilmiş | +0.102 | 0.026 | 0.29 |
| hibrit füzyon, saf yoğun aramaya karşı | parafraz | +0.097 | 0.0088 | 0.35 |

**3. Sözlüksel üstünlük bir kurgu artefaktıydı.** Soru yeniden yazıldığında
yalnızca BM25 kolları çöküyor:

| konfigürasyon | düşüş | p | |
|---|---|---|---|
| lexical-nostem | −0.155 | 0.0014 | anlamlı düşüş |
| lexical-only | −0.146 | 0.0020 | anlamlı düşüş |
| weighted-0.5 | −0.108 | 0.0121 | Holm'a takıldı |
| rrf | −0.069 | 0.0735 | anlamsız |
| baseline-dense | −0.064 | 0.1121 | anlamsız |

Geçersiz ilk koşuda sözlüksel arama 0.86 ile yoğun aramanın 0.58'ini eziyor
görünüyordu. Şimdi: üretilmiş sette avantaj anlamsız (p=0.065), parafraz sette
tamamen yok (p=0.47). O üstünlük soruların yazılış biçiminden geliyormuş.

### Çekimserlik: füzyon skoru yanlış sinyal

Sistemin cevabı bilmediğinde susması gerekiyor, ve eşik tahmin edilmemeli.
`scripts/calibrate_abstention.py` üç aday sinyali 60 cevaplanabilir / 52
cevaplanamaz soru üzerinde ölçtü:

| sinyal | AUC | cevaplanamaz sorulardaki **en yüksek** skor |
|---|---|---|
| **ham BM25** | **0.843** | 19.07 (aralık 2.41–47.78) |
| ham kosinüs | 0.792 | 0.672 |
| füzyon skoru | 0.727 | **1.000** — mümkün olan en yüksek değer |

Füzyon skoru yapısal olarak kaybediyor. `weighted_fusion` her kolu **o sorgu
için getirilen adaylar içinde** min-max normalize ediyor, yani en iyi aday iyi
olduğu için değil en iyi olduğu için 1.0 alıyor. "Hiçbir şey bulamadım"
diyemeyen bir sayı, bir şey bulunup bulunmadığına karar veremez.

Seçilen eşik 6.866: duyarlılığı %90'da tutan en katı kesme noktası. F1'i değil
duyarlılığı hedeflemek bilinçli bir tercih — gereksiz ret kullanıcıyı sinirlendirir,
uydurulmuş bir yönetmelik maddesi birinin eğitim hayatını etkiler.
Gerekçe ve sınırlar: [docs/05](docs/05-kalibrasyon-abstention.md).

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

**Tamamlanan:** runtime ve telemetri, ingestion, chunking, hibrit indeks,
füzyon, yeniden sıralama, değerlendirme harness'ı (nDCG/MRR/Recall, bootstrap
güven aralıkları, eşleştirilmiş permütasyon testi, Holm-Bonferroni, ECE),
Streamlit dashboard, teori dokümanları, ve geçerli bir benchmark koşusu
(60 soru + 60 parafraz, varyantlar ayrı ölçülmüş).

Çekimserlik eşiği de kalibre edildi ve varsayılanlar ölçüme göre değişti:
füzyon `rrf` → `weighted` (alpha=0.5), çekimserlik sinyali ham BM25, eşik 6.866.

**Sırada:**
1. Sorgu genişletme — "kayıt dondurma" ile "izinli sayılma" arasındaki
   kelime dağarcığı boşluğu ölçüldü ama kapatılmadı.
2. Chunk boyutunun sweep'e sokulması — 900 karakter gerekçeli bir tahmin,
   ölçülmüş bir seçim değil.
3. Dayanaklılık yargıcı — atıfların geçerliliği ölçülüyor, iddiayı gerçekten
   destekleyip desteklemediği ölçülmüyor.
4. MCP sunucusu — sistemi Microsoft ekosistemindeki araçlardan çağırılabilir
   yapmak.

**Bilinen sınırlar** (hepsi ölçülmüş, hiçbiri gizlenmemiş):

- Eval setini modelin kendisi üretiyor. Parafraz katmanı bunu hafifletiyor ama
  çözmüyor; 60 parafrazın 11'i (%18) orijinaliyle 0.80'in altında anlamsal
  benzerlik gösteriyor (`scripts/check_paraphrase_fidelity.py`), yani ölçülen
  kelime farkı gerçek sözlüksel avantajın bir **üst** sınırı.
- Küçük etkiler için güç yetersiz: n=60, d=0.2'yi görmek için 197 gerekiyor.
- Korpus tek kurum, tek dil. İngilizce kontrol seti yok.
- Gövdeleme morfolojik analizci değil; çekim eklerinde çalışıyor, türetim
  eklerinde çalışmıyor ("dondurulması" ile "dondurma" hâlâ eşleşmiyor).
- Çekimserlik eşiği 52 negatif üzerinde kalibre edildi ve cevaplanamaz
  soruların %38'ini kaçırıyor. BM25 skoru sorgu uzunluğuna duyarlı; bu
  karıştırıcı kontrol edilmedi.
- Bağlam bütçesi ablasyonu n=5 ile yapıldı. Etkinin yönü güvenilir, 1600
  karakter rakamı gürültülü.
