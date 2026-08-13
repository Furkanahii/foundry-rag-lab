# 07 — Model seçimi

> Bağlı kod: `scripts/compare_chat_models.py`, `config.RuntimeConfig`

Bu projede iki model rolü var ve ikisi tamamen farklı kısıtlar altında
seçiliyor: **gömme modeli** indeksleme boyunca sürekli çağrılıyor (verim
odaklı), **üretim modeli** soru başına bir kez (gecikme odaklı).

---

## 1. Üretim modeli: dört aday, üç ölçüt

Foundry Local kataloğundaki dört model, aynı grounded Türkçe soru-cevap
görevinde ölçüldü (Apple M2, WebGPU):

| model | gecikme | tekrar skoru | dayanaklı | reddediyor mu | karar |
|---|---|---|---|---|---|
| qwen2.5-0.5b | 660 ms | 0,000 | ✗ | ✗ **uyduruyor** | güvensiz |
| qwen2.5-1.5b | 10,8 s | **0,315** | ✗ | ✗ | dejenere |
| **phi-3.5-mini** | 7,2 s | 0,000 | ✓ | ✓ | **varsayılan** |
| qwen2.5-7b | 23,8 s | 0,000 | ✓ | ✓ | çevrimdışı işler |

### Tekrar skoru nedir

Üretilen metindeki 5-gram'ların kaçının tekrar olduğu. Dejenere çözümleme
(degenerate decoding) — modelin aynı cümleyi döngüye girip tekrarlaması —
küçük modellerde en sık görülen bozulma biçimi ve hiçbir prompt düzeltmesi
onu kurtarmıyor. 0,15 üzeri bir skor modeli kullanılamaz yapıyor.

qwen2.5-1.5b'nin 0,315'i tam olarak bu: model gramerli cümleler üretiyor ama
aynı cümleleri üretiyor.

### Ölçümün kendisi hakkında bir uyarı

qwen2.5-0.5b tekrar skorunda **mükemmel** (0,000) ama içeriği uyduruyor.

Bu, projedeki en öğretici tek gözlem olabilir: **tekrar metriği dejenerasyonu
yakalıyor, halüsinasyonu yakalamıyor.** İki farklı bozulma biçimi ve tek
metrik ikisini birden göremiyor. Bir metriğin iyi görünmesi, ölçmediği bir
şeyin iyi olduğu anlamına gelmiyor.

Bu yüzden değerlendirme katmanında ayrı bir dayanaklılık kontrolü var
(atıf doğrulama, `docs/04`), ve bu yüzden reddetme davranışı ayrı bir sütun
olarak ölçülüyor.

### Neden en iyisi değil de phi-3.5-mini

qwen2.5-7b her kalite ölçütünde en iyisi. 23,8 saniye.

Bu bir **ürün kararı**, teknik bir sonuç değil ve öyle sunulmamalı: interaktif
bir demoda 24 saniyelik bekleme kullanılamaz. Ama gecikmenin önemsiz olduğu
işlerde 7b tercih ediliyor — değerlendirme seti üretimi (`build_eval_set.py
--model qwen2.5-7b`) çevrimdışı çalışıyor ve orada kalite tek ölçüt. Nitekim
ilk denemede phi-3.5-mini bozuk Türkçe sorular üretmişti.

Yani "varsayılan model" tek bir seçim değil, role göre iki seçim.

---

## 2. Gömme modeli

`qwen3-embedding-0.6b`, 1024 boyut. Katalogdaki alternatif
(`qwen3-embedding-8b`) bu makinede indeksleme süresini kabul edilemez hale
getiriyor ve 347 chunk'lık bir korpusta getirisi ölçülmedi — dürüst durum
budur: bu bir karşılaştırma sonucu değil, bir kaynak kısıtı.

Gömme modeli değişirse indeks yeniden kurulmak zorunda. `RunConfig` bunu
`index_fingerprint()` ile ayrı hashliyor: erişim ve üretim ayarları
yeniden gömme gerektirmiyor, chunking ve gömme modeli gerektiriyor. Bu
ayrım, çoğu deneyi 20 dakikalık bir yeniden indekslemeden 2 saniyelik bir
config yüklemesine indiriyor.

---

## 3. Yürütme sağlayıcısı (execution provider) neden kaydediliyor

Foundry, çözümlenen model kimliğinde donanım hedefini de veriyor:
`qwen2.5-0.5b-instruct-generic-gpu:4`. `LoadedModel.execution_provider` bunu
ayrıştırıyor ve **her benchmark sonucuna yazılıyor**.

Sebep basit: gecikme sayıları yürütme sağlayıcısı bilinmeden anlamsız. CPU
üzerinde ölçülmüş 7 saniye ile GPU üzerinde ölçülmüş 7 saniye aynı cümleyi
kurmuyor. Sonuçlar yalnızca donanım yolu eşleştiğinde karşılaştırılabilir, bu
yüzden sağlayıcı bir dipnot değil deney kaydının parçası.

---

## 4. Ortam tuzağı: aynı paket, iki farklı API

`pip install foundry-local-sdk` mimariye göre **tamamen farklı** bir paket
kuruyor:

| mimari | sürüm | ne geliyor |
|---|---|---|
| arm64 (Apple Silicon) | 1.2.4 | yerel çalışma zamanı dahil, `onnxruntime-genai` ile |
| x86_64 | 0.5.1 | ince bir HTTP istemcisi, API'si bambaşka |

Anaconda Python'u genelde x86_64 olduğu için bu makinede
`/opt/homebrew/bin/python3.12` şart. Ayrıca Apple Silicon'da Homebrew
`foundrylocal` kurulumuna **gerek yok** — wheel çalışma zamanını taşıyor.

Bir uyarı daha: Microsoft Learn 1.x API'sini belgeliyor, PyPI x86_64'te 0.5.x
sunuyor. Dokümantasyon ile paket çelişebiliyor; kod yazmadan önce imzaları
doğrulamak gerekiyor.

---

## 5. Sampling kontrolü neden web servisi üzerinden

Native istemci (`model.get_chat_client()`) basit ve bağımlılıksız, ama
`complete_chat(messages, tools)` **hiçbir örnekleme parametresi sunmuyor** —
sıcaklığı sabitlemenin yolu yok.

`manager.start_web_service()` OpenAI protokolü konuşan yerel bir port açıyor
ve `temperature`, `max_tokens`, `logprobs` kabul ediyor.

Bu proje ikisini de kullanıyor: **gömme** native istemciyle (gömmeler
deterministik, ayarlanacak bir şey yok), **üretim** web servisiyle (sıcaklığın
sabit olması benchmark'ın karşılaştırılabilirliği için şart). Web servisi
başlatılamazsa native istemciye düşülüyor ve bu durum kaydediliyor —
`sampling_is_controlled` False olan bir koşu, True olan bir koşuyla
karşılaştırılamaz.
