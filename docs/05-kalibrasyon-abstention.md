# 05 — Kalibrasyon ve çekimserlik

> Bağlı kod: `evaluation/stats.py` (`find_best_threshold`,
> `expected_calibration_error`), `generate/answerer.py`,
> `scripts/calibrate_abstention.py`

Bir RAG sisteminin en önemli davranışı doğru cevap vermek değil, **cevabı
bilmediğinde susmaktır.** Yanlış cevap tek bir kullanıcıyı yanıltır; uydurulmuş
bir yönetmelik maddesi, o maddeye dayanarak karar veren herkesi yanıltır.

Bu doküman, "skor düşükse cevaplama" cümlesinin neden yetersiz olduğunu ve
eşiğin nereden geldiğini anlatıyor.

---

## 1. Hangi skor?

İlk sezgi, pipeline'ın ürettiği füzyon skorunu kullanmaktır. Bu yanlış, ve
yanlışlığı ayar meselesi değil **yapısal**.

`weighted_fusion`, her iki kolun skorlarını min-max normalize ediyor — ama
korpus genelinde değil, **o sorgu için getirilen adaylar içinde**:

```
norm(v) = (v - min) / (max - min)
```

Bu formülde en iyi aday her zaman 1.0 alır. İyi olduğu için değil, en iyi
olduğu için. Yani füzyon skoru "bulduğumun en iyisi bu" der; "bulduğum şey
işe yarar" demez. Cevabı korpusta olmayan bir soru için de en iyi aday yine
1.0 alır.

Bu, ölçülebilir bir iddia. `scripts/calibrate_abstention.py` üç aday sinyali
60 cevaplanabilir ve 52 cevaplanamaz soru üzerinde karşılaştırıyor:

| sinyal | AUC | cevaplanabilir ort. | cevaplanamaz ort. | cevaplanamaz **maks** |
|---|---|---|---|---|
| **lexical_raw** (ham BM25) | **0.843** | 13.86 | 7.56 | 19.07 |
| dense_raw (ham kosinüs) | 0.792 | 0.597 | 0.474 | 0.672 |
| fused (normalize füzyon) | 0.727 | 0.850 | 0.712 | **1.000** |

Son sütun teoriyi doğruluyor: cevaplanamaz bir soru, füzyon skorunda
**mümkün olan en yüksek değeri** almış. Ham skorlar normalize edilmediği için
sorgular arasında karşılaştırılabilir, ki bir eşiğin ihtiyaç duyduğu tek
özellik budur.

AUC neden önce ölçülüyor: AUC eşikten bağımsızdır. "Bu skor sinyali taşıyor
mu" sorusunu, kesme noktasının nereye konduğuna bakmadan yanıtlar. Sinyalin
kendisi ayırt edemiyorsa hiçbir eşik onu kurtarmaz — ve bu, eşik aramadan
önce bilinmesi gereken şeydir.

**Karar:** `abstention_signal = "lexical_raw"`.

---

## 2. Eşik nereden geliyor

`find_best_threshold` gözlenen skorların arasındaki her orta noktayı deniyor
ve seçilen amacı maksimize edeni döndürüyor. Ama "en iyi eşik" sorusunun
cevabı, hangi hatayı daha kötü saydığına bağlı:

| amaç | eşik | F1 | doğruluk | kesinlik | duyarlılık | TP/FP/FN/TN |
|---|---|---|---|---|---|---|
| F1-optimal | 6.590 | 0.832 | 0.795 | 0.740 | 0.950 | 57/20/3/32 |
| **güvenlik ağırlıklı** | **6.866** | 0.806 | 0.768 | 0.730 | 0.900 | 54/20/6/32 |

F1 iki hata türüne eşit ağırlık verir. Bu sistem için bu yanlış bir varsayım:

- **Yanlış reddetme (FN):** kullanıcı sinirlenir, soruyu yeniden sorar.
- **Yanlış cevaplama (FP):** sistem, birinin eğitim hayatı hakkında var
  olmayan bir kural uydurur.

İkincisi daha ağır. Bu yüzden varsayılan, F1'i değil **duyarlılığı %90'da
tutan en katı eşiği** alıyor: cevaplanabilir soruların %90'ı hâlâ
cevaplanıyor, karşılığında daha fazla gereksiz ret satın alınıyor.

Bu bir mühendislik kararı, matematiksel bir sonuç değil — ve öyle olduğu
açıkça söylenmeli. Farklı bir dağıtım bağlamı (örneğin bir danışman
öğrencinin yanında oturuyorsa) tersini tercih edebilir.

**Karar:** `abstention_threshold = 6.865533`.

---

## 3. Beklenen Kalibrasyon Hatası (ECE) — ve neden burada hesaplanmadı

ECE, bir skorun olasılık gibi okunup okunamayacağını ölçer: tahminleri güven
düzeyine göre kutulara ayırır, her kutuda güven ile gözlenen doğruluk
arasındaki farkı alır, kutu büyüklüğüne göre ağırlıklandırır.

```
ECE = Σ (n_kutu / N) · |doğruluk_kutu − güven_kutu|
```

Kutu ağırlığı önemli: iki örnek içeren berbat bir kutu, iki yüz örnek içeren
iyi bir kutuyu bastırmamalı. `MCE` ise en kötü kutuyu raporlar, çünkü
çekimserlik gibi bir *güvenlik* kararında ortalama değil en kötü durum
belirleyici olabilir. Brier skoru da hesaplanıyor, çünkü ECE'nin aksine
uygun bir skorlama kuralı — taban oranı tahmin eden bir model onu
kandıramaz.

**Ama seçilen sinyalde ECE hesaplanmıyor.** BM25 skoru `[0,1]` aralığında
değil; bu korpusta gözlenen aralık 2.41–47.78. Bir olasılıkla
karşılaştırılamayan bir sayıyı kutulara bölmek, tüm değerleri son kutuya
yığar ve **ECE = 0.0000** gibi gurur verici bir sonuç üretir — yanında
Brier = 152 dururken. İkisi aynı anda doğru olamaz.

Bu yüzden script, ön koşulu sağlanmayan metriği raporlamak yerine **reddediyor**
ve nedenini yazıyor. Bir metriğin sessizce yanlış bir sayı üretmesi,
hesaplanmamasından daha tehlikelidir.

Eşiğin geçerliliği bundan etkilenmiyor: eşik yalnızca **monotonluk** ister
(skor arttıkça cevaplanabilir olma ihtimali artsın), skorun olasılık olmasını
değil. Kullanıcıya "%80 eminim" göstermek isteseydik, o zaman kalibre edilmiş
bir olasılığa ihtiyaç olurdu — örneğin BM25 skorunu Platt ölçekleme ile bir
olasılığa dönüştürerek.

---

## 4. Bu eşiğin sınırları

**52 negatif az.** Eşik, gözlenen negatif skorların dağılımına oturuyor; 52
örnekle o dağılımın kuyruğu gürültülü. Negatif sayısını artırmak, bu sayıyı
iyileştirmenin en ucuz yolu ve elle yazılabilir olduğu için model maliyeti
sıfır.

**Negatifler elle yazıldı, bu hem güç hem zayıflık.** Güç: korpusun onları
cevaplayamayacağından eminiz — üretilmiş bir negatif, yanlışlıkla korpusun
kapsadığı bir konuya düşüp metriği sessizce bozabilir. Zayıflık: yazarın
hayal gücüyle sınırlılar. Liste bu yüzden zorluk derecesine göre kademeli
(`qagen.FIXED_UNANSWERABLE`): kolay olanlar korpusla hiç kelime paylaşmaz,
zor olanlar yönetmelik diliyle yazılmıştır ve her iki kolda da yüksek skor
alır. Eşiği belirleyen, zor olanlardır.

**BM25 skoru sorgu uzunluğuna duyarlı.** Uzun sorgu daha çok terim eşleştirir,
daha yüksek skor alır. Pozitifler ile negatifler arasında sistematik bir
uzunluk farkı varsa, ölçülen ayrımın bir kısmı uzunluk farkıdır. Bu korpusta
ortalama soru uzunluğu 78 karakter ve iki sınıf birbirine yakın, ama bu
kontrol edilmedi — bir sonraki iterasyonda uzunluğa göre normalize edilmiş
bir varyant ölçülmeli.

**AUC 0.843 iyi değil, sadece kullanılabilir.** Kusursuz ayrım 1.0'dır.
Bu skor, cevaplanamaz soruların yaklaşık %38'ini (20/52) yanlışlıkla
cevaplanabilir sayıyor. Çekimserlik tek savunma hattı değil, olmamalı da:
üretim modeli de sistem mesajındaki "bilmiyorsan bilmiyorum de" talimatıyla
ikinci bir hat oluşturuyor — ve `docs/04` o hattın bağlam uzunluğuyla nasıl
çöktüğünü anlatıyor.

---

## 5. İlgili

- [04 — Üretim ve grounding](04-uretim-ve-grounding.md): ikinci savunma hattı
- [06 — Değerlendirmenin istatistiği](06-istatistik-degerlendirme.md): AUC,
  bootstrap ve permütasyon testinin arka planı
- `data/results/abstention_calibration.json`: bu dokümandaki her sayının
  ham çıktısı
