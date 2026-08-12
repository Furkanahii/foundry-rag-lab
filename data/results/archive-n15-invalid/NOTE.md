# Neden bu sonuçlar geçersiz (1 Ağustos 2026 koşusu)

Bu klasördeki 11 sonuç dosyası projenin ilk benchmark koşusundan kalma.
Silinmediler çünkü bir ölçümün neden çöpe gittiğini göstermek, ölçümün
kendisi kadar öğretici. Ama **hiçbiri alıntılanmamalı**. İki bağımsız
kusur var, ikisi de sonuçları yorumlanamaz kılıyor.

## 1. Yetersiz istatistiksel güç

`n_queries: 15`. Eşleştirilmiş bir tasarımda %80 güçle orta büyüklükte bir
etkiyi (Cohen's d = 0.5) yakalamak için 32 soru gerekiyor; küçük bir etki
(d = 0.2) için 197. Onbeş soruyla yalnızca *büyük* etkiler görünür hale
gelir.

Sonuç: on karşılaştırmanın hiçbiri Holm-Bonferroni düzeltmesinden geçemedi.
Beşi ham p < 0.05 verdi ama düzeltme sonrası hepsi düştü. Bu, "konfigürasyonlar
eşdeğer" demek değil — "bu eval seti ayırt edemedi" demek. İkisini birbirine
karıştırmak, güçsüz bir çalışmayı boş bir sonuç gibi raporlamaktır.

Kök neden koddaydı: `generate_questions(n_questions=30)` otuz chunk örnekliyor
ve elenenlerden sonra geriye kalanı döndürüyordu. İstenen sayı ile elde edilen
sayı arasındaki fark hiçbir yerde uyarı üretmiyordu. Artık `--n` *kullanılabilir*
soru hedefi; havuzdan hedefe ulaşana kadar çekiliyor.

## 2. Sorular kaynak chunk'ın kelimelerini taşıyordu

Sorular chunk'lardan üretildiği için chunk'ın söz varlığını yeniden kullanıyorlar.
BM25 tam da bu örtüşmeyi puanlar. Ölçülen sonuç:

    lexical-only    nDCG 0.8595
    baseline-dense  nDCG 0.5766
    lexical-nostem  nDCG 0.4595

Bu tablo "Türkçede BM25 dense'i eziyor" demiyor; "sorularımız belgelerin
kopyasıydı" diyor. Gövdeleme etkisi (0.8595 vs 0.4595) muhtemelen gerçek,
çünkü her iki kol da aynı avantajdan yararlanıyor — ama lexical/dense
karşılaştırması kurgunun kendisiyle kirlenmiş.

Artık her soru için gold chunk ile kelime örtüşmesi (`meta.gold_overlap`,
BM25'in gördüğü aynı gövdelemeyle hesaplanır) kaydediliyor, ve benchmark
üretilmiş ve parafrazlanmış varyantları **ayrı ayrı** koşup aradaki düşüşü
eşleştirilmiş testle raporluyor. Yani bu kusur artık ölçülen bir büyüklük.

Geçerli sonuçlar bir üst klasörde `generated/` ve `paraphrased/` altında.
