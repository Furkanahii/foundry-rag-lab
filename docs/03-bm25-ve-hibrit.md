# 3. BM25, IDF ve iki kolu birleştirmenin matematiği

Bu doküman `index/store.py` (BM25 kolu) ve `index/fusion.py` (birleştirme)
arkasındaki teoriyi açıklar.

---

## 3.1 BM25 nereden geliyor

BM25 sezgisel bir formül değil, **olasılıksal erişim çerçevesinden** (Probabilistic
Relevance Framework) türetilmiştir. Temel soru şudur:

$$ P(R = 1 \mid d, q) $$

"Bu belge $d$, bu sorgu $q$ için alakalı mıdır?" Belgeleri bu olasılığa göre
sıralamak istiyoruz. Olasılık yerine **olasılık oranını** (odds) sıralamak da
aynı sırayı verir ve matematiği kolaylaştırır:

$$ \frac{P(R=1 \mid d,q)}{P(R=0 \mid d,q)} $$

Bayes uygulayıp, terimlerin bağımsız olduğunu varsayarsak (Binary Independence
Model), skor terim başına katkıların toplamına dönüşür. Her terimin katkısı
**log-olasılık oranıdır** — ve işte IDF buradan çıkar:

$$ \text{IDF}(t) = \log \frac{N - n_t + 0.5}{n_t + 0.5} $$

burada $N$ toplam belge sayısı, $n_t$ terimi içeren belge sayısı.

**Bu bir sezgi değil, bir çıkarım.** "Nadir kelimeler daha bilgilendiricidir"
sezgisi doğrudur ama IDF'nin logaritmik biçimi tahmin edilmiş değil, olasılık
oranından türetilmiştir.

### Ölçtüğümüz IDF çöküşü

**Önce bir düzeltme.** Bu bölümün ilk hali şöyle diyordu: "`MADDE` 9 dokümanın
9'unda da var, dolayısıyla $n_t = N$ ve IDF sıfır veya negatif." İki hata
vardı ve ikisi de sunum öncesi kontrolde yakalandı:

1. `MADDE` dokuz belgenin **sekizinde** geçiyor. Dokuzuncu belge (pedagojik
   formasyon programı) İngilizce ve bu terimi içermiyor.
2. Daha önemlisi: **BM25 indeksi belgeler üzerinde değil, chunk'lar üzerinde
   kurulu.** `index/store.py` FTS5 tablosunu chunk başına bir satır olacak
   şekilde dolduruyor. Dolayısıyla IDF hesabında $N = 347$ (chunk sayısı),
   $n_t$ = terimi içeren chunk sayısıdır. Belge seviyesinde yapılan bir hesap
   indeksin gerçekte ne yaptığını tarif etmiyordu.

Doğru ölçüm, 347 chunk üzerinde:

| terim | $n_t$ | $\log(N/n_t)$ |
|---|---|---|
| öğrenci | 245 | **0,348** |
| madde | 179 | 0,662 |
| kayıt | 121 | 1,054 |
| tez | 41 | 2,136 |
| disiplin | 40 | 2,160 |
| burs | 24 | **2,671** |

IDF sıfıra düşmüyor, ama asıl mesele oran: yönetmelik kalıp sözcükleri
(`öğrenci`, `madde`) konu sözcüklerinin (`burs`, `disiplin`, `tez`) **dörtte
biri ile sekizde biri kadar** ağırlık taşıyor. Sorgu ağırlıklı olarak kalıp
sözcüklerden oluşuyorsa — ki bir öğrencinin doğal sorusu genelde öyledir —
BM25'in elindeki ayırt edici bilgi çok azalıyor.

Bu, sunumda anlatılacak temiz bir örnek: **BM25'in gücü nadir terimlerdedir;
sorgu yaygın terimlerden oluşuyorsa zayıflar.** Ve düzeltmenin kendisi de bir
ders: bir metriği hangi birim üzerinde hesapladığını bilmek, formülü doğru
yazmak kadar önemli.

---

## 3.2 Terim frekansı doygunluğu

BM25'in ikinci bileşeni, bir terimin belgede kaç kez geçtiğidir. Ama katkı
**doğrusal değildir**:

$$ \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)} $$

İki fikir var:

**Doygunluk ($k_1$).** Bir kelimenin 20 kez geçmesi, 10 kez geçmesinden iki kat
daha alakalı yapmaz. Formül $f \to \infty$ iken $k_1 + 1$'e yakınsar. Bu,
anahtar kelime doldurmayı (keyword stuffing) etkisiz kılar.

**Uzunluk normalizasyonu ($b$).** Uzun bir belgede bir kelimenin geçme olasılığı
zaten yüksektir. $|d| / \text{avgdl}$ terimi bunu düzeltir. $b = 0$ hiç
düzeltme yapmaz, $b = 1$ tam düzeltme yapar.

SQLite FTS5 bunu C'de uygular; biz yeniden yazmıyoruz, ama parametrelerin ne
yaptığını bilmek gerekiyor çünkü chunk uzunluğu dağılımımız (medyan 830
karakter) doğrudan $|d|/\text{avgdl}$ terimini etkiler.

---

## 3.3 Türkçe'nin BM25'e ettiği

BM25 **tam token eşleşmesine** dayanır. Türkçe sondan eklemeli bir dildir:

```
kayıt · kayıtlar · kayıtların · kayıtlardan · kaydı · kayıtlı
```

Bunların hepsi farklı token. Öğrenci "kayıt dondurma" yazar, yönetmelik
"kayıtların dondurulması" der → **BM25 sinyali sıfır.**

`ingest/normalize.py` içindeki `turkish_stem` bunu hafifletir. Ölçtüğümüz sonuç:

```
kayıtların dondurulması → ['kayit', 'dondurulm']
kayit dondurma          → ['kayit', 'dondurm']
                             ↑ eşleşti    ↑ eşleşmedi
```

**Çekim ekleri** (sondan gelen: -lar, -ın, -dan) kırpılabiliyor → `kayit`
eşleşti. **Türetim ekleri** (kelimenin ortasına giren edilgen `-ul-`)
kırpılamıyor → `dondurulm` ≠ `dondurm`.

Bu, sonek-kırpma yaklaşımının bilinen sınırıdır. Gerçek çözüm morfolojik
çözümleyicidir (Zemberek), ama o ağır bir Java bağımlılığı ve projenin
"her şey yerel ve hafif" iddiasını bozar. Ara çözüm olarak
`lexical_tokens(truncate=6)` gövdeyi sabit uzunluğa kırpar:

```
dondurulm[:6] = dondur
dondurm[:6]   = dondur   ← eşleşir
```

Bedeli kesinlik kaybı (daha çok yanlış eşleşme). Hangisinin kazandığı korpusa
bağlıdır → **ölçülecek parametre**, varsayılacak değil.

Ayrıca bir tuzak daha: Python'un `lower()` Türkçe bilmez.

```
"IŞIK".lower()      → "işik"     ✗ (doğrusu "ışık")
"İSTANBUL".lower()  → "i̇stanbul"  ✗ (birleşik nokta kalıyor)
```

Bu düzeltilmezse indeks, hiçbir sorgunun üretemeyeceği token'lar saklar.
`turkish_lower` önce İ→i ve I→ı eşlemesini yapar, sonra genel `lower()`'ı
uygular.

---

## 3.4 Neden iki kol birden

Dense ve BM25'in **hata modları farklıdır**:

| | Dense (yoğun) | BM25 (sözcüksel) |
|---|---|---|
| Güçlü olduğu | Başka kelimelerle aynı anlam | Tam eşleşme: kod, tarih, özel isim |
| Başarısız olduğu | "CMPE 150" gibi tanımlayıcılar | Yaygın terimler (IDF ≈ 0), paraf |

Kendi ölçümümüz:

```
SORU: "CMPE 150"
  DENSE  : 0.368 · 0.308 · 0.304    ← marj yok
  LEXICAL: 0.998                     ← tam isabet
```

Hataların **bağımsız** olması, birleştirmenin tek tek her koldan iyi olması için
gereken koşuldur. (Tamamen bağımlı olsalardı, ikinci kol hiçbir yeni bilgi
katmazdı.)

---

## 3.5 Skorları toplayamayız — RRF

Bariz yaklaşım — iki skoru toplamak — **yanlıştır**, çünkü ölçekleri
karşılaştırılamaz:

- Dense: $[-1, 1]$ aralığında kosinüs, pratikte 0.3–0.6 bandında sıkışık
- BM25: sınırsız, korpus büyüklüğüne ve terim nadirliğine bağlı

Yukarıdaki örnekte aynı doğru chunk için BM25 = 0.998, dense = 0.368. Bunları
ortalamak anlamsız.

**Reciprocal Rank Fusion** skorları tamamen atar, sadece **sırayı** kullanır:

$$ \text{RRF}(d) = \sum_{\text{kollar}} \frac{1}{k + r_i(d)} $$

Sıralar **ordinal**dir, yani doğası gereği karşılaştırılabilir. Kalibrasyon
problemini çözmeye çalışmak yerine **tamamen ortadan kaldırır**.

**$1/(k+r)$ biçimi neden?** Dik biçimde azalır: 1.→2. sıra düşüşü, 20.→21.
düşüşünden çok daha pahalıdır. Bu, alakanın sırayla nasıl azaldığına uyar.

**$k$ ne yapar?** ($k = 60$, Cormack ve ark. 2009'dan gelen konvansiyon)

- $k \to 0$: ilk sıra her şeyi alır, tek kol domine eder
- $k \to \infty$: tüm sıralar eşitlenir, füzyon "kaç kol bunu getirdi?" saymaya indirgenir

**Kritik davranış:** İki kolun da *orta* sırada getirdiği bir belge, tek kolun
1. sıraya koyup diğerinin hiç getirmediği bir belgeyi geçebilir. Bu **uzlaşma
arayan** davranıştır ve hata modları bağımsızken tam istediğimiz şeydir.

### Sosyal seçim bağlantısı

RRF, **Borda sayımı** tarzı bir sıra birleştirme yöntemidir. Birden fazla
sıralayıcıyı birleştirmek, biçimsel olarak birden fazla seçmenin tercihlerini
birleştirmekle **aynı problemdir**. Dolayısıyla aynı imkânsızlık sonuçlarını
miras alır — Arrow teoremi burada da geçerlidir: tüm makul aksiyomları aynı
anda sağlayan bir birleştirme yöntemi yoktur. Bu, RRF'nin "en iyi" değil,
"makul bir taviz" olduğunu söyler.

---

## 3.6 Ağırlıklı füzyon ve zayıflığı

Alternatif: skorları $[0,1]$'e min-max normalize edip konveks birleşim al:

$$ s(d) = \alpha \cdot \hat{s}_{\text{dense}}(d) + (1-\alpha) \cdot \hat{s}_{\text{bm25}}(d) $$

Avantajı: büyüklük bilgisini korur (1. ile 2. arasındaki uçurum görünür).

**Gizli tuzağı:** min-max normalizasyon **getirilen adaylar üzerinden**
hesaplanır, tüm korpus üzerinden değil. `dense_top_k`'yı değiştirdiğinizde tüm
normalize skorlar değişir. Yani bir aday derinliğinde ayarlanan $\alpha$
başka bir derinliğe **taşınmaz**. RRF'de böyle bir bağımlılık yoktur.

$\alpha = 1$ ve $\alpha = 0$ aynı zamanda tek-kol temel çizgilerini
(`dense_only`, `lexical_only`) verir — bu yüzden dört strateji de aynı kod
yolundan geçer ve normalizasyondaki bir hata temel çizgide saklanamaz.

### 3.6.1 Teori RRF diyor, ölçüm weighted dedi

Yukarıdaki iki bölüm okunduğunda beklenen sonuç RRF'nin kazanmasıdır: ölçekten
bağımsız, aday derinliğine bağlı değil, kalibrasyon problemi yok. Projenin
varsayılanı da bu gerekçeyle RRF idi.

Ölçüm başka söyledi. 60 üretilmiş + 60 parafraz soruluk sette, Holm-Bonferroni
düzeltmesinden sonra ayakta kalan **tek** erişim kazancı ağırlıklı füzyon oldu,
üstelik her iki kelime varyantında da:

| strateji | üretilmiş | parafraz | temele karşı |
|---|---|---|---|
| weighted-0.5 | 0.7370 | 0.6295 | +0.157 (p=0.0006) ✓ / +0.114 (p=0.0002) ✓ |
| rrf | 0.6817 | 0.6130 | +0.102 (p=0.026) / +0.097 (p=0.0088) |

Neden şaşırtıcı değil, sadece beklenmedik: RRF, büyüklük bilgisini **atarak**
kalibrasyon problemini çözüyor. Bu bir taviz — ve bu korpusta atılan bilgi işe
yarar bilgiymiş. Yönetmelik sorgularında birinci ile ikinci aday arasındaki
uçurum genelde gerçek bir fark; sıraya indirgemek onu siliyor.

**Ama §3.6'daki tuzak ortadan kalkmadı.** `dense_top_k` değişirse $\alpha$
yeniden ayarlanmalı, çünkü normalizasyon aday derinliğine bağlı. Yani bu bir
korpus ve derinlik özelinde ölçülmüş seçim, evrensel bir sonuç değil — RRF'nin
teorik üstünlüğü hâlâ geçerli, sadece bu ölçekte karşılığını vermiyor.

Aynı normalizasyon, çekimserlik sinyali olarak füzyon skorunun neden
kullanılamayacağının da sebebi:
[05 — Kalibrasyon ve çekimserlik](05-kalibrasyon-abstention.md).

---

## 3.7 MMR: çeşitlilik

Üst üste binen chunk'larla, en alakalı 5 sonuç çoğu zaman **aynı pasajın beş
kopyasıdır**. Bu, üreticinin tüm bağlam penceresini tek bir olguya harcar.

Maximal Marginal Relevance, her adımda şunu maksimize eden adayı seçer:

$$ \lambda \cdot \text{sim}(d, q) - (1-\lambda) \max_{s \in S} \text{sim}(d, s) $$

$\lambda = 1$ saf alaka, $\lambda = 0$ saf çeşitlilik.

**Açgözlü ama teorik garantili:** Kesin problem NP-zordur (maksimum dağılım
problemini içerir). Ancak amaç fonksiyonu **submodülerdir**, dolayısıyla açgözlü
çözüm standart $(1 - 1/e) \approx 0.63$ yaklaşım garantisini taşır. $k=5$ için
bu fazlasıyla yeterlidir.

---

## Özet

| Karar | Matematiksel gerekçe |
|---|---|
| BM25 kullanmak | Olasılıksal alaka çerçevesinden türer, IDF = log-olasılık oranı |
| Hibrit | Hata modları bağımsız → füzyon her iki koldan iyi |
| Skor toplamamak | Ölçekler karşılaştırılamaz (0.998 vs 0.368) |
| RRF | Sıra ordinal → kalibrasyon problemi ortadan kalkar |
| **Varsayılan: weighted-0.5** | **Teori RRF'i tercih etti, ölçüm etmedi (§3.6.1)** |
| $k = 60$ | Üst sıra baskınlığı ile uzlaşma arasında denge |
| Türkçe gövdeleme | Sondan eklemeli dilde tam eşleşme çalışmaz |
| MMR açgözlü | Submodüler → $(1-1/e)$ garantisi |
