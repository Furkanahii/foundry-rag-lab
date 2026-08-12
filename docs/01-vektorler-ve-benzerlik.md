# 1. Vektörler, benzerlik ve yüksek boyutun tuhaflığı

Bu doküman, `runtime/foundry.py` ve `index/store.py` içindeki matematiği
açıklar. Amaç: sunumda "kosinüs benzerliği kullandık" demek yerine *neden* ve
*hangi tuzaklarla* kullandığımızı anlatabilmek.

---

## 1.1 Gömme (embedding) nedir

Bir gömme modeli, metni sabit uzunlukta bir gerçel sayı vektörüne eşler:

$$ E: \text{metin} \rightarrow \mathbb{R}^d $$

Bizim durumumuzda `qwen3-embedding-0.6b` ile $d = 1024$. Modelin eğitim hedefi
şudur: anlamca yakın metinler uzayda birbirine yakın düşsün.

Kritik nokta: bu bir *öğrenilmiş* eşleme. "Yakınlık" modelin eğitim verisinden
öğrendiği bir şey, evrensel bir gerçek değil. Türkçe için ne kadar iyi olduğu
ampirik bir soru — bu yüzden ölçüyoruz.

---

## 1.2 Neden kosinüs, neden normalize ediyoruz

İki vektör arasındaki kosinüs benzerliği:

$$ \cos(a,b) = \frac{a \cdot b}{\|a\| \, \|b\|} $$

Bu, iki vektör arasındaki **açıyı** ölçer, uzunluğu değil. Metin gömmelerinde
vektör uzunluğu genellikle metnin uzunluğuyla ilişkilidir, anlamıyla değil —
uzun bir paragraf ile kısa bir cümle aynı şeyi söylüyorsa açıları küçük olmalı,
büyüklükleri farklı olsa da. Kosinüs tam olarak bunu yapar.

**Normalizasyon numarası.** Vektörleri kaydetmeden önce birim uzunluğa
getiriyoruz ($\|a\| = 1$). O zaman:

$$ \cos(a,b) = a \cdot b $$

Yani kosinüs, basit bir iç çarpıma indirgeniyor. Bunun üç faydası var:

1. **Hız.** Tüm chunk'lara karşı arama tek bir matris çarpımı olur:
   `Q @ D.T`. Bu BLAS'a iner, Python döngüsünden ~100 kat hızlıdır.

2. **L2 mesafesi kosinüsle aynı sırayı verir.** Birim vektörler için:

   $$ \|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2(a \cdot b) = 2 - 2(a \cdot b) $$

   Sağ taraf, iç çarpımın **monoton azalan** bir fonksiyonu. Yani L2'ye göre
   artan sırada sıralamak, kosinüse göre azalan sırada sıralamakla **birebir
   aynı sonucu** verir. Bu sayede `sqlite-vec`'in yerel L2 mesafesini
   kullanabiliyoruz ve yine de kosinüs üzerinden düşünmeye devam edebiliyoruz.

3. **Sıfır vektör güvenliği.** `l2_normalize` içinde `eps` koruması var. Model
   boş girdi için sıfır vektör üretebilir; normalize ederken sıfıra bölme NaN
   üretir ve o NaN tüm benzerlik hesabını zehirler.

---

## 1.3 Boyutun laneti: ölçtüğümüz olgu

Yüksek boyutlu uzayda rastgele iki vektör arasındaki mesafeler **birbirine
yakınsar**. Sezgisel açıklama: $d$ boyutta iç çarpım $d$ tane bağımsız terimin
toplamıdır; merkezi limit teoremi gereği bu toplam dar bir bandda toplanır.

Rastgele birim vektörler için iç çarpımın standart sapması yaklaşık
$1/\sqrt{d}$ ile ölçeklenir. $d = 1024$ için bu $\approx 0.031$ — yani
**tamamen rastgele iki vektör bile 0'dan ancak ~0.03 uzaklaşır**.

Bu, gerçek gömmelerde şu şekilde görünür (kendi ölçümümüz):

```
cos("merhaba dünya", "hello world")    = 0.7040   ← aynı anlam
cos("merhaba dünya", "kayıt dondurma") = 0.6203   ← alakasız
```

Aradaki fark sadece **0.08**. Yani gömme uzayı 0'ın etrafında değil, **yüksek
bir taban değerin etrafında** yoğunlaşmış. Bunun iki pratik sonucu var:

**Sonuç 1: Mutlak eşik anlamsızdır.** "Benzerlik > 0.7 ise alakalıdır" kuralı
bu uzayda çalışmaz, çünkü taban zaten 0.62. Eşik korpusa, dile, hatta metin
uzunluğuna göre kayar. Bu yüzden `generation.abstention_threshold` varsayılan
olarak *kalibre edilmemiş* bırakılmıştır ve `evaluation/stats.py` içindeki
`find_best_threshold` ile veriden öğrenilir.

**Sonuç 2: Sıralama, skordan daha güvenilirdir.** RRF'nin skorları atıp sadece
sıraları kullanması tam da bu yüzden. Bkz. [03-bm25-ve-hibrit.md](03-bm25-ve-hibrit.md).

### Kısa metin tuzağı

Daha ince bir bulgu: konsantrasyon **kısa metinlerde çok daha şiddetli**. Uzun
ve içerikli cümlelerde ayrım sağlıklı:

```
soru: "kayıt dondurmak için hangi tarihe kadar başvurmalı?"
  0.6388  Kayıt dondurma başvurusu en geç ders ekleme-bırakma...   ← doğru
  0.2727  Mezuniyet için gereken toplam kredi 240 AKTS'dir.
  0.1970  Kütüphane hafta içi 08:00-22:00 arasında açıktır.
```

Burada fark 0.64 vs 0.27 — gayet net. Sebep: kısa metinde model yeterli sinyal
alamaz ve çıktısı "ortalama" bir bölgeye düşer. Bu, `ChunkConfig.min_chunk_chars`
parametresinin ampirik gerekçesidir: eşiğin altındaki parçalar tek başına
indekslenmez, komşusuna birleştirilir. Aksi halde her sonuç listesine
yüksek skorlu gürültü enjekte ederler.

---

## 1.4 Johnson–Lindenstrauss: boyut indirgeme neden çalışır

Merak edilen bir soru: 1024 boyuta gerçekten ihtiyaç var mı?

**Johnson–Lindenstrauss lemması** şunu söyler: $n$ nokta, $\varepsilon$ hata
payıyla, şu boyuta indirgenebilir ve tüm ikili mesafeler korunur:

$$ k = O\left(\frac{\log n}{\varepsilon^2}\right) $$

Dikkat edilecek şey: $k$, **orijinal boyuta ($d$) hiç bağlı değil** — sadece
nokta sayısına ($n$) ve kabul edilen hataya bağlı. 347 chunk için
$\log(347) \approx 5.85$; $\varepsilon = 0.1$ için $k$ birkaç yüz mertebesinde
çıkar. Yani 1024 boyut bizim korpusumuz için fazlasıyla bol.

Pratik yansıması: **Matryoshka** gömmeler (vektörün ilk $k$ boyutunu kesip
kullanmak) teorik olarak bu lemmaya dayanır. Bellek/hız için boyut kesmek
mantıklı bir deneydir ve bu projede ölçülebilir bir parametre olarak durur.

---

## 1.5 Asimetrik gömme: soru ile belge aynı şey değil

Bir soru ile onu yanıtlayan paragraf **aynı şeyi söylemez**. Soru
"kaç kredi lazım?" der, belge "toplam 240 AKTS'dir" der. Ortak kelime bile
olmayabilir.

`qwen3-embedding` modelleri bu asimetriyi hesaba katacak şekilde, sorgu tarafına
bir **talimat öneki** ile eğitilmiştir:

```
Instruct: Given a question, retrieve passages that answer the question
Query: {soru}
```

Çıplak soru vermek, modeli eğitildiği şablonun dışında kullanmak demektir.
`retrieve/rerank.py` içindeki `apply_query_instruction` bunu uygular ve
`RetrievalConfig.query_instruction` ile açılıp kapatılabilir — çünkü gerçekten
işe yarayıp yaramadığı ölçülmesi gereken bir şey, varsayılması gereken değil.

---

## Özet

| Karar | Gerekçe |
|---|---|
| Kosinüs kullanmak | Uzunluk değil açı önemli |
| Birim normalize etmek | Tek matris çarpımı + L2 ile aynı sıralama |
| `eps` koruması | Sıfır vektör NaN üretmesin |
| Mutlak eşik kullanmamak | Konsantrasyon etkisi ölçüldü (0.62 taban) |
| `min_chunk_chars` alt sınırı | Kısa metinde geometri bozuluyor |
| Sıra bazlı füzyon | Skor ölçekleri güvenilmez |
| Talimat öneki (ölçülecek) | Model asimetrik eğitilmiş |
