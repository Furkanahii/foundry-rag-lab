# 2. Chunking: en çok göz ardı edilen karar

`ingest/chunkers.py` içindeki dört stratejinin gerekçesi.

---

## 2.1 Neden en kritik karar

Chunking **her şeyin yukarısındadır**. İki parçaya bölünmüş bir olgu, gömme
modeli ne kadar iyi olursa olsun, reranker ne kadar akıllı olursa olsun **asla
bütün olarak getirilemez**. Recall'un tavanını chunking belirler.

Proje planı bunu tek satırla geçiyor ("paragraflara böl"). Burada dört
uygulaması olan birinci sınıf bir değişken — çünkü hangisinin kazandığı
korpusa bağlıdır ve iddia edilecek değil, ölçülecek bir şeydir.

---

## 2.2 Sabit pencere (`fixed`) ve örtüşmenin matematiği

Sabit karakter penceresi, adım aralığı `chunk_size - overlap`.

**Örtüşme neden var:** Bir olguyu ortadan bölme riskine karşı sigorta.
Adım aralığı $s = c - o$ olduğunda, uzunluğu $o$'dan kısa **her metin parçası
en az bir chunk içinde bütün olarak bulunur**. Garanti budur.

Bu aynı zamanda `overlap`'ın nasıl seçileceğini söyler: **kaybetmeyi göze
alamayacağın en uzun olgunun uzunluğu kadar.**

**Bedeli depolama:** Toplam indekslenen metin kabaca
$\frac{c}{c-o}$ kat artar. 900/150 için $\approx 1.2\times$.

---

## 2.3 Özyinelemeli (`recursive`) — varsayılan

En güçlü ayırıcıdan en zayıfa doğru inen bir merdiven; sadece parça hâlâ
büyükse bir alt basamağa iner:

```
MADDE \d+  →  numaralı başlık  →  paragraf  →  satır  →  cümle  →  kelime
```

**En değerli satır ilki.** Türk yönetmelikleri "MADDE 7 -" yapısındadır ve her
madde **kendi içinde tam bir kuraldır** — tam olarak bir sorunun sorulduğu
birim. Orada bölmek, bu dosyadaki en yüksek getirili alan uyarlamasıdır.

Ölçüm: 9 dokümanlık korpusta 347 chunk, uzunluk dağılımı medyan 830 / hedef 900.
Chunker hedefine oturuyor.

**Geriye örtüşme:** Yapıya göre bölmek bile çapraz referansı kaybettirir
("bu sürede", "yukarıdaki madde"). Küçük bir geriye örtüşme öncülü geri getirir,
sabit pencere örtüşmesinin depolama maliyeti olmadan.

---

## 2.4 Cümle penceresi (`sentence_window`)

Sabit chunking'in birbirine bağladığı iki şeyi ayırır:

- **Eşleştirdiğin birim**: tek cümle → keskin gömme hedefi, seyreltme yok
- **Okuduğun birim**: cümle + komşuları → üretici için yeterli bağlam

Bedeli indeks boyutu: paragraf başına değil cümle başına bir vektör.

Not: bu stratejide `_merge_short_chunks` **çağrılmaz**. Kısa cümleler tam da
keskin indekslemek istediğimiz şeydir; eksik bağlamı okuma anında pencere
sağlar.

---

## 2.5 Anlamsal (`semantic`) ve yüzdelik eşiğin inceliği

Yöntem: her cümleyi göm, ardışık cümleler arasındaki kosinüs **mesafesini**
hesapla, sıçramalarda böl.

$$ d_i = 1 - (v_i \cdot v_{i+1}) $$

**Kritik tasarım kararı: eşik sabit sayı değil, yüzdeliktir.**

Sebebi doğrudan [01-vektorler-ve-benzerlik.md](01-vektorler-ve-benzerlik.md)'deki
konsantrasyon bulgusudur: mutlak kosinüs değerleri dokümanlar ve diller arasında
karşılaştırılabilir değil — hepsi dar bir bantta. Ama bir dokümanın **kendi
mesafe dağılımının şekli** kendine göre anlamlıdır.

90. yüzdelik demek: *"bu dokümanın en ani %10'luk geçişlerinde böl"*. Bu,
dokümanlar arası taşınır. `mesafe > 0.35` taşınmaz.

**Maliyet:** İndeksleme sırasında cümle başına bir gömme çağrısı. Açık ara en
pahalı strateji. Bu maliyeti hak edip etmediği ampirik bir sorudur — benchmark
tam olarak bunu cevaplar.

---

## 2.6 `min_chunk_chars`: ampirik alt sınır

Bu parametre keyfi değil, ölçümden geldi.

Bu makinede çok kısa metinler, alakasız olanların birbirine ~0.62 kosinüs
verdiği **dejenere bir bölgeye** gömülüyor. Aynı model tam cümleleri temiz
ayırıyor (doğru 0.64 vs alakasız 0.27).

Sonuç: küçük parçalar sadece bilgisiz değil, **aktif olarak zararlıdır** — her
sonuç listesine yüksek skorlu gürültü enjekte ederler. Eşiğin altındakiler tek
başına indekslenmez, komşusuna birleştirilir.

**Geriye birleştirme** tercih edildi: yönetmeliğin sonundaki "Yürürlük" gibi bir
kalıntı, ait olduğu metne yapışsın; kendi başına gürültülü bir vektör olarak
öksüz kalmasın.

---

## 2.7 Türkçe'ye özgü tuzaklar

**Cümle bölme.** Naif `split(".")` yönetmelik metnini paramparça eder. Nokta şu
durumlarda cümle bitirmez:

- kısaltmalar: `vb.`, `Dr.`, `md.`, `bkz.`, `Öğr.`, `Gör.`
- sıra sayıları: `1. madde`, `2019. yıl`
- baş harfler: `M. Kemal`

Test sonucumuz — hiçbirinde bölmedi, doğru yerlerde böldü:

```
"MADDE 7 - ... Öğr. Gör. Dr. Ayşe Yılmaz'a yapılır." → tek cümle ✓
"Bu süre 2. yarıyıl sonuna kadardır."               → tek cümle ✓
```

**Karakter uzunluğu yanlılığı.** `chunk_size` karakter cinsindendir çünkü
sunulan modele birebir uyan bir tokenizer garantimiz yok. Ama Türkçe kelimeler
İngilizce'den ortalama daha uzundur (sondan eklemeli), dolayısıyla **sabit
karakter bütçesi Türkçe'de daha az kelime tutar**. İki dilli karşılaştırma
yapılırken bu bilinmeli.

**PDF kelime kopukluğu.** Çıkarım `Yönetm eliği` üretiyor. Onarım sözlük
kanıtına dayanır: birleşik biçim korpusta gerçek bir kelime olarak geçiyorsa ve
parça geçmiyorsa birleştir. Ölçüm: 21 gerçek onarım, sıfır yanlış birleştirme.
Muhafazakâr — uydurmak yerine kanıt istiyor.

---

## Özet

| Strateji | Güçlü yanı | Maliyeti |
|---|---|---|
| `fixed` | Basit, öngörülebilir, örtüşme garantili | Yapıyı yok sayar |
| `recursive` | MADDE sınırlarını tanır (varsayılan) | Ayırıcı listesi elle ayarlı |
| `sentence_window` | Keskin eşleşme + geniş okuma | İndeks birkaç kat büyür |
| `semantic` | Konu değişiminde böler | Cümle başına gömme çağrısı |
