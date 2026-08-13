# Sunum anlatımı

Bu doküman projeyi sözlü olarak nasıl anlatacağını tarif ediyor: ne açacaksın,
ne söyleyeceksin, ekranda neyi göstereceksin, ve hangi soru geldiğinde nereye
bakacaksın.

Hedef süre **20 dakika + soru-cevap**. Kısaltman gerekirse §5 (canlı demo) ve
§7 (sınırlar) korunur, §4'ün alt başlıkları kısalır — çünkü demo ve dürüstlük
sunumun omurgası, bulgu listesi ise doldurulabilir hacim.

---

## 0. Sunumdan önce — hazırlık listesi

Bunları **dinleyici salona girmeden önce** yap. Hepsinin bir nedeni var,
nedenleri de yazdım çünkü bir şey ters giderse neyi kurtaracağını bilmen lazım.

**1. Streamlit'i başlat ve modeli ısıt.**

```bash
cd ~/Desktop/foundry-rag-lab
PYTHONPATH=src ./.venv/bin/streamlit run app/dashboard.py
```

Açılınca **bir soru sor ve cevabın gelmesini bekle.** Bu adım isteğe bağlı
değil. İlk sorgu modeli belleğe yüklüyor ve bu tek başına 1,5–2 dakika sürüyor.
Sunum sırasında ilk soruyu sorup iki dakika sessiz beklemek, demonun bütün
etkisini öldürür. Isıttıktan sonra tarayıcı sekmesini açık bırak: Streamlit'in
`@st.cache_resource` dekoratörü modeli süreç boyunca bellekte tutuyor, yani
sekmeyi kapatmadığın sürece bir daha yüklenmez.

**2. Kenar çubuğundaki "Üretim modeli" seçicisine sunum boyunca dokunma.**
Model değiştirmek yeni bir model yüklemesi demek — yine dakikalar. Karşılaştırma
yapmak istersen ölçülmüş tabloyu (§4.4) göster, canlı deneme yapma.

**3. İkinci bir terminal sekmesi aç**, içinde proje dizininde bekle. Benchmark
çıktısını göstereceksen komutu önceden yazıp Enter'a basmadan bırak.

**4. Şu üç sekmeyi tarayıcıda hazır et:**
- Streamlit dashboard (ısıtılmış)
- GitHub deposu: https://github.com/Furkanahii/foundry-rag-lab
- `data/results/archive-n15-invalid/NOTE.md` (GitHub üzerinde) — §6 için

**5. Yedek plan.** Demo çökerse: dashboard'un ekran görüntülerini önceden al,
telefonunda ya da slaytta dursun. Canlı demo çökerse toparlanmanın tek yolu
"zaten kaydettiğim çıktı şu" diyebilmektir. İnternet **gerekmiyor**, bu bir
avantaj: wifi çökerse sistem çalışmaya devam eder ve bunu yüksek sesle söyle.

**6. Wifi'yi kapat.** Ciddiyim. Sunumun en güçlü tek anı, birinin "bu gerçekten
yerel mi?" diye sorması ve senin wifi'nin kapalı olduğunu göstermen. Bunu
baştan yapıp §1'de bir cümleyle belirtmek, sonradan savunmaya çalışmaktan çok
daha etkili.

---

## 1. Açılış (1,5 dakika) — ekranda: dashboard ana ekranı

> "Bana verilen konu şuydu: Foundry Local ile ilk yerel RAG uygulamanı yap.
> Verilen plan da netti — metni parçala, göm, SQLite'a yaz, Python'da tek tek
> kosinüs benzerliği hesapla, en iyi iki parçayı modele ver.
>
> Ben o planı bir hafta sonunda bitirdim ve sonra şunu fark ettim: o sistem
> sekiz cümlelik bir örnekte çalışıyor, gerçek bir Türkçe belge kümesinde
> çalışmıyor. Bu yüzden asıl projeyi, planın *çalışmadığı yerleri ölçen* bir
> sisteme çevirdim.
>
> Bugün göstereceğim şey bir RAG demosu değil, bir RAG laboratuvarı. Her tasarım
> kararının arkasında bir ölçüm var, ve o ölçümü yapan araçlar sistemin
> içinde. Tamamen bu makinenin üzerinde çalışıyor — şu anda wifi kapalı."

Ekranda dashboard'un üstündeki dört metrik kutusu duruyor: 9 doküman, 347 chunk,
1024 boyutlu vektörler, 4,63 MB indeks. Bunları göstererek:

> "Korpus, Boğaziçi'nin kamuya açık yönetmelikleri: kayıt, burs, disiplin,
> özel öğrenci, konut tahsisi, lisans ve lisansüstü eğitim-öğretim. Dokuz belge,
> yaklaşık 215 bin karakter."

**Neden bu açılış:** İlk otuz saniyede "verilen işi yaptım" ile "verilen işin
sınırlarını buldum" arasındaki farkı kuruyorsun. Sunumun geri kalanı bu farkı
haklı çıkarmak üzerine.

---

## 2. Problem: Türkçe yönetmelik neden zor (3 dakika) — ekranda: slayt veya terminal

Burada tek bir hikâye anlat, liste sayma. En iyi hikâye şu:

> "Sisteme sorulacak ilk soruyu kendim yazdım: 'kayıt dondurma nasıl yapılır'.
> Sistem alakasız şeyler getirdi. Modeli suçlamadan önce korpusta arama yaptım —
> **'kayıt dondurma' ifadesi dokuz belgenin hiçbirinde geçmiyor.** Yönetmelik
> buna 'izinli sayılma' diyor.
>
> Yani sorun erişimde değildi. Sorun, benim kelime dağarcığımla belgenin kelime
> dağarcığının aynı olmamasıydı. Ve bu, RAG'in gerçek problemi: kullanıcı ile
> belge aynı şeyi farklı kelimelerle söylüyor."

Sonra üç teknik bulguyu hızlıca, her birini tek cümleyle geç:

- **Türkçe `lower()` bozuk.** Python'da `"IŞIK".lower()` → `"işik"`. Noktasız I
  ile noktalı i'yi karıştırıyor, çünkü Unicode varsayılanı İngilizce. Sözlüksel
  indeks bu yüzden dile özgü normalizasyon istiyor.
- **Çekim ekleri eşleşmiyor.** "kayıtların" ile "kayıt" BM25 için iki ayrı
  kelime. Türkçe eklemeli bir dil; bir kök arka arkaya beş ek taşıyabiliyor.
- **`MADDE` kelimesi dokuz belgenin hepsinde, 195 kez geçiyor.** BM25'in IDF
  terimi `log(N/df)`, burada `N = df = 9`, yani IDF sıfır. Yönetmelik metninin
  en karakteristik kelimesi sıfır bilgi taşıyor.

> "Bu üçü birlikte şunu söylüyor: tek başına sözlüksel arama Türkçe'de
> yetmiyor. Ama sadece anlamsal arama da yetmiyor — çünkü biri 'CMPE 150' diye
> sorduğunda tam eşleşme lazım, ve gömme vektörleri kod numaralarında zayıf.
> Ölçtüm: tam eşleşme gereken sorgularda yoğun aramanın marjı 0,368'e karşı
> 0,304, yani doğru pasajı yanlıştan neredeyse ayıramıyor.
>
> Cevap ikisini birleştirmek. Ama 'birleştirdim, daha iyi oldu' demek yetmez —
> ne kadar iyi olduğunu ölçmek gerekiyor. Projenin yarısı bu ölçüm."

---

## 3. Mimari (3 dakika) — ekranda: README'deki mimari şeması

Şemayı göster, akışı **soru üzerinden** anlat, kutu kutu değil:

> "Bir soru geldiğinde iki şey aynı anda oluyor. Soru bir vektöre gömülüyor ve
> sqlite-vec üzerinde en yakın 30 komşu bulunuyor — bu anlamsal kol. Aynı soru
> Türkçe'ye özgü normalizasyondan ve gövdelemeden geçip BM25 indeksinde
> aranıyor — bu sözlüksel kol.
>
> İki liste RRF ile birleşiyor: her belge, her iki listedeki sırasının tersinden
> puan alıyor. Skorları değil sıraları toplamanın nedeni şu — BM25 skoru
> sınırsız bir sayı, kosinüs benzerliği ise -1 ile 1 arasında. Bunları doğrudan
> toplamak elmayla armut toplamak olur. Sıra ise iki kolda da aynı anlama
> geliyor.
>
> Sonra ilk 5 pasaj üretim modeline gidiyor ve model cevabı **sadece o
> pasajlardan** üretiyor, her cümleyi kaynakla numaralandırarak."

Vurgulanacak tek mimari karar:

> "Foundry Local'a dokunan tek modül `runtime/foundry.py`. Bunun sebebi mimari
> zarafet değil, ölçüm: model karşılaştırması yapabilmem için modeli tek bir
> config alanından değiştirebilmem gerekiyordu. Bu izolasyon olmasaydı
> laboratuvarın yarısı mümkün olmazdı."

---

## 4. Ölçümler (5 dakika) — sunumun en değerli kısmı

Dört bulgu, her biri "beklenti → ölçüm → karar" kalıbında. Kalıba sadık kal:
dinleyici üçüncüsünde ritmi tanıyor ve seni daha kolay takip ediyor.

### 4.1 Bağlam bütçesi — sezginin tersi

> "RAG anlatılarının hepsi 'daha çok bağlam, daha iyi cevap' varsayar. Ben de
> öyle varsaydım. `scripts/prompt_ablation.py` ile bağlam boyutunu taradım."

| bağlam | doğru reddetme | doğru cevap | gecikme |
|---|---|---|---|
| **1200 karakter** | **2/3** | **2/2** | 38,6 s |
| 2400 karakter | 1/3 | 1/2 | 59,6 s |
| 4000 karakter | 0/3 | 0/2 | 57,2 s |
| 6000 karakter | 0/3 | 1/2 | 93,9 s |

> "Tek yönlü çöküş. 4000 karakterde model, cevabı bilmediğinde reddetmeyi
> tamamen bırakıyor — üç denemede sıfır. Sebebi şu: 'sadece verilen pasajlardan
> cevapla, bilmiyorsan bilmiyorum de' talimatı sistem mesajının başında duruyor,
> ve arkasından gelen her karakterle seyreliyor. Küçük bir modelde bağlam bir
> kaynak değil, **maliyet**.
>
> Kararım: bağlam bütçesi 1600 karakter. Ama `top_k` 5'te kaldı, çünkü
> ölçtüğüm şey recall@5. Yani sistem geniş getiriyor, dar besliyor."

### 4.2 Model seçimi

| model | gecikme | tekrar skoru | reddediyor mu | karar |
|---|---|---|---|---|
| qwen2.5-0.5b | 660 ms | 0,000 | ✗ uyduruyor | güvensiz |
| qwen2.5-1.5b | 10,8 s | **0,315** | ✗ | dejenere |
| **phi-3.5-mini** | 7,2 s | 0,000 | ✓ | **varsayılan** |
| qwen2.5-7b | 23,8 s | 0,000 | ✓ | en kaliteli, çok yavaş |

Burada söylenmesi gereken kritik cümle:

> "0.5b modeli tekrar skorunda mükemmel: 0,000. Ama içeriği uyduruyor. Bu şunu
> gösteriyor — tekrar metriği dejenerasyonu yakalıyor, halüsinasyonu
> yakalamıyor. Metrikler ölçmediklerini ölçüyormuş gibi görünürler. Bu yüzden
> değerlendirme katmanında ayrı bir dayanaklılık yargıcı var."

### 4.3 Kısa metin sorunu

> "Gömme vektörleri kısa metinlerde dejenere oluyor: birbiriyle alakasız iki
> kısa cümle 0,62 kosinüs benzerliği verebiliyor. Bu, chunk'lara bir alt sınır
> koymamın nedeni — çok kısa parçalar indekse gürültü olarak giriyor."

### 4.4 PDF onarımı

> "PDF'ten metin çıkarırken kelimeler satır sonunda kırılıyor. Kör bir
> birleştirme yeni hatalar üretir, o yüzden korpustan sözlük çıkarıp sadece
> birleşimi sözlükte olan kırıkları onardım: 21 gerçek kopukluk."

---

## 5. Canlı demo (4 dakika) — ekranda: dashboard, "Soru sor" sekmesi

Demonun amacı modelin akıllı olduğunu göstermek **değil**. Amacı sistemin
kararlarının görünür olduğunu göstermek. Bunu açıkça söyleyerek başla:

> "Şimdi canlı göstereceğim. Dikkat etmenizi istediğim şey cevabın kendisi
> değil — cevabın **neden o pasajlardan** geldiği."

### Soru 1 — normal çalışma

Yaz: `burs başvurusu için gereken not ortalaması nedir`

Cevap akarken konuş (boş bekleme, ~7 saniye):

> "Cevap akarken yanda aşamalar görünüyor: 30 yoğun aday, 30 BM25 adayı,
> füzyondan sonra kaç tane kaldığı, her aşamanın kaç milisaniye sürdüğü."

Cevap gelince **aşağıdaki tabloyu göster** — sunumun en önemli ekranı:

> "Bu tablo projenin özeti. Her satır bir pasaj, ve her pasaj için görüyorsunuz:
> yoğun aramada kaçıncı sıradaydı, BM25'te kaçıncı sıradaydı, füzyondan sonra
> hangi skoru aldı. Şuradaki satıra bakın — [BM25 sırası dolu, yoğun sırası boş
> olan bir satır seç] — bu pasajı anlamsal arama hiç getirmemiş, BM25 bulmuş.
> Hibrit olmasaydı bu cevap eksik olurdu."

> **Eğer öyle bir satır yoksa:** panikleme, tersini söyle — "bu soruda iki kol
> da aynı pasajları bulmuş, ki bu da bir bilgi: hibridin faydası her soruda
> değil, *bazı* sorularda ortaya çıkıyor. Hangi sorularda olduğunu ölçmek
> lazım, ki ölçtüm."

### Soru 2 — çekimserlik

Yaz: `yemekhanede bugün ne var`

> "Bu sorunun cevabı korpusta yok. Sistemin uydurmaması gerekiyor."

Model reddettiğinde:

> "Bu, bir RAG sisteminin en önemli davranışı. Uydurmamak, doğru cevaplamaktan
> daha zor — çünkü model her zaman *bir şey* üretebilir."

### Soru 3 — kelime dağarcığı farkı (isteğe bağlı, vaktin varsa)

Yaz: `kayıt dondurma nasıl yapılır`

> "Hatırlayın, bu ifade korpusta hiç geçmiyor. Şu an ne getirdiğine bakalım —
> [ne gelirse onu yorumla]. Bu, çözdüğüm bir problem değil, **ölçtüğüm** bir
> problem. Sorgu genişletme sıradaki iş."

### "Karşılaştır" sekmesi (30 saniye)

Aynı soruyu dört stratejide çalıştır, tabloyu göster:

> "Aynı soru, dört farklı erişim stratejisi. Ama dikkat: **bu bir kanıt değil.**
> Tek soruda gözlem, sezgi verir. Kanıt için istatistik gerekiyor, ki sıradaki
> bölüm o."

Bu cümle çok önemli — demoyu kendin eleştirerek bir sonraki bölüme geçiyorsun.

---

## 6. Değerlendirme ve dürüstlük (4 dakika) — sunumun en güçlü kısmı

Burada anlatacağın şey, projeyi diğerlerinden ayıran şey. Yavaş anlat.

> "Bir konfigürasyonun nDCG'si 0,71, diğerininki 0,74 çıkıyor. Çoğu blog yazısı
> burada 'ikincisi daha iyi' der. Bu yanlış. 15 soruluk bir sette 0,03 fark
> tamamen gürültünün içinde.
>
> Bu yüzden değerlendirme katmanı üç şey yapıyor: bootstrap güven aralıkları —
> bu sayı ne kadar kesin; eşleştirilmiş permütasyon testi — fark gerçek mi;
> Holm-Bonferroni düzeltmesi — çünkü on konfigürasyon karşılaştırıp en iyi
> p-değerini raporlamak, %40 ihtimalle yanlış bir keşif ilan etmek demek."

Sonra asıl anı:

> "Ve şimdi projenin en önemli kısmını anlatacağım: **ilk ölçüm koşumu çöpe
> attım.**"

[GitHub'da `data/results/archive-n15-invalid/NOTE.md` dosyasını aç]

> "Sonuçları silmedim, arşivledim, yanına neden geçersiz olduklarını yazdım.
> İki bağımsız kusur vardı.
>
> **Birincisi güç.** 30 soru istemiştim, 15 tane elde etmiştim, ve bunu bana
> hiçbir şey söylememişti — kod otuz chunk örnekliyor, elenenlerden sonra
> kalanı döndürüyordu. Onbeş soruyla orta büyüklükte bir etkiyi görmek
> matematiksel olarak mümkün değil; 32 soru gerekiyor. Nitekim on
> karşılaştırmanın hiçbiri düzeltmeden geçemedi. Ama bu 'konfigürasyonlar
> eşdeğer' demek değil, 'bu set ayırt edemedi' demek — ikisini karıştırmak
> güçsüz bir çalışmayı boş bir sonuç gibi raporlamaktır.
>
> **İkincisi daha sinsi.** Soruları korpustan üretmiştim, ki bu doğru bir
> tercih — böylece hangi pasajın doğru cevap olduğunu tam olarak biliyorum,
> elle etiketleme yok. Ama chunk'tan üretilen soru, chunk'ın kelimelerini
> taşıyor. BM25 tam olarak bu örtüşmeyi puanlıyor. Sonuç: sözlüksel arama 0,86,
> anlamsal arama 0,58. Bu tablo 'Türkçe'de BM25 daha iyi' demiyordu; 'benim
> sorularım belgelerin kopyasıydı' diyordu."

Sonra çözümü anlat:

> "İkisini de düzelttim. Artık `--n` istenen *kullanılabilir* soru sayısı;
> havuzdan hedefe ulaşana kadar çekiliyor. Ve her soru için gold pasajla kelime
> örtüşmesini kaydediyorum — **BM25 kolunun kullandığı aynı gövdelemeyle**,
> yani bir vekil değil sinyalin kendisi.
>
> Asıl çözüm ise şu: her sorunun bir de günlük dile çevrilmiş parafrazı var, ve
> benchmark iki seti **ayrı ayrı** koşuyor. İki set aynı pasajları hedeflediği
> için eşleştirilmiş test yapılabiliyor. Aradaki düşüş, bir konfigürasyonun
> puanının ne kadarının pasajı *bulmaktan*, ne kadarının kelime *eşleştirmekten*
> geldiğini söylüyor."

[Burada yeni benchmark tablosunu göster — §8'deki sayılar]

---

## 7. Sınırlar ve sıradaki iş (1,5 dakika)

Bunu atlamak cazip gelir; atlama. Sınırlarını kendin söylemek, soru-cevapta
birinin bulmasından çok daha iyi.

> "Dürüst olmam gereken yerler:
>
> **Değerlendirme setini modelin kendisi üretti.** Bu, model sınırlarıyla eval
> sınırlarını ilişkilendiriyor. Parafraz katmanı bunu hafifletiyor ama
> çözmüyor; gerçek çözüm insan yazımı bir set.
>
> **Korpus tek kurumdan, tek dilde.** Bulgularımın hepsi Türkçe yönetmelik
> metnine özgü olabilir. İngilizce bir kontrol seti bunu test ederdi.
>
> **Gövdeleme gerçek bir morfolojik analizci değil**, sezgisel bir ek soyucu.
> Çekim eklerinde çalışıyor, türetim eklerinde çalışmıyor — 'dondurulması' ile
> 'dondurma' hâlâ eşleşmiyor.
>
> **Çekimserlik eşiği henüz kalibre edilmedi.** Kodu hazır
> (`stats.find_best_threshold`), veri hazır, ama koşmadım.
>
> Sıradaki iş: eşik kalibrasyonu, sorgu genişletme, ve bir MCP sunucusu —
> böylece sistem Microsoft ekosistemindeki diğer araçlardan çağrılabilir."

---

## 8. Yeni benchmark sonuçları (2 dakika) — ekranda: README'deki sonuç tablosu

Bu bölümü §6'nın hemen ardına ekle; dürüstlük anlatısının ödülü bu.

> "Geçerli ölçüm şunu söyledi. Üç bulgu var.
>
> **Bir: en büyük etki Türkçe gövdeleme.** BM25 koluna Türkçe ek soyma eklemek
> nDCG'yi 0,21 artırıyor, p = 0,0014. Ve bu, parafrazlanmış sette de aynı
> kalıyor: 0,22, p = 0,0004. İki bağımsız yazımda tekrarlanan tek etki bu.
>
> **İki: füzyon işe yarıyor ama beklediğim biçimde değil.** RRF literatürün
> önerdiği yöntem ve benim varsayılanımdı. Düzeltmeden sonra ayakta kalmıyor.
> Ayakta kalan, ağırlıklı füzyon: temel sisteme göre +0,157, p = 0,0006 — ve
> parafrazlı sette +0,114, p = 0,0002. Ölçüm sonucunda projenin varsayılanını
> değiştirdim.
>
> **Üç: ilk koşumdaki en çarpıcı bulgu yanlıştı.** Hatırlayın, sözlüksel arama
> 0,86 ile anlamsal aramanın 0,58'ini eziyordu. Şimdi: üretilmiş sette avantaj
> 0,124 ve p = 0,065, yani anlamsız. Parafrazlanmış sette 0,042 ve p = 0,47,
> yani tamamen yok.
>
> Ve bunu görebilmemin tek nedeni, iki seti ayrı ölçmüş olmam."

Sonra kelime duyarlılığı tablosunu göster:

| konfigürasyon | düşüş | p | |
|---|---|---|---|
| lexical-nostem | −0,155 | 0,0014 | anlamlı |
| lexical-only | −0,146 | 0,0020 | anlamlı |
| rrf | −0,069 | 0,0735 | anlamsız |
| baseline-dense | −0,064 | 0,1121 | anlamsız |

> "Soruyu yeniden yazdığımda **sadece BM25 kolları çöküyor.** Hibrit ve anlamsal
> kollar dayanıyor. Bu, hibrit mimarinin varlık nedeninin ölçülmüş hâli."

### Çekimserlik bulgusu — bunu mutlaka anlat

> "Son olarak, en sevdiğim bulgu. Sistemin cevabı bilmediğinde susması için bir
> eşik lazım. Doğal aday, pipeline'ın ürettiği füzyon skoru. Ölçtüm ve
> **çalışmıyor**: AUC 0,73, ve cevaplanamaz bir soru mümkün olan **en yüksek
> skoru**, 1,000'i almış.
>
> Sebebini kodda buldum. Füzyon, iki kolun skorlarını normalize ediyor — ama
> korpus genelinde değil, o sorgu için getirilen adaylar içinde. Yani en iyi
> aday her zaman 1,0 alıyor. İyi olduğu için değil, **en iyisi olduğu için.**
> Bu skor 'bulduğumun en iyisi bu' diyor, 'bulduğum şey işe yarar' demiyor.
>
> Ham BM25 skoru normalize edilmiyor ve AUC 0,84 veriyor. Sinyali değiştirdim,
> eşiği ondan kalibre ettim: 6,866, duyarlılığı %90'da tutan en katı nokta."

Neden bu anı sona sakla: bir sayının kötü çıkması, kodu okuyup nedenini bulman,
ve düzeltmen — bir sunumda gösterilebilecek en iyi üç adımlık hikâye bu.

---

## 9. Kapanış (30 saniye)

> "Özetle: bana verilen plan sekiz cümlelik bir örnek için doğruydu. Gerçek bir
> Türkçe korpusta ise her adımı ölçmek gerekti — ve ölçtüğüm şeylerin bir kısmı
> sezgimin tersini söyledi. Bağlam eklemek cevabı bozdu. En hızlı model en
> güvensiz olandı. İlk benchmark'ım geçersizdi.
>
> Projenin asıl çıktısı bu sistem değil, bu sistemi ölçebilen araç takımı.
> Çünkü bir sonraki soru — 'sorgu genişletme işe yarıyor mu' — artık bir tahmin
> değil, bir koşu."

---

## 10. Beklenen sorular ve cevapları

**"Neden bulut kullanmadın, GPT-4 daha iyi olmaz mıydı?"**
> Olurdu, ama soru bu değildi. Yerel çalıştırmanın üç somut nedeni var:
> öğrenci verisi kurumdan çıkmıyor, çalışma zamanı maliyeti sıfır, ve internet
> olmadan çalışıyor. Bunun bedeli ne kadar? Ölçtüm: küçük modelde bağlam
> bütçesi 1600 karakterde tutulmak zorunda, ve 7 saniyelik gecikme var.

**"RRF yerine neden ağırlıklı füzyon kullanmadın?"**
> Denedim, benchmark'ta ikisi de var (`weighted-0.5`, `weighted-0.7`). RRF'i
> tercih etmemin nedeni, ağırlıklı füzyonun iki farklı ölçekteki skoru
> normalize etmeyi gerektirmesi — ve o normalizasyonun korpusa bağlı olması.
> RRF sadece sıra kullanıyor, ölçekten bağımsız.

**"Chunk boyutunu neden 900 karakter seçtin?"**
> Karakter, token değil — çünkü yerel modelin tokenizer'ıyla birebir eşleşen
> bir tokenizer'a erişimim yok. 900 rakamı yönetmelik maddelerinin tipik
> uzunluğundan geliyor, bir maddeyi ikiye bölmemek için. Bu, sweep'e sokulması
> gereken ama henüz sokulmadığım bir parametre.

**"Bu sistemi başka bir korpusa nasıl taşırım?"**
> `data/corpus/` içine belgeleri koyup indeksi yeniden kurmak yeterli. Ama
> ölçtüğüm sabitlerin — bağlam bütçesi, chunk boyutu, eşik — yeni korpusta
> yeniden ölçülmesi gerekir. Zaten laboratuvarın varlık nedeni bu.

**"Neden phi-3.5-mini? qwen2.5-7b daha iyiydi."**
> Daha iyi ama 23,8 saniye. Bu bir ürün kararı, teknik değil: interaktif bir
> demoda 24 saniye kullanılamaz. Değerlendirme seti *üretmek* gibi çevrimdışı
> işlerde 7b kullanıyorum, çünkü orada gecikme önemsiz.

**"Sonuçların istatistiksel olarak anlamlı mı?"**
> Bir kısmı. On karşılaştırmadan ikisi Holm düzeltmesinden sonra ayakta kaldı:
> weighted-0.5 ve weighted-0.7, üstelik iki kelime varyantında da. Ayakta
> kalmayanlar için "fark yok" demiyorum, "bu set ayıramadı" diyorum — n=60 ile
> orta büyüklükte etkileri görebiliyorum, küçük etkiler için 197 soru gerekir.

**"Neden varsayılanı RRF'ten weighted'a değiştirdin? Literatür RRF diyor."**
> Tam da bu yüzden ölçtüm. RRF benim de varsayılanımdı ve düzeltmeden sonra
> ayakta kalmadı; weighted her iki kelime varyantında da kaldı. Aradaki fark
> büyük değil — ama düzeltmeden sonra hayatta kalan ve iki bağımsız yazımda
> tekrarlanan tek erişim sonucu bu. Literatürün RRF'i tercih etme sebebi de
> geçerli ve ben de onu koruyorum: weighted, aday derinliğine bağlı normalize
> ettiği için `dense_top_k` değişince alpha'nın yeniden ayarlanması gerekiyor.
> RRF'te böyle bir bağ yok. Yani bu, bu korpusta ölçülmüş bir seçim, evrensel
> bir sonuç değil.

**"Sistem cevabı bilmediğini nereden biliyor?"**
> Erişim skoru bir eşiğin altındaysa modeli hiç çağırmıyor. Ama hangi skor
> olduğu önemli — ilk denediğim füzyon skoru işe yaramadı, çünkü sorgu içinde
> normalize ediliyor ve her zaman en iyisini 1,0 yapıyor. Ham BM25 skoruna
> geçince AUC 0,73'ten 0,84'e çıktı. Hâlâ kusursuz değil: cevaplanamaz
> soruların %38'ini kaçırıyor, o yüzden ikinci bir savunma hattı var — sistem
> mesajındaki "bilmiyorsan bilmiyorum de" kuralı.

**"Halüsinasyonu nasıl engelliyorsun?"**
> Tam olarak engelleyemiyorum, azaltıyorum ve ölçüyorum. Üç mekanizma: model
> sadece getirilen pasajlardan cevaplıyor, her cümle kaynak numarası taşıyor ve
> geçersiz kaynak numaraları sayılıyor, ve skor düşükse sistem reddediyor.
> Reddetme eşiği henüz kalibre edilmedi — bu bilinen bir eksik.

**"Kaç saat sürdü?"**
> [Kendi rakamını söyle.] Ama asıl cevap şu: sürenin yarısı sistemi yazmaktı,
> yarısı yazdığım şeyin çalışıp çalışmadığını ölçmekti. İkinci yarı olmasaydı
> bağlam bütçesi hâlâ 4000 karakterde olurdu ve sistem hiçbir soruyu
> reddetmiyor olurdu.

---

## 11. Anlatırken kaçınman gerekenler

- **Kod satırı gösterme.** Kimse sunum sırasında Python okumaz. Kodun kendisi
  değil, kodun verdiği kararlar ilginç. İstisna: birisi özellikle sorarsa.
- **"Basit bir şey yaptım" deme.** Ölçüm yapan bir sistem kurdun; küçültme.
- **Bilmediğin bir şeyi doldurma.** "Ölçmedim" tam bir cevap, ve bu projede
  özellikle güçlü bir cevap — çünkü sunumun geri kalanı ölçtüklerinle dolu.
- **Metrik ismi yağdırma.** nDCG'yi bir kez tanımla ("doğru pasaj listede kaçıncı
  sırada, ne kadar yukarıdaysa o kadar iyi") ve sonra sadece kullan.
- **Demoyu uzatma.** Üç soru yeter. Dördüncü soru bilgi eklemez, risk ekler.
