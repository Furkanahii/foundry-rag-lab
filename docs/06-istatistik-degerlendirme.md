# 6. Değerlendirmenin istatistiği

Bu doküman `evaluation/stats.py` ve `evaluation/metrics.py` arkasındaki
matematiği açıklar. Projenin diğer RAG demolarından ayrıldığı yer burasıdır.

**Önlemeye çalıştığımız hata şudur:** A konfigürasyonunu çalıştır, nDCG 0.71
al. B'yi çalıştır, 0.74 al. "B daha iyi" de. 40 soruluk bir sette 0.03'lük fark
gürültünün içindedir. Blog yazılarındaki RAG "iyileştirmelerinin" önemli bir
kısmı bu hatadır.

---

## 6.1 Hangi metrik neyi ölçer

| Metrik | Cevapladığı soru | Ne zaman |
|---|---|---|
| **Recall@k** | Doğru pasajı ilk k içinde getirdik mi? | Her şeyin tavanı |
| **Precision@k** | Getirdiğimiz k'nın kaçı alakalıydı? | Bağlam pahalıysa |
| **MRR** | İlk doğru sonuç kaçıncı sırada? | Kullanıcı tek cevap arıyorsa |
| **nDCG@k** | Kademeli, sıra-indirimli kazanç | Sıralama kalitesi önemliyse |

**Recall@k neden önce gelir:** Erişimin hiç getirmediği bir pasaj yeniden
sıralanamaz, atıf verilemez, okunamaz. Recall her şeyin üst sınırıdır. Önce onu
optimize et.

**MRR neden ilk sonuçtan sonrasını yok sayar:** Tasarım gereği. Kullanıcı tek
bir cevap istiyorsa ve yukarıdan okuyorsa, 2. doğru sonucun değeri yoktur.

### nDCG neden log₂(sıra+1) ile indirimli

$$ \text{DCG} = \sum_{i} \frac{g_i}{\log_2(i+1)} $$

Logaritma, kullanıcı davranışı hakkında bir iddiadır: **dikkat sırayla azalır,
ama yavaş** — kabaca $1/\log(\text{sıra})$ gibi, $1/\text{sıra}$ gibi değil.
Doğrusal indirim, 10. sıranın 1. sıranın onda biri değerinde olduğunu söylerdi;
bu, gerçek kullanıcıların ne kadar çabuk pes ettiğini abartır.

Önemli nokta: bu indirim **karşılaştırdığımız tüm sistemler için aynıdır**,
dolayısıyla birini kayıramaz.

**İdeal DCG'ye bölmek** sorular arası karşılaştırmayı mümkün kılar. Beş alakalı
pasajı olan bir soru, tek alakalı pasajı olandan daha çok ham kazanç
toplayabilir; ideale bölmek bunu ortadan kaldırır.

### Recall'un paydası

`evaluate_retrieval` içinde payda `min(n_relevant, k)`. Sebebi: 8 alakalı pasaj
varken $k=5$ ile en fazla 5 tanesi getirilebilir. Kusursuz bir sistemi 0.625
diye puanlamak, sistemi $k$'nın değeri yüzünden cezalandırmak olurdu.

---

## 6.2 Bootstrap: bu sayı ne kadar hassas

Elimizde **bir örneklem** var: eval setindeki sorular. Bu sorular, kullanıcıların
sorabileceği tüm soruların uzayından çekilmiş bir örnektir. Ölçtüğümüz ortalama
nDCG, gerçek ortalamanın bir **tahminidir** — ve her tahminin hatası vardır.

Bootstrap bu hatayı **dağılım varsayımı yapmadan** tahmin eder:

1. Soruları **yerine koyarak** yeniden örnekle (aynı boyutta)
2. Ortalamayı yeniden hesapla
3. Bunu 10.000 kez yap
4. Elde edilen ortalamaların %2.5 ve %97.5 yüzdeliklerini al

Neden çalışır: ampirik dağılım, gerçek dağılımın tutarlı bir tahminidir.
Ondan yeniden örneklemek, popülasyondan taze örnek çekmeyi taklit eder.

**Burada özellikle uygun olmasının sebebi:** soru başına metrikler ciddi biçimde
normal değil. MRR sadece $1, \tfrac12, \tfrac13, \dots, 0$ değerlerini alır.
Recall@5 altı noktada tanımlıdır. t-dağılımı temelli bir güven aralığı, bu
verinin sahip olmadığı bir şekli varsayar.

Kendi doğrulamamız:

```
n=20  -> 0.6529 [0.5560, 0.7409]   yarı genişlik ±0.092
n=200 -> 0.6968 [0.6657, 0.7272]   yarı genişlik ±0.031
```

Örneklem 10 kat büyüyünce aralık ~3 kat daralıyor — beklenen $\sqrt{n}$ ölçeği.

**Kritik okuma:** n=20 ile ölçtüğünüz 0.65'in gerçek değeri 0.56 ile 0.74
arasında herhangi bir yerde olabilir. Bu aralıkta iki konfigürasyonu ayırt
etmek imkânsızdır.

---

## 6.3 Eşleştirilmiş permütasyon testi: B gerçekten daha mı iyi

İki konfigürasyon **aynı sorular** üzerinde çalıştırılır. Bu bir **eşleştirilmiş**
(paired) tasarımdır ve öyle analiz edilmelidir.

**Eşleştirme neden önemli:** Bazı sorular her sistem için zordur. İki koşuyu
bağımsız örneklem gibi ele almak, hangi soruların zor olduğunu tam olarak
bildiğimiz bilgisini çöpe atar. Eşleştirme, soru zorluğunu karşılaştırmadan
çıkarır ve testin gücünü ciddi biçimde artırır.

**Testin mantığı.** Sıfır hipotezi: "iki konfigürasyon değiştirilebilir"
(exchangeable). Bu doğruysa, herhangi bir soruda A ve B'yi yer değiştirmek eşit
olasılıkla gözlemlediğimiz şeyi üretirdi. Yani her soru için farkın **işaretini
rastgele çevirmek** sıfır dağılımını üretir:

1. Soru başına farkları hesapla: $d_i = b_i - a_i$
2. Her $d_i$'nin işaretini rastgele çevir, ortalamayı al
3. 10.000 kez tekrarla → sıfır dağılımı
4. Gözlenen ortalama farkın bu dağılımda nerede durduğuna bak

**Hiçbir dağılım varsayımı yok.**

p-değeri formülünde pay ve paydaya +1 eklenir:

$$ p = \frac{\#\{|\text{null}| \geq |\text{gözlenen}|\} + 1}{B + 1} $$

Sebebi: gözlenen düzenleme de permütasyonlardan biridir. Bu, $p$'yi kesinlikle
pozitif tutar ve testi geçerli kılar.

Kendi doğrulamamız:

```
gerçek fark yok : p=0.0563  → reddetmiyor  ✓
gerçek iyileşme : p=0.0001, Cohen's d=1.16 → reddediyor  ✓
```

### Güven aralıkları çakışıyorsa fark yok mu?

**Hayır — bu yaygın bir tuzaktır.** İki aralık çakışmıyorsa fark anlamlıdır,
ama çakışıyorsa fark olmadığı **sonucu çıkmaz**. Eşleştirilmiş tasarımda test,
aralıkların karşılaştırılmasından çok daha güçlüdür. `ConfidenceInterval.overlaps`
metodunun docstring'inde bu açıkça uyarı olarak yazılıdır.

---

## 6.4 Çoklu karşılaştırma: sweep'in gizli tuzağı

12 konfigürasyon deneyip en iyi p-değerini raporlamak bir **spesifikasyon
aramasıdır**. $\alpha = 0.05$'te 12 bağımsız test yapıldığında en az bir yanlış
pozitif görme olasılığı:

$$ 1 - 0.95^{12} \approx 0.46 $$

Yani **%46 ihtimalle** hiçbir gerçek fark olmasa bile "anlamlı" bir sonuç
bulursunuz.

**Holm-Bonferroni** bunu düzeltir. p-değerlerini küçükten büyüğe sırala, $i$.
sırayı $\alpha/(m-i)$ ile karşılaştır, ilk başarısızlıkta dur. Düz
Bonferroni'den daha güçlüdür (aynı hata oranını kontrol ederken daha çok gerçek
etkiyi yakalar).

Kendi doğrulamamız — bu, sunumdaki en çarpıcı slayt olabilir:

```
p = [0.001, 0.02, 0.03, 0.04, 0.20]
düzeltmesiz    : [✓, ✓, ✓, ✓, ✗]   → 4 "anlamlı" sonuç
Holm sonrası   : [✓, ✗, ✗, ✗, ✗]   → 1 anlamlı sonuç
```

Dört keşiften üçü buharlaştı.

---

## 6.5 Güç analizi: eval setim yeterli mi

Anlamsız bir sonuç iki şeyden biri anlamına gelir: (a) fark yok, (b) fark var
ama görecek kadar veri yok. Bunları ayırmadan "fark yok" demek yanlıştır.

Eşleştirilmiş tasarım için normal yaklaşımı:

$$ n = \left(\frac{z_{1-\alpha/2} + z_{1-\beta}}{d}\right)^2 $$

Kendi hesabımız (%80 güç, $\alpha=0.05$):

| Etki büyüklüğü | Gereken n |
|---|---|
| küçük (d=0.2) | **197** |
| orta (d=0.5) | **32** |
| büyük (d=0.8) | **13** |

**Pratik sonuç:** 40 soruluk bir eval seti orta ve büyük etkileri görebilir, ama
küçük etkileri **göremez**. Dolayısıyla "rrf-k20 ile rrf-k60 arasında fark
bulamadık" cümlesi, "fark yok" değil, "bu setle görülemez" demektir.
`run_benchmark.py` bunu her koşuda otomatik raporlar.

---

## 6.6 Kalibrasyon: skora güvenebilir miyim

Çekimserlik eşiği için gerekli. Erişim skoru düşükse cevap vermeyi reddediyoruz
— ama o skorun bir anlamı olması gerekir.

**Expected Calibration Error:**

$$ \text{ECE} = \sum_{b} \frac{n_b}{N} \left| \text{doğruluk}_b - \text{güven}_b \right| $$

Tahminleri güven değerine göre kutulara ayır, her kutuda güven ile gerçek
doğruluk arasındaki farkı ölç, kutu büyüklüğüne göre ağırlıklandır.

**Ağırlıklandırma neden önemli:** İki örnek içeren berbat kalibre bir kutu, iki
yüz örnek içeren iyi kalibre bir kutuyu domine etmemeli.

**MCE** (en kötü kutu) ayrıca raporlanır, çünkü çekimserlik gibi bir **güvenlik**
kararında ortalama değil en kötü durum önemli olabilir.

**Brier skoru** da raporlanır çünkü **proper scoring rule**'dur: ECE'nin aksine,
sürekli taban oranını tahmin eden bir model tarafından kandırılamaz.

Kendi doğrulamamız — bunlar **sentetik** veri üzerinde, uygulamanın doğru
çalıştığını göstermek için; sistemin kalibrasyon sonucu değil:

```
iyi kalibre  : ECE=0.0385  MCE=0.0795  Brier=0.1687
kötü kalibre : ECE=0.2562  MCE=0.4525  Brier=0.3337
```

### Ve bu projede ECE neden hesaplanmadı

Yukarıdaki her şey doğru, ama seçilen çekimserlik sinyaline **uygulanamadı**.
Sinyal ham BM25 skoru ve $[0,1]$ aralığında değil (bu korpusta 2.41–47.78).
ECE bir skoru doğrulukla karşılaştırır; sınırsız bir sayıda tanımsızdır.
Kutulara zorlandığında tüm değerler son kutuya yığılıyor ve **ECE = 0.0000**
gibi gurur verici bir sonuç çıkıyor — yanında Brier = 152 dururken. İkisi aynı
anda doğru olamaz.

`calibrate_abstention.py` bu yüzden metriği raporlamak yerine **reddediyor** ve
nedenini yazıyor. Sessizce yanlış bir sayı üreten bir metrik, hiç
hesaplanmayandan tehlikelidir.

Eşiğin geçerliliği bundan etkilenmiyor: bir eşik yalnızca **monotonluk**
gerektirir, skorun olasılık olmasını değil. Ayrıntı:
[05 — Kalibrasyon ve çekimserlik](05-kalibrasyon-abstention.md).

### Eşiği veriden öğrenmek

`find_best_threshold` tüm olası kesme noktalarını tarar ve seçilen amacı
maksimize edeni döndürür. Varsayılan F1'dir, ama bu **açıkça belirtilmesi
gereken bir değer yargısıdır**:

- Cevaplanabilir soruyu reddetmek → kullanıcıyı rahatsız eder
- Cevaplanamaz soruyu cevaplamak → **yönetmelik uydurur**

İnsanların güvenebileceği bir sistemde ikinci hata daha ağırdır. Gerçek bir
dağıtımda, gereksiz reddetmeleri göze alıp uydurmayı azaltmak için `recall`
optimize edilebilir. Bu bir mühendislik kararı değil, bir **politika** kararıdır.

---

## Özet: bir iddiayı nasıl savunuruz

Bir konfigürasyonun daha iyi olduğunu söylemek için gereken zincir:

1. **Sabit eval seti** — korpustan üretilmiş, kaydedilmiş, değişmeyen
2. **Soru başına skorlar** — ortalama değil, ham değerler saklanır
3. **Bootstrap güven aralığı** — sayının hassasiyeti
4. **Eşleştirilmiş permütasyon testi** — fark gerçek mi
5. **Etki büyüklüğü** — fark önemli mi (istatistiksel anlamlılık ≠ pratik önem)
6. **Holm düzeltmesi** — sweep yaptıysak
7. **Güç kontrolü** — anlamsız sonuç, yokluk kanıtı değil

Bu zincirin herhangi bir halkası eksikse, iddia bir gözlemdir, kanıt değil.
