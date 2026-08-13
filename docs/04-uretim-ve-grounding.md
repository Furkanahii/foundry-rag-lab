# 04 — Üretim ve grounding

> Bağlı kod: `generate/answerer.py`, `scripts/prompt_ablation.py`

Erişim doğru pasajı bulduktan sonra iş bitmiyor. Model o pasajları kullanmak
zorunda değil — kendi ön eğitim bilgisinden cevap üretebilir, pasajları
karıştırabilir, ya da hiç var olmayan bir kaynağa atıf yapabilir. Bu doküman,
üretim aşamasında dayanaklılığı (grounding) sağlayan üç mekanizmayı ve her
birinin ölçülmüş sınırını anlatıyor.

---

## 1. Bağlam bütçesi: sezginin tersi

RAG anlatılarının ortak varsayımı "daha çok bağlam, daha iyi cevap"tır.
`scripts/prompt_ablation.py` bunu phi-3.5-mini üzerinde ölçtü:

| bağlam | doğru reddetme | doğru cevap | gecikme |
|---|---|---|---|
| **1200 kr** | **2/3** | **2/2** | 38,6 s |
| 2400 kr | 1/3 | 1/2 | 59,6 s |
| 4000 kr | 0/3 | 0/2 | 57,2 s |
| 6000 kr | 0/3 | 1/2 | 93,9 s |

Tek yönlü çöküş. 4000 karakterde model hem doğru cevap veremiyor hem de
bilmediğinde reddetmeyi tamamen bırakıyor.

**Neden:** grounding kuralları sistem mesajının *başında* duruyor
(`_SYSTEM_TR`), bağlam ise arkasından geliyor. Küçük bir modelde dikkat, uzun
bir bağlam boyunca talimatı taşıyamıyor — kural, sonrasında gelen her token'la
seyreliyor. Büyük modellerde bu etki çok daha zayıf; bu bulgu **modele özgü**,
RAG'e genel bir yasa değil.

**Karar:** bağlam bütçesi 1600 karakter (`Answerer.max_context_chars`).

Bu, erişimle üretim arasında kasıtlı bir gerilim yaratıyor. `top_k` 5'te
kalıyor çünkü ölçtüğümüz metrik recall@5 — sistemin doğru pasajı *bulup
bulmadığını* bilmek istiyoruz. Ama bütçe pratikte yalnızca ilk ~2 chunk'ın
modele ulaşmasını sağlıyor. Slogan hâline getirilebilir: **geniş getir, dar
besle.**

Bunun bedeli var ve saklanmamalı: doğru pasaj 4. sırada geldiğinde erişim
metriği başarı sayıyor, kullanıcı ise yanlış cevap alıyor. Erişim metriği ile
uçtan uca doğruluk arasındaki bu boşluk, `evaluate_end_to_end`'in neden ayrı
bir fonksiyon olduğunun cevabı.

---

## 2. Atıf doğrulama: modelin kendi kendini ele vermesi

Modele "kaynak göster" demek kolay. Gösterdiği kaynağın var olduğunu
doğrulamak, uydurmayı yakalayan kısım.

`answerer.py` üretilen metindeki `[n]` işaretlerini ayrıştırıyor ve her birini
gerçekten verilen pasaj sayısıyla karşılaştırıyor. Beş pasaj verilmişken `[7]`
diyen bir cevap, sistemin **kendi başına**, yargıç modeli veya doğru cevap
etiketi olmadan tespit edebildiği bir halüsinasyon.

```python
is_grounded = bool(citations) and not invalid_citations
```

Hiç atıf içermeyen bir cevap yanlış olmak zorunda değil, ama **doğrulanamaz**.
Bu sistem için doğrulanamaz olmak bir başarısızlık türü: projenin tüm amacı
her iddianın bir yönetmelik maddesine kadar izlenebilmesi.

Sınırı da açık: geçerli bir atıf, o pasajın iddiayı gerçekten *desteklediğini*
göstermez. Model doğru numarayı verip yanlış şey söyleyebilir. Bunu yakalamak
için ayrı bir dayanaklılık yargıcı gerekiyor — kurulmadı, `evaluate_answer`
şu an yalnızca atıfların varlığını ve geçerliliğini ölçüyor.

---

## 3. Çekimserlik: modelden önce

Reddetme kararı, model çağrılmadan **önce** veriliyor. Erişim skoru eşiğin
altındaysa `answer()` doğrudan sabit bir ret metni döndürüyor.

Bu bir güvenlik özelliği olduğu kadar bir gecikme özelliği: cevaplanamaz bir
soruya verilebilecek en ucuz cevap, dil modeline hiç uğramayan cevaptır. Bu
makinede 7 saniyelik yanlış bir cevap, 300 milisaniyelik dürüst bir cevaba
dönüşüyor.

Eşiğin ve sinyalin nereden geldiği ayrı bir konu:
[05 — Kalibrasyon ve çekimserlik](05-kalibrasyon-abstention.md).

**İki savunma hattı var ve ikisi de gerekli.** Eşik, erişim zayıf olduğunda
devreye giriyor (AUC 0.843 — cevaplanamaz soruların yaklaşık %38'ini
kaçırıyor). Sistem mesajındaki "bağlam yetmiyorsa şunu yaz" kuralı, eşiği
geçen ama yine de cevaplanamayan soruları yakalıyor — ta ki bağlam 1600
karakteri aşana kadar, ki §1 o hattın nerede çöktüğünü gösteriyor. Yani iki
hat birbirinin yedeği değil: birincisi erişim başarısızlığını, ikincisi bağlam
yetersizliğini yakalıyor.

---

## 4. Dil, korpusu değil soruyu takip ediyor

Türkçe sorulan bir soru Türkçe cevaplanıyor, kaynak belge İngilizce olsa bile.
`detect_language` çok kısa sorgularda "unknown" döndürüyor ve bu durumda
Türkçe'ye düşülüyor — korpusun varsayılan dili o, ve kullanıcıyı sessizce
başka bir dile geçirmek, yanlış tahmin edildiğinde en can sıkıcı hata.

---

## 5. Ölçülmemiş olanlar

Dürüstlük için: bu bölümdeki iddiaların hangileri ölçülmedi.

- **Bağlam bütçesi ablasyonu n=5 soruyla yapıldı** (3 cevaplanamaz, 2
  cevaplanabilir). Etki büyük ve tek yönlü olduğu için yön güvenilir, ama
  1600 karakter rakamının kendisi gürültülü. `docs/06`'daki güç analizi bu
  ablasyona uygulanmadı.
- **Atıfların doğruluğu** değil yalnızca geçerliliği ölçülüyor.
- **Sıcaklık 0.1 seçildi ama taranmadı.** Gerekçe "üretimde kararlılık
  istiyoruz" — makul, ama ölçüm değil.
