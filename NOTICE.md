# Üçüncü taraf içerik hakkında

Bu deponun **kodu** MIT lisanslı ([LICENSE](LICENSE)). Aşağıdaki içerik o
lisansın kapsamı dışında.

## Korpus

`data/corpus/` altındaki dokuz belge, Boğaziçi Üniversitesi'nin kamuya açık
yönetmelik ve yönergeleri (bogazici.edu.tr): kayıt, burs, özel öğrenci,
disiplin, konut tahsisi, sertifika programları, pedagojik formasyon, lisans ve
lisansüstü eğitim-öğretim.

Bu belgeler yayıncısının mülkiyetindedir ve burada **değerlendirme materyali**
olarak, kamuya açık oldukları anlayışıyla yer alıyor. Depoyu başka bir bağlamda
kullanacaksan bu varsayımı kendi yargı alanın için gözden geçir.

Korpusu değiştirmek zor değil ve sistem buna göre tasarlandı: `data/corpus/`
içeriğini değiştirip indeksi yeniden kur.

```bash
PYTHONPATH=src ./.venv/bin/python scripts/build_index.py --force
PYTHONPATH=src ./.venv/bin/python scripts/build_eval_set.py --n 60 --paraphrase --model qwen2.5-7b
```

İkinci komut şart: chunk id'leri yeniden kurulumda değiştiği için eski eval
setinin gold etiketleri geçersizleşir.

## Türetilmiş veriler

`data/eval/eval_set.json` içindeki sorular, korpustan yerel bir dil modeliyle
üretildi; `gold_answer` alanları kaynak pasajların kısaltılmış hâlleridir,
dolayısıyla yukarıdaki kapsam dışılık onlar için de geçerli.

`data/index/bogazici.db` korpusun gömülmüş ve indekslenmiş hâlini içeriyor.

## Modeller

Foundry Local üzerinden indirilen modeller (phi-3.5-mini, qwen2.5-*,
qwen3-embedding-0.6b) bu depoda **yer almıyor**; çalışma zamanında indiriliyor
ve `~/.foundry_rag_lab` altında saklanıyor. Her biri kendi lisansına tabidir.
