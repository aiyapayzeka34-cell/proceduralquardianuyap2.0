"""
BERTurk İnce Ayar (Fine-Tuning) İskeleti - Prosedürel Uygunluk Sınıflandırması
================================================================================
UYAP 2.0 Prosedürel Uyum Muhafızı - Bitirme Tezi Ek Materyali

Temel model: dbmdz/bert-base-turkish-cased (BERTurk)
Görev: Elle etiketlenmiş Yargıtay karar metinlerinden ikili/çoklu sınıf
       prosedürel uygunluk tahmini (yargitay_scraper.py çıktısı olan
       etiketleme_sablonu.csv dosyası doldurulduktan sonra buraya girdi olur).

Kurulum:
    pip install transformers datasets torch scikit-learn pandas accelerate --break-system-packages

NOT: Bu bir İSKELETtir - küçük ölçekli pilot (200-500 etiketli örnek) için
     tasarlanmıştır. Üretim ölçeğine geçmeden önce veri miktarı, sınıf
     dengesi ve değerlendirme metodolojisi genişletilmelidir.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("berturk_finetune")

MODEL_ADI = "dbmdz/bert-base-turkish-cased"
MAKS_TOKEN_UZUNLUGU = 512          # BERT sınırı; uzun kararlar için ileride
                                     # chunk/longformer stratejisi değerlendirilmeli
ETIKET_HARITASI = {"0": 0, "1": 1}  # 0 = uygun değil, 1 = uygun (ikili örnek)
CIKTI_DIZINI = "berturk_prosedur_model"


class KararDataset(Dataset):
    """Yargıtay karar metinlerini BERT girdisine dönüştüren basit Dataset sınıfı."""

    def __init__(self, metinler, etiketler, tokenizer, maks_uzunluk=MAKS_TOKEN_UZUNLUGU):
        self.metinler = list(metinler)
        self.etiketler = list(etiketler)
        self.tokenizer = tokenizer
        self.maks_uzunluk = maks_uzunluk

    def __len__(self):
        return len(self.metinler)

    def __getitem__(self, idx):
        kodlama = self.tokenizer(
            self.metinler[idx],
            truncation=True,
            padding="max_length",
            max_length=self.maks_uzunluk,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in kodlama.items()}
        item["labels"] = torch.tensor(self.etiketler[idx], dtype=torch.long)
        return item


def veri_yukle_ve_bol(
    csv_yolu: str,
    metin_sutunu: str = "metin_temiz",
    etiket_sutunu: str = "prosedur_uygun",
    test_orani: float = 0.2,
    val_orani: float = 0.1,
    random_state: int = 42,
):
    """
    yargitay_scraper.py'nin etiketleme_sablonu.csv çıktısını okur.
    UYARI: prosedur_uygun sütunu elle doldurulmuş olmalı; boş satırlar atılır.
    """
    df = pd.read_csv(csv_yolu, encoding="utf-8-sig")
    df = df.dropna(subset=[metin_sutunu, etiket_sutunu])
    df = df[df[etiket_sutunu].astype(str).isin(ETIKET_HARITASI.keys())]

    if len(df) < 30:
        logger.warning(
            f"Sadece {len(df)} etiketli örnek bulundu. Küçük ölçekli pilot için "
            "en az 150-200 örnek önerilir; sonuçlar gösterge niteliğinde kalacaktır."
        )

    df["etiket_int"] = df[etiket_sutunu].astype(str).map(ETIKET_HARITASI)

    egitim_val, test = train_test_split(
        df, test_size=test_orani, random_state=random_state,
        stratify=df["etiket_int"] if df["etiket_int"].nunique() > 1 else None,
    )
    egitim, val = train_test_split(
        egitim_val, test_size=val_orani / (1 - test_orani), random_state=random_state,
        stratify=egitim_val["etiket_int"] if egitim_val["etiket_int"].nunique() > 1 else None,
    )

    logger.info(f"Eğitim: {len(egitim)} | Doğrulama: {len(val)} | Test: {len(test)}")
    return (
        egitim[metin_sutunu].tolist(), egitim["etiket_int"].tolist(),
        val[metin_sutunu].tolist(), val["etiket_int"].tolist(),
        test[metin_sutunu].tolist(), test["etiket_int"].tolist(),
    )


def metrikleri_hesapla(eval_pred):
    """Trainer için değerlendirme fonksiyonu - gerçek Precision/Recall/F1 üretir."""
    tahminler, etiketler = eval_pred
    tahmin_siniflari = np.argmax(tahminler, axis=1)

    dogruluk = accuracy_score(etiketler, tahmin_siniflari)
    kesinlik, duyarlilik, f1, _ = precision_recall_fscore_support(
        etiketler, tahmin_siniflari, average="binary", zero_division=0
    )

    return {
        "accuracy": dogruluk,
        "precision": kesinlik,
        "recall": duyarlilik,
        "f1": f1,
    }


def modeli_egit(
    csv_yolu: str = "etiketleme_sablonu.csv",
    epoch_sayisi: int = 4,
    batch_boyutu: int = 8,
    ogrenme_orani: float = 2e-5,
):
    """Ana ince ayar döngüsü."""
    logger.info(f"Tokenizer ve model yükleniyor: {MODEL_ADI}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ADI, num_labels=len(ETIKET_HARITASI)
    )

    (egitim_x, egitim_y, val_x, val_y, test_x, test_y) = veri_yukle_ve_bol(csv_yolu)

    egitim_ds = KararDataset(egitim_x, egitim_y, tokenizer)
    val_ds = KararDataset(val_x, val_y, tokenizer)
    test_ds = KararDataset(test_x, test_y, tokenizer)

    egitim_ayarlari = TrainingArguments(
        output_dir=CIKTI_DIZINI,
        num_train_epochs=epoch_sayisi,
        per_device_train_batch_size=batch_boyutu,
        per_device_eval_batch_size=batch_boyutu,
        learning_rate=ogrenme_orani,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{CIKTI_DIZINI}/logs",
        logging_steps=10,
        report_to="none",  # wandb/tensorboard entegrasyonu isteğe bağlı eklenebilir
    )

    trainer = Trainer(
        model=model,
        args=egitim_ayarlari,
        train_dataset=egitim_ds,
        eval_dataset=val_ds,
        compute_metrics=metrikleri_hesapla,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("İnce ayar başlıyor...")
    trainer.train()

    logger.info("Test seti üzerinde değerlendirme yapılıyor (GERÇEK metrikler)...")
    test_sonuclari = trainer.evaluate(test_ds)
    logger.info(f"Test sonuçları: {test_sonuclari}")

    # Karışıklık matrisi - tez Bölüm 9 ekine gerçek pilot verisi olarak eklenebilir
    tahminler = trainer.predict(test_ds)
    tahmin_siniflari = np.argmax(tahminler.predictions, axis=1)
    cm = confusion_matrix(test_y, tahmin_siniflari)
    logger.info(f"Karışıklık Matrisi:\n{cm}")

    trainer.save_model(f"{CIKTI_DIZINI}/final")
    tokenizer.save_pretrained(f"{CIKTI_DIZINI}/final")
    logger.info(f"Model kaydedildi -> {CIKTI_DIZINI}/final")

    return trainer, test_sonuclari, cm


def tahmin_yap(metin: str, model_dizini: str = f"{CIKTI_DIZINI}/final") -> dict:
    """Eğitilmiş modelle tek bir metin üzerinde çıkarım (inference) yapar."""
    tokenizer = AutoTokenizer.from_pretrained(model_dizini)
    model = AutoModelForSequenceClassification.from_pretrained(model_dizini)
    model.eval()

    girdi = tokenizer(metin, truncation=True, padding=True,
                       max_length=MAKS_TOKEN_UZUNLUGU, return_tensors="pt")
    with torch.no_grad():
        cikti = model(**girdi)
        olasiliklar = torch.softmax(cikti.logits, dim=1).squeeze().tolist()

    tahmin_sinifi = int(np.argmax(olasiliklar))
    return {
        "tahmin": tahmin_sinifi,
        "guven": olasiliklar[tahmin_sinifi],
        "tum_olasiliklar": olasiliklar,
    }


if __name__ == "__main__":
    # Ön koşul: yargitay_scraper.py -> etiketleme_sablonu.csv üretilmiş ve
    # 'prosedur_uygun' sütunu elle doldurulmuş olmalı.
    csv_yolu = "etiketleme_sablonu.csv"

    if not Path(csv_yolu).exists():
        logger.error(
            f"{csv_yolu} bulunamadı. Önce yargitay_scraper.py çalıştırılmalı ve "
            "etiketleme sütunları elle doldurulmalı."
        )
    else:
        trainer, sonuclar, cm = modeli_egit(csv_yolu=csv_yolu)

        # Örnek çıkarım
        ornek_metin = (
            "Davalıya usulüne uygun tebligat yapılmadan yargılamaya devam edilmesi "
            "hukuki dinlenilme hakkının ihlali niteliğindedir."
        )
        sonuc = tahmin_yap(ornek_metin)
        logger.info(f"Örnek tahmin: {sonuc}")
