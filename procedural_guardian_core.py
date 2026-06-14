"""
Procedural Guardian - Core Engine
Prosedürel Uyum Muhafızı - Çekirdek Motor
Nişantaşı Üniversitesi | Yapay Zeka Yüksek Lisans
Danışman: Dr. Ali Özkurt
2026
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

# ============================================
# VERİ MODELLERİ
# ============================================

class UyumDurumu(Enum):
    UYGUN = "uygun"
    UYARİ = "uyari"
    KRİTİK = "kritik"
    HATA = "hata"

class DavaTuru(Enum):
    ALACAK = "alacak_davasi"
    KİRA = "kira_davasi"
    BOŞANMA = "bosanma_davasi"
    İCRA = "icra_takibi"
    TAZMİNAT = "tazminat_davasi"
    İŞ_DAVASI = "is_davasi"
    CEZA = "ceza_davasi"
    İDARE = "idare_davasi"

@dataclass
class ProsedurAdimi:
    adim_id: str
    adim_adi: str
    aciklama: str
    sira: int
    sure_gun: int
    gerekli_belgeler: List[str] = field(default_factory=list)
    bagimli_adimlar: List[str] = field(default_factory=list)
    tamamlandi: bool = False
    tamamlanma_tarihi: Optional[datetime] = None

@dataclass
class UyumRaporu:
    dava_id: str
    genel_skor: float
    durum: UyumDurumu
    uyari_mesajlari: List[str] = field(default_factory=list)
    kritik_hatalar: List[str] = field(default_factory=list)
    oneriler: List[str] = field(default_factory=list)
    adim_raporlari: List[Dict] = field(default_factory=list)

# ============================================
# PROSEDÜR HARİTASI VERİTABANI
# ============================================

class ProsedurHaritasiDB:
    def __init__(self):
        self.haritalar = self._init_default_haritalar()

    def _init_default_haritalar(self) -> Dict:
        return {
            DavaTuru.ALACAK.value: [
                ProsedurAdimi("A1", "Dava Dilekçesi Hazırlama", 
                    "Alacak davası dilekçesi hazırlanır", 1, 7,
                    ["dava_dilekcesi", "vekaletname", "alacak_belgesi"]),
                ProsedurAdimi("A2", "Mahkemeye Başvuru",
                    "Yetkili mahkemeye dilekçe verilir", 2, 3,
                    ["dava_dilekcesi"], ["A1"]),
                ProsedurAdimi("A3", "Harc Yatirma",
                    "Dava harcı yatırılır", 3, 3,
                    ["harc_makbuzu"], ["A2"]),
                ProsedurAdimi("A4", "Tebligat",
                    "Davalıya tebligat yapılır", 4, 15,
                    [], ["A3"]),
                ProsedurAdimi("A5", "Cevap Dilekçesi",
                    "Davalı cevap dilekçesi verir", 5, 30,
                    ["cevap_dilekcesi"], ["A4"]),
                ProsedurAdimi("A6", "İlk Duruşma",
                    "İlk duruşma yapılır", 6, 60,
                    ["duruşma_tutanağı"], ["A5"]),
                ProsedurAdimi("A7", "Bilirkişi İncelemesi",
                    "Gerekirse bilirkişi atanır", 7, 90,
                    ["bilirkişi_raporu"], ["A6"]),
                ProsedurAdimi("A8", "Karar",
                    "Mahkeme karar verir", 8, 30,
                    ["mahkeme_karari"], ["A6", "A7"]),
            ],
            DavaTuru.KİRA.value: [
                ProsedurAdimi("K1", "İhtarname Gönderimi",
                    "Kiracıya ihtarname gönderilir", 1, 7,
                    ["ihtarname", "teslim_tutanagi"]),
                ProsedurAdimi("K2", "Dava Dilekçesi",
                    "Tahliye veya alacak davası açılır", 2, 7,
                    ["dava_dilekcesi", "kira_sozlesmesi", "ihtarname"], ["K1"]),
                ProsedurAdimi("K3", "Harc ve Tebligat",
                    "Harc yatırılır, tebligat yapılır", 3, 15,
                    ["harc_makbuzu"], ["K2"]),
                ProsedurAdimi("K4", "Tahliye Kararı",
                    "Mahkeme tahliye kararı verir", 4, 60,
                    ["mahkeme_karari"], ["K3"]),
                ProsedurAdimi("K5", "İcra Takibi",
                    "Tahliye icra takibi başlatılır", 5, 30,
                    ["icra_talebi"], ["K4"]),
            ],
            DavaTuru.BOŞANMA.value: [
                ProsedurAdimi("B1", "Dava Dilekçesi",
                    "Boşanma davası dilekçesi hazırlanır", 1, 7,
                    ["dava_dilekcesi", "nufus_cuzdani", "evlilik_cuzdani", "mal_bildirimi"]),
                ProsedurAdimi("B2", "Arabulucuya Sevk",
                    "Zorunlu arabulucuya sevk", 2, 30,
                    ["arabulucu_raporu"], ["B1"]),
                ProsedurAdimi("B3", "Mahkemeye Başvuru",
                    "Anlaşmazlık varsa mahkemeye başvuru", 3, 7,
                    ["dava_dilekcesi", "arabulucu_raporu"], ["B2"]),
                ProsedurAdimi("B4", "Tebligat",
                    "Karşı tarafa tebligat", 4, 15,
                    [], ["B3"]),
                ProsedurAdimi("B5", "Duruşma",
                    "Duruşmalar yapılır", 5, 90,
                    ["duruşma_tutanağı"], ["B4"]),
                ProsedurAdimi("B6", "Karar",
                    "Boşanma kararı verilir", 6, 30,
                    ["mahkeme_karari"], ["B5"]),
            ]
        }

    def get_harita(self, dava_turu: DavaTuru) -> List[ProsedurAdimi]:
        return self.haritalar.get(dava_turu.value, [])

    def add_harita(self, dava_turu: DavaTuru, adimlar: List[ProsedurAdimi]):
        self.haritalar[dava_turu.value] = adimlar

# ============================================
# NLP - DAVA TÜRÜ TESPİTİ
# ============================================

class DavaTuruTespitici:
    def __init__(self):
        self.anahtar_kelimeler = {
            DavaTuru.ALACAK: ["alacak", "borç", "ödeme", "senet", "çek", "fatura"],
            DavaTuru.KİRA: ["kira", "kiracı", "ev sahibi", "tahliye", "kira bedeli", "konut"],
            DavaTuru.BOŞANMA: ["boşanma", "evlilik", "evli", "eş", "çocuk velayeti", "nafaka"],
            DavaTuru.İCRA: ["icra", "takip", "haciz", "tahsilat", "borçlu", "alacaklı"],
            DavaTuru.TAZMİNAT: ["tazminat", "zarar", "kaza", "maluliyet", "manevi tazminat"],
            DavaTuru.İŞ_DAVASI: ["işçi", "işveren", "kıdem", "ihbar", "fazla mesai", "iş kazası"],
            DavaTuru.CEZA: ["suç", "ceza", "tutuklama", "hapis", "adli para cezası", "şikayet"],
            DavaTuru.İDARE: ["idare", "iptal", "tam yargı", "vergi", "belediye", "kamu"],
        }

    def tespit_et(self, dilekce_metni: str) -> Tuple[DavaTuru, float]:
        metin_lower = dilekce_metni.lower()
        skorlar = {}

        for dava_turu, kelimeler in self.anahtar_kelimeler.items():
            skor = sum(1 for kelime in kelimeler if kelime in metin_lower)
            skorlar[dava_turu] = skor

        if not any(skorlar.values()):
            return DavaTuru.ALACAK, 0.3

        en_yuksek = max(skorlar, key=skorlar.get)
        toplam = sum(skorlar.values())
        guven = skorlar[en_yuksek] / toplam if toplam > 0 else 0

        return en_yuksek, round(guven, 2)

# ============================================
# UYUM MOTORU - ÇEKİRDEK
# ============================================

class ProsedurelUyumMotoru:
    def __init__(self):
        self.db = ProsedurHaritasiDB()
        self.tespitici = DavaTuruTespitici()
        self.davalar = {}

    def dava_olustur(self, dava_id: str, dilekce_metni: str, 
                     baslangic_tarihi: Optional[datetime] = None) -> Dict:
        if baslangic_tarihi is None:
            baslangic_tarihi = datetime.now()

        dava_turu, guven = self.tespitici.tespit_et(dilekce_metni)
        harita = self.db.get_harita(dava_turu)

        dava = {
            "dava_id": dava_id,
            "dava_turu": dava_turu.value,
            "guven_skoru": guven,
            "baslangic_tarihi": baslangic_tarihi,
            "prosedur_haritasi": harita,
            "mevcut_adim": 0,
            "durum": "aktif"
        }

        self.davalar[dava_id] = dava

        return {
            "dava_id": dava_id,
            "tespit_edilen_tur": dava_turu.value,
            "guven_skoru": guven,
            "toplam_adim": len(harita),
            "tahmini_sure_gun": sum(a.sure_gun for a in harita),
            "prosedur_haritasi": [(a.sira, a.adim_adi, a.sure_gun) for a in harita]
        }

    def uyum_kontrolu(self, dava_id: str, mevcut_tarih: Optional[datetime] = None) -> UyumRaporu:
        if mevcut_tarih is None:
            mevcut_tarih = datetime.now()

        if dava_id not in self.davalar:
            return UyumRaporu(dava_id, 0, UyumDurumu.HATA, 
                            kritik_hatalar=["Dava bulunamadı!"])

        dava = self.davalar[dava_id]
        harita = dava["prosedur_haritasi"]
        baslangic = dava["baslangic_tarihi"]

        uyari_mesajlari = []
        kritik_hatalar = []
        oneriler = []
        adim_raporlari = []

        toplam_adim = len(harita)
        tamamlanan = sum(1 for a in harita if a.tamamlandi)
        genel_skor = (tamamlanan / toplam_adim * 100) if toplam_adim > 0 else 0

        for adim in harita:
            adim_durumu = "bekliyor"
            risk = 0

            beklenen_bitis = baslangic + timedelta(days=adim.sure_gun)
            kalan_gun = (beklenen_bitis - mevcut_tarih).days

            if adim.tamamlandi:
                adim_durumu = "tamamlandi"
                if adim.tamamlanma_tarihi and adim.tamamlanma_tarihi > beklenen_bitis:
                    gecikme = (adim.tamamlanma_tarihi - beklenen_bitis).days
                    uyari_mesajlari.append(f"{adim.adim_adi}: {gecikme} gün gecikmeli tamamlandı")
            else:
                bagimli_tamam = all(
                    next((a for a in harita if a.adim_id == b), None).tamamlandi 
                    for b in adim.bagimli_adimlar
                ) if adim.bagimli_adimlar else True

                if not bagimli_tamam:
                    adim_durumu = "bagimli_bekleniyor"
                elif kalan_gun < 0:
                    adim_durumu = "sure_asimi"
                    kritik_hatalar.append(f"KRİTİK: {adim.adim_adi} süresi {abs(kalan_gun)} gün aşıldı!")
                    risk = 100
                    genel_skor -= 10
                elif kalan_gun <= 3:
                    adim_durumu = "acil"
                    uyari_mesajlari.append(f"{adim.adim_adi}: {kalan_gun} gün kaldı!")
                    risk = 70
                elif kalan_gun <= 7:
                    adim_durumu = "yaklasiyor"
                    uyari_mesajlari.append(f"{adim.adim_adi}: {kalan_gun} gün içinde tamamlanmalı")
                    risk = 40
                else:
                    adim_durumu = "zamaninda"
                    risk = 0

            adim_raporlari.append({
                "adim_id": adim.adim_id,
                "adim_adi": adim.adim_adi,
                "durum": adim_durumu,
                "risk": risk,
                "kalan_gun": kalan_gun if not adim.tamamlandi else 0,
                "gerekli_belgeler": adim.gerekli_belgeler
            })

        if kritik_hatalar:
            durum = UyumDurumu.KRİTİK
        elif uyari_mesajlari:
            durum = UyumDurumu.UYARİ
        else:
            durum = UyumDurumu.UYGUN

        if kritik_hatalar:
            oneriler.append("Acilen eksik adımları tamamlayın veya süre uzatması talep edin")
        if any(a["durum"] == "acil" for a in adim_raporlari):
            oneriler.append("Yaklaşan süreli adımlar için hazırlık yapın")

        genel_skor = max(0, min(100, genel_skor))

        return UyumRaporu(
            dava_id=dava_id,
            genel_skor=round(genel_skor, 1),
            durum=durum,
            uyari_mesajlari=uyari_mesajlari,
            kritik_hatalar=kritik_hatalar,
            oneriler=oneriler,
            adim_raporlari=adim_raporlari
        )

    def adim_tamamla(self, dava_id: str, adim_id: str, 
                     belgeler: List[str] = None) -> Dict:
        if dava_id not in self.davalar:
            return {"hata": "Dava bulunamadı"}

        dava = self.davalar[dava_id]
        harita = dava["prosedur_haritasi"]

        for adim in harita:
            if adim.adim_id == adim_id:
                bagimli_tamam = all(
                    next((a for a in harita if a.adim_id == b), None).tamamlandi 
                    for b in adim.bagimli_adimlar
                ) if adim.bagimli_adimlar else True

                if not bagimli_tamam:
                    return {
                        "hata": f"Önce bağımlı adımlar tamamlanmalı: {adim.bagimli_adimlar}"
                    }

                eksik_belgeler = []
                if belgeler:
                    eksik_belgeler = [b for b in adim.gerekli_belgeler if b not in belgeler]

                if eksik_belgeler:
                    return {
                        "uyari": f"Eksik belgeler: {eksik_belgeler}",
                        "tamamlandi": False
                    }

                adim.tamamlandi = True
                adim.tamamlanma_tarihi = datetime.now()

                return {
                    "mesaj": f"{adim.adim_adi} tamamlandı",
                    "tamamlandi": True,
                    "tamamlanma_tarihi": adim.tamamlanma_tarihi.isoformat()
                }

        return {"hata": "Adım bulunamadı"}

    def rapor_olustur(self, dava_id: str) -> str:
        rapor = self.uyum_kontrolu(dava_id)

        rapor_metni = f"""
╔══════════════════════════════════════════════════════════════╗
║         PROSEDÜREL UYUM MUHAFIZI - UYUM RAPORU              ║
╚══════════════════════════════════════════════════════════════╝

Dava ID: {rapor.dava_id}
Genel Uyum Skoru: {rapor.genel_skor}/100
Durum: {rapor.durum.value.upper()}
Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}

{'═' * 60}
KRİTİK HATALAR
{'═' * 60}
"""
        if rapor.kritik_hatalar:
            for hata in rapor.kritik_hatalar:
                rapor_metni += f"  ❌ {hata}\n"
        else:
            rapor_metni += "  ✅ Kritik hata bulunmuyor\n"

        rapor_metni += f"""
{'═' * 60}
UYARI MESAJLARI
{'═' * 60}
"""
        if rapor.uyari_mesajlari:
            for uyari in rapor.uyari_mesajlari:
                rapor_metni += f"  ⚠️ {uyari}\n"
        else:
            rapor_metni += "  ✅ Uyarı bulunmuyor\n"

        rapor_metni += f"""
{'═' * 60}
ÖNERİLER
{'═' * 60}
"""
        if rapor.oneriler:
            for oneri in rapor.oneriler:
                rapor_metni += f"  💡 {oneri}\n"
        else:
            rapor_metni += "  ✅ Tüm prosedürler düzgün ilerliyor\n"

        rapor_metni += f"""
{'═' * 60}
ADIM DETAYLARI
{'═' * 60}
"""
        for adim in rapor.adim_raporlari:
            durum_emoji = {
                "tamamlandi": "✅",
                "bekliyor": "⏳",
                "acil": "🚨",
                "yaklasiyor": "⚠️",
                "sure_asimi": "❌",
                "bagimli_bekleniyor": "🔗",
                "zamaninda": "✓"
            }.get(adim["durum"], "?")

            rapor_metni += f"\n  {durum_emoji} Adım {adim['adim_id']}: {adim['adim_adi']}\n"
            rapor_metni += f"     Durum: {adim['durum']} | Risk: {adim['risk']}%\n"
            if adim['kalan_gun'] > 0:
                rapor_metni += f"     Kalan süre: {adim['kalan_gun']} gün\n"
            if adim['gerekli_belgeler']:
                rapor_metni += f"     Gerekli belgeler: {', '.join(adim['gerekli_belgeler'])}\n"

        rapor_metni += f"\n{'═' * 60}\n"

        return rapor_metni


# ============================================
# KULLANIM ÖRNEĞİ
# ============================================

if __name__ == "__main__":
    motor = ProsedurelUyumMotoru()

    # Alacak davası
    dilekce = "Müvekkilim şirketine ait fatura bedelinin ödenmemesi nedeniyle alacak davası açıyoruz."
    sonuc = motor.dava_olustur("DAV-2025-001", dilekce)
    print(f"Dava oluşturuldu: {sonuc['dava_id']}")

    # Uyum kontrolü
    rapor = motor.uyum_kontrolu("DAV-2025-001")
    print(f"Uyum Skoru: {rapor.genel_skor}/100")

    # Adım tamamlama
    motor.adim_tamamla("DAV-2025-001", "A1", ["dava_dilekcesi", "vekaletname", "alacak_belgesi"])

    # Rapor
    print(motor.rapor_olustur("DAV-2025-001"))
