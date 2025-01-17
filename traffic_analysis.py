"""
Elazığ Trafik Analiz Sistemi
----------------------------
Bu program, şehir güvenlik kameralarından alınan görüntüleri kullanarak araç tespiti ve sayımı yapar.
Kullanılan Teknolojiler:
- YOLO (You Only Look Once): Gerçek zamanlı nesne tespiti
- DeepSORT: Araç takibi ve yeniden tanımlama
- OpenCV: Görüntü işleme ve görselleştirme
"""

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from collections import defaultdict

class TrafikAnalizci:
    def __init__(self):
        """
        Trafik analiz sistemini başlatır ve gerekli bileşenleri yükler.
        - YOLO modeli: Araç tespiti için önceden eğitilmiş model
        - DeepSORT: Araç takibi için kullanılan algoritma
        - Sayaçlar: Farklı araç türlerinin sayımı için
        """
        # YOLO modelini yükle (önceden eğitilmiş ağırlıklar)
        self.yolo = YOLO('yolov8n.pt')
        
        # DeepSORT takip algoritmasını başlat (30 kare boyunca takip eder)
        self.tracker = DeepSort(max_age=30)
        
        # Araç sayaçları ve türleri
        self.arac_sayilari = defaultdict(int)
        self.arac_turleri = {
            2: 'otomobil',
            5: 'kamyon',
            3: 'motosiklet',
            7: 'otobüs'
        }
    
    def kare_isle(self, kare):
        """
        Her bir video karesini işler ve araç tespiti yapar.
        
        Parametreler:
            kare: numpy.ndarray - İşlenecek video karesi
            
        Dönüş:
            numpy.ndarray - İşlenmiş ve görselleştirilmiş video karesi
        """
        # YOLO ile araç tespiti yap
        sonuclar = self.yolo(kare)[0]
        tespitler = []
        
        # Her bir tespit edilen nesne için
        for r in sonuclar.boxes.data.tolist():
            x1, y1, x2, y2, guven, sinif_id = r
            # Eğer tespit güvenilirliği %50'den yüksekse ve araç sınıfındaysa
            if guven > 0.5 and int(sinif_id) in self.arac_turleri:
                tespitler.append([x1, y1, x2, y2, guven])
        
        # DeepSORT ile araç takibi yap
        takip_sonuclari = self.tracker.update(np.array(tespitler), kare)
        
        # Tespit edilen her araç için görselleştirme ve sayım
        for takip in takip_sonuclari:
            takip_id = takip[4]
            x1, y1, x2, y2 = takip[:4]
            
            # Aracı çerçeve içine al
            cv2.rectangle(kare, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Araç ID'sini göster
            cv2.putText(kare, f"Araç ID: {takip_id}", (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Yeni tespit edilen aracı say
            if takip_id not in self.arac_sayilari:
                self.arac_sayilari[takip_id] = 1
        
        return kare
    
    def istatistikleri_al(self):
        """
        Anlık trafik istatistiklerini hesaplar.
        
        Dönüş:
            dict - Toplam araç sayısı ve zaman damgası
        """
        toplam_arac = len(self.arac_sayilari)
        return {
            "toplam_arac": toplam_arac,
            "zaman": cv2.getTickCount() / cv2.getTickFrequency()
        }

def main():
    """
    Ana program döngüsü. Video akışını başlatır ve her kareyi işler.
    """
    # Trafik analizcisini başlat
    analizci = TrafikAnalizci()
    
    # Video kaynağını aç
    video = cv2.VideoCapture("traffic_video.mp4")
    
    print("Trafik Analiz Sistemi Başlatıldı...")
    print("Çıkış için 'q' tuşuna basın")
    
    while video.isOpened():
        ret, kare = video.read()
        if not ret:
            break
            
        # Kareyi işle ve araçları tespit et
        islenmis_kare = analizci.kare_isle(kare)
        
        # İstatistikleri al
        istatistikler = analizci.istatistikleri_al()
        
        # Ekranda göster
        cv2.putText(islenmis_kare, f"Toplam Araç: {istatistikler['toplam_arac']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Elazığ Trafik Analiz Sistemi", islenmis_kare)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    video.release()
    cv2.destroyAllWindows()
    print("Program sonlandırıldı.")

if __name__ == "__main__":
    main() 