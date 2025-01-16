import numpy as np
from collections import Counter

class InformationGain:
    @staticmethod
    def calculate_entropy(labels):
        """
        Verilen etiketler için entropi hesaplar
        """
        if len(labels) == 0:
            return 0
        
        # Etiketlerin frekanslarını hesapla
        counter = Counter(labels)
        probabilities = [count / len(labels) for count in counter.values()]
        
        # Entropi formülü: -sum(p_i * log2(p_i))
        entropy = -sum(p * np.log2(p) for p in probabilities)
        return entropy

    @staticmethod
    def calculate_information_gain(S, A, data):
        """
        S: Tüm veri seti
        A: Özellik (feature) sütunu
        data: Hedef değişken (target) sütunu
        """
        # Tüm veri setinin entropisi
        total_entropy = InformationGain.calculate_entropy(data)
        
        # A özelliğinin değerlerine göre alt kümelerin entropisi
        values = Counter(A)
        weighted_entropy = 0
        
        for value in values:
            # Her bir değer için alt küme oluştur
            subset_indices = [i for i in range(len(A)) if A[i] == value]
            subset = [data[i] for i in subset_indices]
            
            # Alt kümenin ağırlıklı entropisini hesapla
            prob = len(subset) / len(data)
            weighted_entropy += prob * InformationGain.calculate_entropy(subset)
        
        # Bilgi kazancı = Total Entropy - Weighted Entropy
        information_gain = total_entropy - weighted_entropy
        return information_gain

# Örnek kullanım
if __name__ == "__main__":
    # Örnek veri seti
    hava = ["Güneşli", "Güneşli", "Yağmurlu", "Güneşli"]
    piknik = ["Evet", "Evet", "Hayır", "Evet"]
    
    # Information Gain hesaplama
    ig = InformationGain()
    gain = ig.calculate_information_gain(piknik, hava, piknik)
    print(f"Hava özelliği için Bilgi Kazancı: {gain:.3f}") 