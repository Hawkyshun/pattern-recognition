import numpy as np

class ARTNetwork:
    def __init__(self, input_size, vigilance=0.8):
        self.input_size = input_size
        self.vigilance = vigilance
        self.categories = []
        self.weights = np.ones((0, input_size))  # Başlangıçta boş ağırlık matrisi
        
    def normalize(self, x):
        return x / (np.sum(x) + 1e-10)
    
    def similarity(self, x, w):
        return np.sum(np.minimum(x, w)) / (np.sum(x) + 1e-10)
    
    def train(self, X):
        clusters = []
        for x in X:
            x = np.array(x)
            if len(self.weights) == 0:
                # İlk kategoriyi oluştur
                self.weights = np.vstack([self.weights, x])
                clusters.append(0)
                continue
                
            # Tüm kategorilerle benzerliği hesapla
            similarities = [self.similarity(x, w) for w in self.weights]
            best_match = np.argmax(similarities)
            
            if similarities[best_match] >= self.vigilance:
                # Mevcut kategoriye ekle
                self.weights[best_match] = np.minimum(x, self.weights[best_match])
                clusters.append(best_match)
            else:
                # Yeni kategori oluştur
                self.weights = np.vstack([self.weights, x])
                clusters.append(len(self.weights) - 1)
                
        return clusters

# Test verisi ve farklı vigilance değerleri için test
if __name__ == "__main__":
    # Örnek veri seti
    X = np.array([
        [1, 0, 1],  # M1
        [1, 1, 0],  # M2
        [0, 0, 1]   # M3
    ])
    
    # ρ > 0.8 için test
    print("Test için ρ = 0.9 (>0.8):")
    art_high = ARTNetwork(input_size=3, vigilance=0.9)
    clusters_high = art_high.train(X)
    print(f"Oluşan kümeler: {clusters_high}")
    print(f"Küme sayısı: {len(np.unique(clusters_high))}")
    
    print("\nTest için ρ = 0.7 (<0.8):")
    art_low = ARTNetwork(input_size=3, vigilance=0.7)
    clusters_low = art_low.train(X)
    print(f"Oluşan kümeler: {clusters_low}")
    print(f"Küme sayısı: {len(np.unique(clusters_low))}") 