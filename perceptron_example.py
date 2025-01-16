# Tek Katmanlı Algılayıcı (TKA) Örneği
# Elma ve Portakal sınıflandırması

# Giriş verileri (x1, x2)
X1 = [1, 0]  # Portakal örneği
X2 = [0, 1]  # Elma örneği

# Beklenen çıktılar
B1 = 1  # Portakal için
B2 = 0  # Elma için

# Başlangıç parametreleri
w1 = 1  # İlk ağırlık
w2 = 2  # İkinci ağırlık
threshold = -1  # Eşik değeri (Φ)
learning_rate = 0.5  # Öğrenme katsayısı (λ)

def calculate_net(x, w1, w2):
    return w1 * x[0] + w2 * x[1]

# İterasyonlar
iteration = 1
max_iterations = 14  # Maksimum iterasyon sayısı

while iteration <= max_iterations:
    print(f"\n{iteration}. iterasyon:")
    
    # 1. örnek için (Portakal)
    print(f"\n1. örnek gösteriliyor:")
    net1 = calculate_net(X1, w1, w2)
    C1 = 1 if net1 > threshold else 0
    print(f"NET = w1*x1 + w2*x2 = {w1}*{X1[0]} + {w2}*{X1[1]} = {net1}")
    print(f"NET > Φ olduğundan Ç = {C1}")
    
    if C1 != B1:
        w1 = w1 + learning_rate * X1[0]
        w2 = w2 + learning_rate * X1[1]
        print("Ağırlıklar güncellendi:")
        print(f"w1 = {w1}")
        print(f"w2 = {w2}")
    else:
        print("Ağırlıklar değiştirilmez.")
    
    # 2. örnek için (Elma)
    print(f"\n2. örnek gösteriliyor:")
    net2 = calculate_net(X2, w1, w2)
    C2 = 1 if net2 > threshold else 0
    print(f"NET = w1*x1 + w2*x2 = {w1}*{X2[0]} + {w2}*{X2[1]} = {net2}")
    print(f"NET > Φ olduğundan Ç = {C2}")
    
    if C2 != B2:
        w1 = w1 - learning_rate * X2[0]
        w2 = w2 - learning_rate * X2[1]
        print("Ağırlıklar güncellendi:")
        print(f"w1 = {w1}")
        print(f"w2 = {w2}")
    else:
        print("Ağırlıklar değiştirilmez.")
    
    # Eğer her iki örnek de doğru sınıflandırılıyorsa
    if C1 == B1 and C2 == B2:
        print("\nÖğrenme tamamlandı!")
        break
        
    iteration += 1

print(f"\nSon ağırlık değerleri:")
print(f"w1 = {w1}")
print(f"w2 = {w2}")

# Test
print("\nTest sonuçları:")
print("1. örnek için:")
test_net1 = calculate_net(X1, w1, w2)
test_C1 = 1 if test_net1 > threshold else 0
print(f"NET = {test_net1} > Φ ve Ç = B1 = {test_C1}")

print("\n2. örnek için:")
test_net2 = calculate_net(X2, w1, w2)
test_C2 = 1 if test_net2 > threshold else 0
print(f"NET = {test_net2} > Φ ve Ç = B2 = {test_C2}") 