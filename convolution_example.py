import numpy as np

def apply_convolution(input_volume, filter_w0, filter_w1, bias_b0, bias_b1):
    """
    Konvolüsyon işlemini uygular
    Args:
        input_volume: Giriş görüntüsü (7x7x3)
        filter_w0: İlk filtre (3x3x3)
        filter_w1: İkinci filtre (3x3x3)
        bias_b0: İlk filtrenin bias değeri
        bias_b1: İkinci filtrenin bias değeri
    Returns:
        output: Çıkış görüntüsü (3x3x2)
    """
    # Çıkış boyutunu hesapla: [(n + 2p - f) / s + 1] x [(n + 2p - f) / s + 1]
    # n=5 (giriş boyutu), p=1 (padding), f=3 (filtre boyutu), s=2 (stride)
    output_height = output_width = 3  # (5 + 2*1 - 3) / 2 + 1 = 3
    output = np.zeros((output_height, output_width, 2))
    
    # Padding ekle
    padded_input = np.pad(input_volume, ((1,1), (1,1), (0,0)), mode='constant')
    
    # Her çıkış pozisyonu için
    for i in range(output_height):
        for j in range(output_width):
            # Giriş penceresini al (3x3x3)
            h_start = i * 2  # stride = 2
            h_end = h_start + 3  # filtre boyutu = 3
            w_start = j * 2
            w_end = w_start + 3
            input_slice = padded_input[h_start:h_end, w_start:w_end, :]
            
            # İlk filtre için konvolüsyon
            conv_sum_w0 = np.sum(input_slice * filter_w0) + bias_b0
            output[i, j, 0] = conv_sum_w0
            
            # İkinci filtre için konvolüsyon
            conv_sum_w1 = np.sum(input_slice * filter_w1) + bias_b1
            output[i, j, 1] = conv_sum_w1
    
    return output

def relu_activation(x):
    """ReLU aktivasyon fonksiyonu"""
    return np.maximum(0, x)

# Örnek giriş verisi ve filtreler
input_volume = np.array([
    [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
    [[0,0,0], [1,0,2], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
    [[0,1,0], [2,0,1], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
    [[0,1,0], [2,2,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
    [[0,2,0], [0,2,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
    [[0,2,1], [2,2,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
    [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]
])

# Filtreler
filter_w0 = np.array([
    [[-1,0,1], [0,0,1], [1,-1,1]],
    [[-1,0,1], [1,-1,1], [0,1,0]],
    [[-1,1,1], [1,1,0], [0,-1,0]]
])

filter_w1 = np.array([
    [[0,1,-1], [0,-1,0], [0,-1,1]],
    [[-1,0,0], [1,-1,0], [1,-1,0]],
    [[-1,1,-1], [0,-1,-1], [1,0,0]]
])

# Bias değerleri
bias_b0 = 1
bias_b1 = 0

# Konvolüsyon işlemini uygula
output = apply_convolution(input_volume, filter_w0, filter_w1, bias_b0, bias_b1)

# ReLU aktivasyonunu uygula
output_relu = relu_activation(output)

print("Konvolüsyon çıktısı:")
print(output)
print("\nReLU sonrası çıktı:")
print(output_relu) 