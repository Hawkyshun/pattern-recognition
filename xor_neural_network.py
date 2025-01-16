import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self):
        # Öğrenme parametreleri
        self.learning_rate = 0.5  # lambda
        self.momentum = 0.8       # alpha
        
        # Ağırlıkların başlangıç değerleri (görseldeki A^i matrisi)
        self.weights_input_hidden = np.array([
            [0.129952, 0.570345],
            [-0.923123, -0.328932]
        ])
        
        # Ara katman - çıktı katmanı ağırlıkları (görseldeki A^a)
        self.weights_hidden_output = np.array([0.164732, 0.752621])
        
        # Eşik değerleri (görseldeki beta^a ve beta^c)
        self.bias_hidden = np.array([0.341332, -0.115223])
        self.bias_output = np.array([-0.993423])
        
        # Önceki değişimleri saklamak için
        self.prev_delta_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.prev_delta_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.prev_delta_bias_hidden = np.zeros_like(self.bias_hidden)
        self.prev_delta_bias_output = np.zeros_like(self.bias_output)

    def forward(self, inputs):
        # İleri besleme
        self.hidden_net = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_net)
        
        self.output_net = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_net)
        
        return self.output

    def backward(self, inputs, target):
        # Çıktı katmanındaki hata
        output_error = target - self.output
        output_delta = output_error * sigmoid_derivative(self.output)
        
        # Ara katmandaki hata
        hidden_error = output_delta * self.weights_hidden_output
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # Ağırlık güncellemeleri
        # Çıktı katmanı ağırlıkları
        delta_weights_hidden_output = (self.learning_rate * output_delta * self.hidden_output + 
                                     self.momentum * self.prev_delta_weights_hidden_output)
        
        # Ara katman ağırlıkları
        delta_weights_input_hidden = (self.learning_rate * np.outer(inputs, hidden_delta) + 
                                    self.momentum * self.prev_delta_weights_input_hidden)
        
        # Bias güncellemeleri
        delta_bias_output = self.learning_rate * output_delta
        delta_bias_hidden = self.learning_rate * hidden_delta
        
        # Ağırlıkları güncelle
        self.weights_hidden_output += delta_weights_hidden_output
        self.weights_input_hidden += delta_weights_input_hidden
        self.bias_output += delta_bias_output
        self.bias_hidden += delta_bias_hidden
        
        # Önceki değişimleri sakla
        self.prev_delta_weights_hidden_output = delta_weights_hidden_output
        self.prev_delta_weights_input_hidden = delta_weights_input_hidden
        self.prev_delta_bias_output = delta_bias_output
        self.prev_delta_bias_hidden = delta_bias_hidden
        
        return np.abs(output_error)

    def train(self, X, y, epochs=10000, error_threshold=0.03):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                output = self.forward(X[i])
                error = self.backward(X[i], y[i])
                total_error += error
            
            mse = total_error / len(X)
            if mse <= error_threshold:
                print(f"Eğitim {epoch} iterasyonda tamamlandı. MSE: {mse}")
                break
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}, MSE: {mse}")

# XOR problemi için veri seti
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 0])

# Ağı oluştur ve eğit
nn = NeuralNetwork()
print("Eğitim başlıyor...")
nn.train(X, y)

# Test
print("\nTest sonuçları:")
for i in range(len(X)):
    prediction = nn.forward(X[i])
    print(f"Girdi: {X[i]}, Hedef: {y[i]}, Tahmin: {prediction[0]:.6f}") 