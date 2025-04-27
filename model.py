import numpy as np
class Model:
    def __init__(self, input_size=9, hidden_size=1000, output_size=1):
        self.w1 = np.random.randn(hidden_size, input_size) * 0.01  # (1000, 9)
        self.w2 = np.random.randn(output_size, hidden_size) * 0.01  # (1, 1000)

    def predict(self, inputs):
        x = inputs.T  # (9, 534) → (534, 9)
        A1 = np.maximum(0, self.w1 @ x.T)  # (1000, 534)
        A2 = 1 / (1 + np.exp(-(self.w2 @ A1)))  # (1, 534)
        return A1, A2

    def update_weights_for_one_epoch(self, inputs, outputs, learning_rate):
        x = inputs.T  # (9, 534) → (534, 9)
        y_true = outputs.reshape(1, -1)  # (534,) → (1, 534)
        
        A1, A2 = self.predict(inputs)  # A1: (1000, 534), A2: (1, 534)
        n = x.shape[0]  # 534
        
        dA2 = (A2 - y_true) / n  # (1, 534)
        relu_gradient = np.where(A1 > 0, 1, 0)  # (1000, 534)
        
        # Update W2: dW2 = dA2 @ A1.T
        self.w2 -= learning_rate * (dA2 @ A1.T)  # (1, 1000)
        
        # Update W1: dW1 = (W2.T @ dA2 * relu_gradient) @ x
        dA1 = (self.w2.T @ dA2) * relu_gradient  # (1000, 534)
        self.w1 -= learning_rate * (dA1 @ x)  # (1000, 9)

    def fit(self, inputs, outputs, learning_rate, epochs=100):
        while epochs > 0 :
            self.update_weights_for_one_epoch(inputs, outputs, learning_rate)
            epochs -= 1 

