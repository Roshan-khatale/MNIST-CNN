from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np

_, (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_test = to_categorical(y_test)

model = load_model("models/mnist_cnn.h5")
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")