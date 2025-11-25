# scripts/train.py

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# ----------------- Ensure result folders exist -----------------
os.makedirs('./models', exist_ok=True)
os.makedirs('./results/graphs', exist_ok=True)

# ----------------- Load preprocessed data -----------------
X_train = np.load("./dataset/processed/X_train.npy")
y_train = np.load("./dataset/processed/y_train.npy")

X_test = np.load("./dataset/processed/X_test.npy")
y_test = np.load("./dataset/processed/y_test.npy")

# ----------------- Build CNN model -----------------
model = Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(25, activation='softmax')  # 25 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------- Train model -----------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# ----------------- Save the trained model -----------------
model.save("./models/model.h5")

# ----------------- Plot Accuracy -----------------
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./results/graphs/accuracy.png')
plt.close()

# ----------------- Plot Loss -----------------
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./results/graphs/loss.png')
plt.close()

print("Model trained, saved, and plots generated successfully!")
