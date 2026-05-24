"""
Handwritten Digit Recognition using TensorFlow and MNIST Dataset

Author: Arsh Shaikh

Description:
A neural network model trained on the MNIST dataset
to recognize handwritten digits from custom images.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Toggle model training
TRAIN_MODEL = False

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model only if needed
if TRAIN_MODEL:
    model.fit(x_train, y_train, epochs=4)
    model.save('handwritten_digits.keras')

# Load saved model
model = tf.keras.models.load_model('handwritten_digits.keras')

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)

print(f"\nModel Loss: {loss:.4f}")
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Predict custom digit images
image_number = 1

while os.path.isfile(f"digits/digit{image_number}.png"):

    try:
        # Load image in grayscale
        img = cv2.imread(
            f"digits/digit{image_number}.png",
            cv2.IMREAD_GRAYSCALE
        )
        if img is None:
         raise ValueError("Image not found or unreadable.")

        # Resize to MNIST format
        img = cv2.resize(img, (28, 28))

        # Invert and normalize image
        img = np.invert(img)
        img = img / 255.0

        # Reshape for model input
        img = img.reshape(1, 28, 28)

        # Predict digit
        prediction = model.predict(img)

        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        print(f"Digit {image_number}")
        print(f"Predicted Digit: {predicted_digit}")
        print(f"Confidence: {confidence:.2f}%\n")

        # Display image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(f"Predicted: {predicted_digit}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error with digit{image_number}.png: {e}")

    finally:
        image_number += 1