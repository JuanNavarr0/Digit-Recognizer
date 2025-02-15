import numpy as np
import tensorflow as tf

def load_model():
    return tf.keras.models.load_model("models/mnist_model.keras")

def predict_digit(model, image):
    image = image.reshape(1, 28, 28, 1).astype('float32') / 255
    prediction = model.predict(image)
    return np.argmax(prediction)

if __name__ == "__main__":
    model = load_model()
    print("Model loaded successfully.")