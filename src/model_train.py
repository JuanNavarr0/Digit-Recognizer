import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from data_loader import load_and_preprocess_data

def build_and_train_model():
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    
    model.fit(train_images, train_labels, epochs=30, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
    
    model.save("models/mnist_model.keras")
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    build_and_train_model()