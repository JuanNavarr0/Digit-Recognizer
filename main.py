from model_train import build_and_train_model
from model_predict import load_model, predict_digit
import numpy as np

def main():
    print("Training the model...")
    build_and_train_model()
    
    print("Loading trained model...")
    model = load_model()
    
    sample_image = np.random.rand(28, 28)  # Replace with actual image input
    predicted_digit = predict_digit(model, sample_image)
    print(f"Predicted Digit: {predicted_digit}")

if __name__ == "__main__":
    main()