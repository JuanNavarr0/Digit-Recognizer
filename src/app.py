import gradio as gr
from model_predict import load_model, predict_digit

model = load_model()

def classify_image(image):
    return int(predict_digit(model, image))

interface = gr.Interface(fn=classify_image, inputs="sketchpad", outputs="label")
interface.launch()