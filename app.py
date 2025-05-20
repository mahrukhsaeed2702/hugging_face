from transformers import pipeline
import gradio as gr

# Force framework to PyTorch (no TF/Keras dependency)
model = pipeline("summarization", model="google-t5/t5-small", framework="pt")

def predict(prompt):
    summary = model(prompt)[0]["summary_text"]
    return summary

# Gradio interface
with gr.Interface(predict, "textbox", "text") as interface:
    interface.launch()
