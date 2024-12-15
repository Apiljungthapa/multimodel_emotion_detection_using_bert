# app.py
import streamlit as st
from multimodel import (
    load_text_model,
    load_image_model,
    load_audio_model,
    predict_text_emotion,
    predict_image_emotion,
    predict_audio_emotion,
    calculate_weighted_average,
)

# Load models (paths should be adjusted as per your setup)
text_model_path = r"C:\Users\apilt\OneDrive\Desktop\AI CLZ\text_emotion_model\checkpoint-1000"
image_model_path = r"C:\Users\apilt\OneDrive\Desktop\AI CLZ\image_emotion_model"
audio_model_path = r"C:\Users\apilt\OneDrive\Desktop\AI CLZ\speech_emotion_model"

# Load all models
tokenizer, text_model = load_text_model(text_model_path)
image_model, image_transform = load_image_model(image_model_path)
audio_model, audio_processor = load_audio_model(audio_model_path)

# Streamlit UI
st.title("Multimodal Emotion Detection")

# Text Input
st.header("Enter Text")
text_input = st.text_area("Text Input", "I am happy.")
if st.button("Generate Text Emotion"):
    text_results = predict_text_emotion([text_input], text_model, tokenizer)
    st.write("Top 3 Predicted Emotions for Text:")
    for result in text_results[0]:
        st.write(f"{result[0]}: {result[1]:.4f}")

# Image Input
st.header("Upload Image")
image_input = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
if image_input is not None:
    image_path = image_input
    if st.button("Generate Image Emotion"):
        image_results = predict_image_emotion(image_path, image_model, image_transform)
        st.write("Top 3 Predicted Emotions for Image:")
        for result in image_results:
            st.write(f"{result[0]}: {result[1]:.4f}")

# Audio Input
st.header("Upload Audio")
audio_input = st.file_uploader("Upload an Audio", type=["wav"])
if audio_input is not None:
    audio_path = audio_input
    if st.button("Generate Audio Emotion"):
        audio_results = predict_audio_emotion(audio_path, audio_model, audio_processor)
        st.write("Top 3 Predicted Emotions for Audio:")
        for result in audio_results:
            st.write(f"{result[0]}: {result[1]:.4f}")

# Final Results
if st.button("Generate Overall Prediction"):
    results = []
    
    # Add predictions to results
    if text_input:
        results.append(predict_text_emotion([text_input], text_model, tokenizer)[0])
    if image_input:
        results.append(predict_image_emotion(image_path, image_model, image_transform))
    if audio_input:
        results.append(predict_audio_emotion(audio_path, audio_model, audio_processor))

    # Calculate weighted average
    weighted_results = calculate_weighted_average(results)
    st.write("Overall Prediction with Weighted Average:")
    for result in weighted_results:
        for label, prob in result:
            st.write(f"{label}: {prob:.4f}")
