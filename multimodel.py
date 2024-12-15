# multimodel.py
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, DistilBertTokenizer, DistilBertForSequenceClassification
from PIL import Image
from transformers import ViTForImageClassification
from torchvision import transforms
import torch.nn.functional as F
import librosa
import numpy as np

# Load text model
def load_text_model(model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# Load image model
def load_image_model(model_path):
    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return model, transform

# Load audio model
def load_audio_model(model_path):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model.eval()
    return model, processor

# Process text input for emotion prediction
def predict_text_emotion(texts, model, tokenizer):
    label_map = {0: "sadness", 3: "anger", 2: "love", 5: "surprise", 4: "fear", 1: "joy"}
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top_k = 3
    top_probs, top_labels = torch.topk(probabilities, top_k, dim=-1)
    results = []
    for probs, labels in zip(top_probs, top_labels):
        results.append([(label_map[label.item()], prob.item()) for label, prob in zip(labels, probs)])
    return results

# Process image input for emotion prediction
def predict_image_emotion(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    top_k = 3
    top_prob, top_class_idx = torch.topk(probabilities, top_k, dim=1)
    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    top_classes_with_probabilities = [(class_labels[idx], prob.item()) for idx, prob in zip(top_class_idx[0], top_prob[0])]
    return top_classes_with_probabilities

# Process audio input for emotion prediction
def predict_audio_emotion(audio_path, model, processor):
    label_map = {0: "fear", 1: "angry", 2: "disgust", 3: "neutral", 4: "sad", 5: "ps", 6: "happy"}
    speech, sample_rate = librosa.load(audio_path, sr=16000)
    max_length = 16000 * 30
    inputs = processor(speech, sampling_rate=16000, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_3_labels = [label_map[idx] for idx in top_3_indices]
    top_3_probs = [probabilities[idx] for idx in top_3_indices]
    top_3_results = list(zip(top_3_labels, top_3_probs))
    return top_3_results

# Calculate weighted average of predictions
def calculate_weighted_average(results):
    weighted_results = []
    for result in results:
        sum_probs = sum([prob for _, prob in result])
        weighted_result = [(label, prob / sum_probs) for label, prob in result]
        weighted_results.append(weighted_result)
    return weighted_results
