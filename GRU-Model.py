import streamlit as st
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.utils import to_categorical

# File paths
DATASET_PATH = "Roman-Urdu-Poetry.csv"
MODEL_PATH = "best_gru_model.h5"

# Load dataset
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset file '{DATASET_PATH}' not found!")
        return []
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    return df["Poetry"].astype(str).tolist()

# Clean text
def clean_text(text):
    text = re.sub(r"\n", " ", text)  # Replace newlines with space
    text = re.sub(r"[^a-zA-Z0-9āçéíóúñüĀÇÉÍÓÚÑÜāñ' ]", "", text)  # Remove special chars
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text.lower()

# Load poetry dataset
poetry_texts = load_dataset()
cleaned_poetry = [clean_text(poem) for poem in poetry_texts]
corpus = " ".join(cleaned_poetry)

# Character mappings
chars = sorted(set(corpus))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for c, i in char_to_index.items()}

# Define GRU model
def build_model():
    model = Sequential([
        Embedding(input_dim=len(chars), output_dim=64, input_length=40),
        GRU(128, return_sequences=True),
        GRU(128),
        Dense(len(chars), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load or train model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = build_model()
    st.warning("No pre-trained model found. Please train the model first.")

# Generate poetry function
def generate_poetry(seed_text, length=200, temperature=1.0):
    result = seed_text
    for _ in range(length):
        sequence = [char_to_index.get(c, 0) for c in result[-40:]]
        sequence = np.array(sequence).reshape(1, -1)
        prediction = model.predict(sequence, verbose=0)[0]
        prediction = np.log(prediction + 1e-8) / temperature
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        next_char = index_to_char[np.random.choice(len(chars), p=prediction)]
        result += next_char
    return result

# Streamlit UI
st.title("Roman Urdu Poetry Generator")

seed_text = st.text_input("Enter a seed text to generate poetry:")

temperature = st.slider("Select temperature (creativity level):", 0.1, 2.0, 1.0, 0.1)

generate_button = st.button("Generate Poetry")

if generate_button:
    generated_poetry = generate_poetry(seed_text, length=200, temperature=temperature)
    st.subheader("Generated Poetry:")
    st.write(generated_poetry)