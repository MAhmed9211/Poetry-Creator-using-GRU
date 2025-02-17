import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# File paths
DATASET_PATH = "Roman-Urdu-Poetry.csv"
MODEL_PATH = "best_gru_model.h5"

# Load dataset
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file '{DATASET_PATH}' not found!")
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    return df["Poetry"].astype(str).tolist()

# Clean text
def clean_text(text):
    text = re.sub(r"\n", " ", text)  # Replace newlines with space
    text = re.sub(r"[^a-zA-Z0-9āçéíóúñüĀÇÉÍÓÚÑÜāñ' ]", "", text)  # Remove special chars
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text.lower()

# Prepare dataset
poetry_texts = load_dataset()
cleaned_poetry = [clean_text(poem) for poem in poetry_texts]
corpus = " ".join(cleaned_poetry)

# Character mappings
chars = sorted(set(corpus))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for c, i in char_to_index.items()}

# Convert text to sequences
sequences = [char_to_index[c] for c in corpus]
SEQ_LENGTH = 40  # Length of input sequences
X, y = [], []
for i in range(len(sequences) - SEQ_LENGTH):
    X.append(sequences[i:i+SEQ_LENGTH])
    y.append(sequences[i+SEQ_LENGTH])

X = np.array(X)
y = to_categorical(y, num_classes=len(chars))

# Define GRU model
def build_model():
    model = Sequential([
        Embedding(input_dim=len(chars), output_dim=64, input_length=SEQ_LENGTH),
        GRU(128, return_sequences=True),
        GRU(128),
        Dense(len(chars), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load or train model
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    print("Training new model...")
    model = build_model()
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='loss', save_best_only=True)
    model.fit(X, y, epochs=50, batch_size=64, callbacks=[checkpoint])
    print("Model saved as 'best_gru_model.h5'")

# Generate poetry with temperature sampling
def generate_poetry(seed_text, length=200, temperature=1.0):
    result = seed_text
    for _ in range(length):
        sequence = [char_to_index.get(c, 0) for c in result[-SEQ_LENGTH:]]
        sequence = np.array(sequence).reshape(1, -1)
        prediction = model.predict(sequence, verbose=0)[0]
        prediction = np.log(prediction + 1e-8) / temperature  # Adjust temperature
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        next_char = index_to_char[np.random.choice(len(chars), p=prediction)]
        result += next_char
    return result

# Example
print(generate_poetry("\n\n(1)ishq ek mashal hai"),"\n\n")
print(generate_poetry("(2)ishq ek mashal hai"),"\n\n")
print(generate_poetry("(3)ishq ek mashal hai"),"\n\n")
print(generate_poetry("\n\n(4)ishq ek mashal hai"),"\n\n")