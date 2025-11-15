import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from textblob import TextBlob
import jiwer

# ------------------------ LOAD CNN MODEL ------------------------
MODEL_PATH = "model_cnn.h5"

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading CNN model: {e}")
else:
    st.warning("❌ CNN model not found! Handwriting analysis will be disabled.")

# ------------------------ STREAMLIT UI ------------------------
st.set_page_config(page_title="Dyslexia Detection (Combined)")
st.title("🧠 Dyslexia Detection Web App (Combined)")

st.write("""
Upload a handwriting sample or input text to check for dyslexia indicators.
""")

# ------------------------ HANDWRITING ANALYSIS ------------------------
st.subheader("1️⃣ Handwriting Analysis")
image = st.file_uploader("Upload handwriting sample", type=["jpg", "jpeg", "png"])

if image:
    img = Image.open(image).convert("RGB")
    st.image(img, caption="Uploaded Sample", width=300)

    if st.button("Analyze Handwriting"):
        if model:
            resized = img.resize((128, 128))
            arr = np.array(resized) / 255.0
            arr = arr.reshape(1, 128, 128, 3)

            pred = model.predict(arr)[0][0]
            st.write(f"Model Score: {pred:.4f}")

            if pred >= 0.5:
                st.success("✅ Likely **non-dyslexic handwriting** detected.")
            else:
                st.warning("⚠ Likely *dyslexic handwriting* detected.")
        else:
            st.error("CNN model not loaded, cannot analyze handwriting.")

# ------------------------ TEXT ANALYSIS ------------------------
st.subheader("2️⃣ Text Input Analysis")
text_input = st.text_area("Type or paste a paragraph of text:")

if st.button("Analyze Text") and text_input:

    # --- 1. Spelling Correction ---
    blob = TextBlob(text_input)
    corrected = str(blob.correct())

    # --- 2. Spelling Accuracy ---
    input_words = text_input.split()
    corrected_words = corrected.split()

    # Count how many words were correct already
    correct_count = sum([1 for iw, cw in zip(input_words, corrected_words) if iw.lower() == cw.lower()])
    spelling_accuracy = (correct_count / max(len(input_words), 1)) * 100

    # --- 3. Grammar / Readability Score ---
    # Placeholder metric: fewer corrections = better grammar
    num_corrections = len([1 for iw, cw in zip(input_words, corrected_words) if iw.lower() != cw.lower()])
    grammar_score = max(0, 100 - (num_corrections / max(len(input_words), 1)) * 100)

    # --- 4. Dyslexia-related traits ---
    # Rough approximations:
    traits = {}
    traits['Frequent misspellings'] = num_corrections > 0
    traits['Short / simple words'] = all(len(w) <= 5 for w in input_words)
    traits['Low text complexity'] = len(input_words) < 5

    # --- 5. Display results ---
    st.markdown("**Spelling & Grammar Correction:**")
    st.write(corrected)

    st.markdown("**Spelling Accuracy:**")
    st.write(f"{spelling_accuracy:.2f}%")

    st.markdown("**Grammar Score:**")
    st.write(f"{grammar_score:.2f}% (higher is better)")

    st.markdown("**Dyslexia Trait Flags:**")
    for k, v in traits.items():
        st.write(f"- {k}: {'Yes' if v else 'No'}")

# ------------------------ DYSLEXIA TRAITS ANALYSIS ------------------------
st.subheader("3️⃣ Dyslexia Traits & Suggestions")

traits = []
suggestions = []

if image or text_input:
    # Simple trait evaluation based on text & image (simulated)
    # You can later replace with ML-based scoring
    if text_input:
        text_len = len(text_input.split())
        if text_len < 5 or any(len(word) > 15 for word in text_input.split()):
            traits.append("Difficulty with reading / long words")
            suggestions.append("Practice reading daily and break words into smaller parts.")

        if any(len(word) > 10 for word in text_input.split()):
            traits.append("Spelling difficulties")
            suggestions.append("Use spelling apps or phonics exercises.")

    if image:
        traits.append("Potential handwriting inconsistencies")
        suggestions.append("Handwriting practice and occupational therapy may help.")

    if not traits:
        traits.append("No clear traits detected from input")
        suggestions.append("Keep practicing reading, writing, and attention exercises.")

    st.markdown("**Detected Dyslexia Traits:**")
    for t in traits:
        st.write(f"• {t}")

    st.markdown("**Suggestions / Next Steps:**")
    for s in suggestions:
        st.write(f"• {s}")

# ------------------------ INPUT WARNING ------------------------
if not image and not text_input:
    st.warning("Please provide at least a handwriting sample or text input for analysis.")

# ------------------------ AUDIO FEATURE PLACEHOLDER ------------------------
st.subheader("4️⃣ Audio / Pronunciation Analysis (Coming Soon)")
st.info("Audio analysis will be implemented in the next version.")
