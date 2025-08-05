# 🎶 Sentence2Song

**Sentence2Song** is an algorithmic music generation system that creates emotionally expressive compositions from user-inputted text. By integrating sentiment analysis, musical structure, and machine learning, this project transforms written input into melodies and harmonies that reflect the emotional landscape of the text.

![Image](https://github.com/user-attachments/assets/05252cc1-3dae-42d9-a7b6-2c9e139c5226)

---

## 📜 Overview

Given a string of text, Sentence2Song:

1. **Analyzes sentiment** to extract emotional tone.
2. **Determines musical parameters** (key signature, tempo, dissonance level) based on sentiment.
3. **Generates a melody** using selected data structures (e.g., linked lists, sorting algorithms, hash tables).
4. **Generates harmony** using neural networks trained on real musical data.
5. Optionally, segments long inputs into “movements,” producing an entire LP-like structure.

---

## 🧠 Core Ideas

- **Emotion → Music Mapping**: The sentiment score determines key musical elements:
  - High positivity → Major key, fast tempo, low dissonance
  - High negativity → Minor key, slower tempo, higher dissonance

- **Melody from Algorithms**: Users can select how melodies are generated from a set of algorithmic processes taught in class.

- **Harmony via Neural Network**: A neural network generates chord progressions conditioned on the melody. Two variants (trained on positive vs. negative music) are mixed proportionally based on sentiment.

- **Multi-Voice Architecture**: If multiple sentiments are extracted (e.g., joy, anger, fear), each one can be assigned to a separate harmonic line.

---

## 🛠 Implementation Details

### 📥 Input
- Free-form user text
- NLP model from HuggingFace to extract sentiment score(s)

### 🎼 Melody Generation
- We insert the words of the input sentence into 3 data structures (RB-Tree, Circular buffer, and Hash Table)
- Melody is encoded as: [(scale degree (1–7), accidental (-1, 0, 1), duration (0.0–4.0)), ...]
- Three algorithms used for melody generation, each tied to a data structure concept.

### 🎹 Harmony Generation
- Harmony is generated using a neural network:
- Input: melody sequence
- Output: harmony sequence
- Output format:
  ```
  [(scale degree, duration, accidental), ...]
  ```
- Harmony is post-processed into two chords per measure

- Measures are sampled from either the **positive** or **negative** neural net depending on sentiment distribution (e.g., 70% positive → 70% positive harmonies).

---

## 🧪 Model Training

- LSTM-based model using Keras
- Trained on existing music datasets with labeled sentiment/dissonance
- Data converted to tokenized format compatible with the melody/harmony structure

---

## 🎨 Visualization & Output

- Sheet music displayed using Music21
- Harmonies shown in bass clef
- Circular buffer or data structure visualizations (in slides)
- LP-style output for longer texts, simulating multiple songs/movements

---

## 📚 Resources Used

- **Music21**: For musical representation and visualization
- **Keras / TensorFlow**: For LSTM neural network training
- **HuggingFace Transformers**: For sentiment analysis
- **Todd / Hörne papers**: Algorithmic composition inspiration and harmony network structure
- **ChordNet**: Reference for chord-based neural network architecture

---

## 🗣 Future Directions

- Expand from sentiment to full emotional spectrum (e.g., joy, sadness, anger)
- More advanced LP structure: key modulations, transitions between movements
- Real-time music generation from streaming text input
- Interactive web app deployment

