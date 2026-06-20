# 🎬 Movie Dialogue Chatbot (Seq2Seq with Attention)

Welcome to the **Movie Dialogue Chatbot** project! This repository contains a deep learning-based conversational AI trained on a dataset of movie dialogues. The model utilizes a **Sequence-to-Sequence (Seq2Seq)** architecture enhanced with an **Attention Mechanism** and **Bidirectional LSTMs** to generate contextually relevant responses.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Model Architecture](#-model-architecture)
3. [Dataset & Preprocessing](#-dataset--preprocessing)
4. [Tech Stack](#-tech-stack)
5. [Installation & Setup](#-installation--setup)
6. [Usage & Inference](#-usage--inference)
7. [Future Improvements](#-future-improvements)

---

## 🚀 Project Overview
The goal of this project is to build an end-to-end chatbot capable of understanding user queries (questions) and generating human-like responses (answers) based on patterns learned from movie scripts. By leveraging pre-trained word embeddings and attention mechanisms, the model can focus on relevant parts of the input sequence when generating each word of the output sequence.

---

## 🧠 Model Architecture
The core of the chatbot is built using TensorFlow/Keras and follows an Encoder-Decoder paradigm:

*   **Embedding Layer**: Uses pre-trained **GloVe (50D)** word vectors to capture semantic relationships between words.
*   **Encoder**: 
    *   3-layer **Bidirectional LSTM** network.
    *   Captures both forward and backward context of the input question.
    *   Outputs the final hidden states (`h`) and cell states (`c`) to initialize the decoder, along with the sequence of encoder outputs for the Attention mechanism.
*   **Attention Mechanism**: 
    *   Implements **Dot-Product Attention**.
    *   Allows the decoder to "look back" at the encoder's outputs and weigh the importance of each input word when predicting the next output word.
*   **Decoder**: 
    *   1-layer **LSTM** network.
    *   Takes the previous word (starting with `<sos>`) and the attention context to predict the next word in the sequence.
    *   Ends generation when the `<eos>` (End of Sequence) token is predicted.

---

## 📊 Dataset & Preprocessing
### Dataset
The model is trained on the **Cleaned Data for the Chatbot Collected from Movies** dataset, which contains pairs of questions and answers extracted from movie scripts.
*   **Total Samples**: ~127,000 unique dialogue pairs after cleaning.
*   **Split**: 90% Training, 10% Testing.

### Preprocessing Pipeline
1.  **Text Cleaning**: 
    *   Lowercasing all text.
    *   Expanding contractions (e.g., `don't` $\rightarrow$ `do not`).
    *   Removing special characters, punctuation, and URLs using Regular Expressions.
2.  **Sequence Markers**: Adding `<sos>` (Start of Sequence) to decoder inputs and `eos>` (End of Sequence) to decoder labels.
3.  **Tokenization & Padding**: Converting text to integer sequences and padding them to a uniform length (`maxlen=25` for questions, `maxlen=28` for answers).

---

## 🛠 Tech Stack
*   **Language**: Python 3
*   **Deep Learning Framework**: TensorFlow / Keras
*   **NLP Libraries**: NLTK (Stopwords), Custom Tokenizer
*   **Data Manipulation**: Pandas, NumPy
*   **Visualization**: Matplotlib, Seaborn
*   **Word Embeddings**: GloVe (Global Vectors for Word Representation)

---

## 💻 Installation & Setup

### Prerequisites
Make sure you have Python 3.8+ installed. It is highly recommended to use a virtual environment or a platform like **Kaggle** / **Google Colab** as this project relies on specific input directories for datasets and GloVe embeddings.

### Required Libraries
Install the necessary Python packages:
```bash
pip install tensorflow numpy pandas nltk matplotlib seaborn
