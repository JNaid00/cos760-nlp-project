# 📘 Subword Tokenization in Neural Network Sentiment Analysis

Welcome to Group 20’s research project on **Subword Tokenization in Neural Network Sentiment Analysis**. This project investigates how subword-level tokenization techniques can enhance the performance of sentiment classification models—particularly in the context of social media text for low resource African Languages.

---

## 👥 Group Members

- **Luke Bradford**
- **Jesse Naidoo**
- **Tawanda Jimu**

---

## 🧪 Project Overview

Sentiment analysis is a fundamental task in natural language processing, often challenged by noisy, short-form text such as tweets. This project explores the use of **subword tokenization** in combination with a **TF-IDF-based neural network** to improve classification performance across three sentiment categories: **positive**, **negative**, and **neutral**.

---


## 📚 Dataset

This project uses the **[NaijaSenti](https://github.com/hausanlp/NaijaSenti)** dataset — a large-scale, richly annotated sentiment analysis dataset for four Nigerian languages:

- Hau - **Hausa**
- Yor - **Yoruba**
- Ibo -  **Igbo**
- Pcm -  **Nigerian Pidgin**

For more details and access to the dataset, visit the official repository:  
🔗 [https://github.com/hausanlp/NaijaSenti](https://github.com/hausanlp/NaijaSenti)

--- 

## 📂 Important Files in This Repository

| File | Description |
|------|-------------|
| `final.ipynb` | The complete experimental notebook containing training, evaluation, and commentary. |
| `final_script.py` | A script version of the final model for running from the command line. |
| `hyperparameter_script.ipynb` | A Jupyter Notebook designed for experimenting with hyperparameter tuning |
| `requirements.txt` | List of Python dependencies required to run the project. |
| `[All other].py files` | Provide functionality or data the final notebook or script can import from and use. |
| `/experimentation_scripts/*` | Scripts used in the experimentation phase of the project and archived in this folder. |
---

## 🛠️ Getting Started

To run the project on your local machine:

### 1. (Optional) Create a Virtual Environment

Using `venv`:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the code

#### Option 1: Jupyter Notebook

```bash
jupyter notebook final.ipynb
```
#### Option 2: Python Script

```bash
python final_script.py
```

---

## 📊 Results

All results are saved in the `output/` directory, organized by model:

```
output/
├── logistic_regression/
├── naive_bayes/
└── neural_network/
```

Each subfolder contains `.csv` files representing performance on different languages and preprocessing types:

- **lg_ / nb_ / nn_** → Model prefix (Logistic Regression, Naive Bayes, Neural Network)
- **hau / ibo / yor / pcm** → Language (Hausa, Igbo, Yoruba, Pidgin)
- **bow / tfidf / sp / wp** → Feature type: Bag of Words, TF-IDF, SentencePiece, WordPiece
- **_no_clean** → Indicates uncleaned input data

> **Note:** All Neural Network (NN) models were run using only TF-IDF as a vectorizer.