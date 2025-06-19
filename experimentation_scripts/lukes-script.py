# %%
import os
import pandas as pd
import subprocess
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase, NFD, StripAccents
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Input,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.utils import to_categorical

from models import BasicModelEncapsulator, NeuralNetworkModel
from custom_vectorizers import initialise_count_vectorizer, initialise_tfidf_vectorizer
from constants import LOCAL_DIR_AS, LOCAL_DIR_NS, REPO_URL_AS, REPO_URL_NS, NS_LANGUAGES
from custom_datasets import MultiLangDataset, load_local_datasets

from custom_datasets import Languages


# %%
def clone_repo(repo_url: str, local_dir: str) -> None:
    if os.path.isdir(local_dir):
        print("Repository exists. Updating...")
        subprocess.run(["git", "-C", local_dir, "pull", "origin", "main"], check=True)
    else:
        print("Repository not found. Cloning...")
        subprocess.run(["git", "clone", repo_url], check=True)


clone_repo(REPO_URL_NS, LOCAL_DIR_NS)
clone_repo(REPO_URL_AS, LOCAL_DIR_AS)


# %%
ns_dataset: MultiLangDataset = load_local_datasets(
    local_base_dir=LOCAL_DIR_NS + "/data/annotated_tweets", languages=NS_LANGUAGES
)

# %%
as_dataset: MultiLangDataset = load_local_datasets(
    local_base_dir=f"afrisent-semeval-2023/data",
    languages=NS_LANGUAGES,
)

# %%
print("NaijaSenti dataset loaded with languages:", ns_dataset.all_languages())
print("Afrisenti dataset loaded with languages:", as_dataset.all_languages())

# %%


print("NaijaSenti hau: ", ns_dataset.get(Languages.HAUSA).test)
# Print each row in the dev set for the column 'tweet'
for index, row in ns_dataset.get(Languages.HAUSA).test.iterrows():
    print(f"Index: {index}, Tweet: {row['tweet']}")

# write all the tweets into a textfile
# check if the dir data exists, if not create it
if not os.path.exists("data"):
    os.makedirs("data")
with open("data/naija_senti_hau_dev_tweets.txt", "w", encoding="utf-8") as f:
    for index, row in ns_dataset.get(Languages.HAUSA).dev.iterrows():
        f.write(f"{row['tweet']}\n")

# %%
df = ns_dataset.get(Languages.HAUSA).train
text_train, text_test, y_train, y_test = train_test_split(
    df.tweet, df.label, test_size=0.3
)
X_train_tfidf, vectorizer_tfidf = initialise_tfidf_vectorizer(text_train)
X_train_count, vectorizer_count = initialise_count_vectorizer(text_train)

# %%


# %%
# Get the number of features from your TF-IDF matrix
# tfidf_features = X_train_tfidf.shape[1]  # Number of TF-IDF features
# num_classes = len(np.unique(y_train))    # Number of sentiment classes

# Initialize models


logistic_regression_model = BasicModelEncapsulator(LogisticRegression(max_iter=1000))
naive_bayes_model = BasicModelEncapsulator(MultinomialNB())

tfidf_features = X_train_tfidf.shape[1]  # Number of TF-IDF features
num_classes = len(np.unique(y_train))  # Number of classes


# Initialize the corrected neural network
neural_network_model = NeuralNetworkModel(
    input_dim=tfidf_features, num_classes=num_classes
)

# Perform pipelines
print("Training models...")

# %%
# Logistic Regression with TF-IDF
accuracy_lr, report_lr = logistic_regression_model.perform_pipeline(
    X_train_tfidf, y_train
)
print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression Classification Report:\n", report_lr)

# Logistic Regression with Count Vectorizer
X_train_count, vectorizer_count = initialise_count_vectorizer(text_train)
accuracy_lr_count, report_lr_count = logistic_regression_model.perform_pipeline(
    X_train_count, y_train
)
print("Logistic Regression with Count Vectorizer Accuracy:", accuracy_lr_count)
print(
    "Logistic Regression with Count Vectorizer Classification Report:\n",
    report_lr_count,
)

# %%
# Naive Bayes with TF-IDF
accuracy_nb, report_nb = naive_bayes_model.perform_pipeline(X_train_tfidf, y_train)
print("Naive Bayes Accuracy:", accuracy_nb)
print("Naive Bayes Classification Report:\n", report_nb)

# Naive Bayes with Count Vectorizer
X_train_count, vectorizer_count = initialise_count_vectorizer(text_train)
accuracy_nb_count, report_nb_count = naive_bayes_model.perform_pipeline(
    X_train_count, y_train
)
print("Naive Bayes with Count Vectorizer Accuracy:", accuracy_nb_count)
print("Naive Bayes with Count Vectorizer Classification Report:\n", report_nb_count)

# %%
# Neural Network with TF-IDF
ohe_labels = [
    0 if label == "neutral" else 1 if label == "positive" else 2 for label in y_train
]

accuracy_nn, report_nn = neural_network_model.perform_pipeline(
    X_train_tfidf, ohe_labels
)
print("Neural Network Accuracy:", accuracy_nn)
print("Neural Network Classification Report:\n", report_nn)

# Neural Network with Count Vectorizer
accuracy_nn_count, report_nn_count = neural_network_model.perform_pipeline(
    X_train_count, ohe_labels
)
print("Neural Network with Count Vectorizer Accuracy:", accuracy_nn_count)
print("Neural Network with Count Vectorizer Classification Report:\n", report_nn_count)


# %%
# Wordpiece tokenized models TFIDF

from subword_tokenizer import get_tokenizer, wordpiece_tokenize_dataframe

tokenizer = get_tokenizer(df=ns_dataset.get(Languages.HAUSA).train)

train_df = ns_dataset.get(Languages.HAUSA).train
test_df = ns_dataset.get(Languages.HAUSA).test
# Naive Bayes with wordpiece tokenized data
wp_train_df = wordpiece_tokenize_dataframe(train_df, tokenizer)
wp_test_df = wordpiece_tokenize_dataframe(test_df, tokenizer)

wp_X_train_list = wp_train_df["tokenized_tweets"].tolist()
wp_X_test_list = wp_test_df["tokenized_tweets"].tolist()

# join sub lists into strings
wp_X_train_list = [" ".join(tokens) for tokens in wp_X_train_list]
wp_X_test_list = [" ".join(tokens) for tokens in wp_X_test_list]
# Convert labels to numerical format (0 for neutral, 1 for positive, 2 for negative)
wp_train_df["label"] = wp_train_df["label"].apply(
    lambda x: 0 if x == "neutral" else 1 if x == "positive" else 2
)
wp_test_df["label"] = wp_test_df["label"].apply(
    lambda x: 0 if x == "neutral" else 1 if x == "positive" else 2
)

wp_y_train = wp_train_df["label"].tolist()
wp_y_test = wp_test_df["label"].tolist()

tfidf_wp_train, vectorizer_wp = initialise_tfidf_vectorizer(wp_X_train_list)
tfidf_wp_test, _ = initialise_tfidf_vectorizer(wp_X_test_list)

tfidf_features = tfidf_wp_train.shape[1]  # Number of TF-IDF features
num_classes = len(np.unique(wp_y_train))  # Number of classes


# Initialize the corrected neural network
neural_network_model = NeuralNetworkModel(
    input_dim=tfidf_features, num_classes=num_classes
)


# Naive Bayes with WordPiece tokenized data
accuracy_nb, report_nb = naive_bayes_model.perform_pipeline(tfidf_wp_train, wp_y_train)
print("Naive Bayes Accuracy:", accuracy_nb)
print("Naive Bayes Classification Report:\n", report_nb)

# Logistic Regression with WordPiece tokenized data
accuracy_lr_wp, report_lr_wp = logistic_regression_model.perform_pipeline(
    tfidf_wp_train, wp_y_train
)
print("Logistic Regression Accuracy:", accuracy_lr_wp)
print("Logistic Regression Classification Report:\n", report_lr_wp)

# Neural Network with WordPiece tokenized data
accuracy_nn_wp, report_nn_wp = neural_network_model.perform_pipeline(
    tfidf_wp_train, wp_y_train
)
print("Neural Network Accuracy:", accuracy_nn_wp)
print("Neural Network Classification Report:\n", report_nn_wp)


# %%
# Method to optimize n-grams and max features for TF-IDF
def tfidf_score(input_x, y_train, score=None):
    clf = LogisticRegression(max_iter=1000)
    return cross_val_score(clf, X=input_x, y=y_train, scoring=score)


scores_tfidf = tfidf_score(X_train_tfidf, y_train)
print(
    "5-fold Cross-Validation Accuracy for TFIDF: %0.2f (+/- %0.2f)"
    % (scores_tfidf.mean(), scores_tfidf.std() * 2)
)

scores_tfidf_f1 = tfidf_score(X_train_tfidf, y_train, score="f1_macro")

print(
    "5-fold Cross-Validation F1 score for TFIDF: %0.2f (+/- %0.2f)"
    % (scores_tfidf_f1.mean(), scores_tfidf_f1.std() * 2)
)


def test_param_combos(X_train, y_train, param_combos):
    results = []
    for params in param_combos:
        X_train_tfidf, vectorizer_tfidf = initialise_tfidf_vectorizer(
            X_train,
            ngram=params.get("ngram_range"),
            max_features=params.get("max_features"),
        )
        score = tfidf_score(X_train_tfidf, y_train)
        results.append(
            {
                "ngram_range": params.get("ngram_range"),
                "max_features": params.get("max_features"),
                "score": score.mean(),
                "std_dev": score.std(),
            }
        )

    return pd.DataFrame(results)


# Example parameter combinations to test
param_combos = [
    {"ngram_range": (1, 2), "max_features": 5000},
    {"ngram_range": (1, 3), "max_features": 5000},
    {"ngram_range": (1, 2), "max_features": 10000},
    {"ngram_range": (1, 3), "max_features": 10000},
    {"ngram_range": (1, 2), "max_features": None},
    {"ngram_range": (1, 3), "max_features": None},
    {"ngram_range": (1, 2), "max_features": 2000},
    {"ngram_range": (1, 3), "max_features": 2000},
    {"ngram_range": (1, 2), "max_features": 3000},
    {"ngram_range": (1, 3), "max_features": 3000},
    {"ngram_range": (1, 2), "max_features": 4000},
    {"ngram_range": (1, 3), "max_features": 4000},
    {"ngram_range": (1, 2), "max_features": 6000},
    {"ngram_range": (1, 3), "max_features": 6000},
    {"ngram_range": (1, 2), "max_features": 7000},
    {"ngram_range": (1, 3), "max_features": 7000},
    {"ngram_range": (1, 2), "max_features": 8000},
    {"ngram_range": (1, 3), "max_features": 8000},
    {"ngram_range": (1, 2), "max_features": 9000},
    {"ngram_range": (1, 3), "max_features": 9000},
    {"ngram_range": (1, 2), "max_features": 10000},
    {"ngram_range": (1, 3), "max_features": 10000},
    {"ngram_range": (1, 2), "max_features": 12000},
    {"ngram_range": (1, 4), "max_features": 5000},
    {"ngram_range": (1, 4), "max_features": 10000},
    {"ngram_range": (1, 4), "max_features": None},
    {"ngram_range": (1, 4), "max_features": 2000},
    {"ngram_range": (1, 4), "max_features": 3000},
    {"ngram_range": (1, 4), "max_features": 4000},
    {"ngram_range": (1, 4), "max_features": 6000},
    {"ngram_range": (1, 4), "max_features": 7000},
    {"ngram_range": (1, 4), "max_features": 8000},
    {"ngram_range": (1, 4), "max_features": 9000},
    {"ngram_range": (1, 4), "max_features": 10000},
    {"ngram_range": (1, 4), "max_features": 12000},
    {"ngram_range": (2, 5), "max_features": 5000},
    {"ngram_range": (2, 5), "max_features": 10000},
    {"ngram_range": (2, 5), "max_features": None},
    {"ngram_range": (2, 5), "max_features": 2000},
    {"ngram_range": (2, 5), "max_features": 3000},
    {"ngram_range": (2, 5), "max_features": 4000},
    {"ngram_range": (2, 5), "max_features": 6000},
    {"ngram_range": (2, 5), "max_features": 7000},
    {"ngram_range": (2, 5), "max_features": 8000},
    {"ngram_range": (2, 5), "max_features": 9000},
    {"ngram_range": (2, 5), "max_features": 10000},
    {"ngram_range": (2, 5), "max_features": 12000},
    {"ngram_range": (3, 5), "max_features": 5000},
    {"ngram_range": (3, 5), "max_features": 10000},
    {"ngram_range": (3, 5), "max_features": None},
    {"ngram_range": (3, 5), "max_features": 2000},
    {"ngram_range": (3, 5), "max_features": 3000},
    {"ngram_range": (3, 5), "max_features": 4000},
    {"ngram_range": (3, 5), "max_features": 6000},
    {"ngram_range": (3, 5), "max_features": 7000},
    {"ngram_range": (3, 5), "max_features": 8000},
    {"ngram_range": (3, 5), "max_features": 9000},
    {"ngram_range": (3, 5), "max_features": 10000},
    {"ngram_range": (3, 5), "max_features": 12000},
]
# Test the parameter combinations
results_df = test_param_combos(text_train, y_train, param_combos)
# Sort the results by mean score
results_df = results_df.sort_values(by="score", ascending=False)
# Save the results to a CSV file
results_df.to_csv("data/tfidf_param_combos_results.csv", index=False)
# Print the top results
print("Top parameter combinations based on accuracy:")
print(results_df.head(10))
# Print the results DataFrame


# %%
# Plot playground (based off neural network training history)
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()
