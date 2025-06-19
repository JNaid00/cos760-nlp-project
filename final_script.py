# %% [markdown]
# # Read in Data

# %%
import pandas as pd
import numpy as np
from typing import Optional

# %%
from custom_datasets import MultiLangDataset, SplitSet
from custom_datasets import ns_dataset
from custom_datasets import Languages
from custom_datasets import clean_tweet
from constants import TokenizerEnum, VectorizerEnum
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from custom_vectorizers import get_vectorizer

# %%
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

# %%
YOR_DATASET: SplitSet = ns_dataset.get(Languages.YORUBA)
HAU_DATASET: SplitSet = ns_dataset.get(Languages.HAUSA)
IBO_DATASET: SplitSet = ns_dataset.get(Languages.IGBO)
PCM_DATASET: SplitSet = ns_dataset.get(Languages.NIGERIAN_PIDGIN)

# %%
# Evaluator
from analysis import compare_results

# Tokenizer
from subword_tokenizer import (
    get_tokenizer,
    wordpiece_tokenize_dataframe,
    get_sentencepiece_tokenizer,
    sentencepiece_tokenize_dataframe,
    get_wordpiece_tokeized_data,
    get_sentencepiece_tokeized_data,
)

# Compare Results
from sklearn.metrics import accuracy_score, classification_report


# %%
def encode_labels(df: pd.DataFrame):
    label_mapping = {"positive": 0, "neutral": 1, "negative": 2}
    df["label_encoded"] = df["label"].str.lower().map(label_mapping)


# %%
# VECTORIZER_KWARGS = {
#     "ngram": (1, 2),
#     "max_features": 3700,}

VECTORIZER_KWARGS = {}

# %% [markdown]
# # Naive Bayes

# %%
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def naive_bayes(
    dataset: SplitSet,
    vectorizer: VectorizerEnum,
    tokenizer: Optional[TokenizerEnum] = None,
) -> dict:
    """
    Naive Bayes classifier for text classification.
    Args:
        dataset (SplitSet): The dataset to use for training and testing.
        tokenizer (TokenizerEnum): The tokenizer to use.
        If tokenizer is None, then no tokenization is applied. Only vectorization is applied.
        vectorizer (VectorizerEnum): The vectorizer to use.
    Returns:
        dict: A dictionary containing the results of the classification.
        Returns a classification report
    """

    X_train, y_train = dataset.train["tweet"], dataset.train["label"]
    X_test, y_test = dataset.test["tweet"], dataset.test["label"]

    if tokenizer is TokenizerEnum.SENTENCEPIECE:
        print("Using SentencePiece tokenizer")
        sp_tokenizer = get_sentencepiece_tokenizer(
            df=dataset.train,
        )

        def sentencepiece_tokenizer(text):
            return sp_tokenizer.encode(text, out_type=str)

        vectorizer = (
            CountVectorizer(tokenizer=sentencepiece_tokenizer)
            if vectorizer == VectorizerEnum.BOW
            else TfidfVectorizer(tokenizer=sentencepiece_tokenizer)
        )
    elif tokenizer is TokenizerEnum.WORDPIECE:
        print("Using WordPiece tokenizer")
        wp_tokenizer = get_tokenizer(
            df=dataset.train,
        )

        def wordpiece_tokenizer(text):
            return wp_tokenizer.encode(text).tokens

        vectorizer = (
            CountVectorizer(tokenizer=wordpiece_tokenizer)
            if vectorizer == VectorizerEnum.BOW
            else TfidfVectorizer(tokenizer=wordpiece_tokenizer)
        )
    elif tokenizer is None:
        vectorizer = (
            CountVectorizer(stop_words=dataset.stopwords)
            if vectorizer == VectorizerEnum.BOW
            else TfidfVectorizer(stop_words=dataset.stopwords)
        )

    model = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("classifier", MultinomialNB()),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    word_result: dict = classification_report(y_test, y_pred, output_dict=True)
    return word_result


# %% [markdown]
# # Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression


def logistic_regression(
    dataset: SplitSet,
    vectorizer: VectorizerEnum,
    tokenizer: Optional[TokenizerEnum] = None,
) -> dict:
    """
    Logistic Regression classifier for text classification.
    Args:
        dataset (SplitSet): The dataset to use for training and testing.
        tokenizer (TokenizerEnum): The tokenizer to use.
        If tokenizer is None, then no tokenization is applied. Only vectorization is applied.
        vectorizer (VectorizerEnum): The vectorizer to use.
    Returns:
        dict: A dictionary containing the results of the classification.
        Returns a classification report
    """
    X_train, y_train = dataset.train["tweet"], dataset.train["label"]
    X_test, y_test = dataset.test["tweet"], dataset.test["label"]

    if tokenizer is TokenizerEnum.SENTENCEPIECE:
        print("Using SentencePiece tokenizer")
        sp_tokenizer = get_sentencepiece_tokenizer(
            df=dataset.train,
        )

        def sentencepiece_tokenizer(text):
            return sp_tokenizer.encode(text, out_type=str)

        vectorizer = (
            CountVectorizer(tokenizer=sentencepiece_tokenizer)
            if vectorizer == VectorizerEnum.BOW
            else TfidfVectorizer(tokenizer=sentencepiece_tokenizer)
        )
    elif tokenizer is TokenizerEnum.WORDPIECE:
        print("Using WordPiece tokenizer")
        wp_tokenizer = get_tokenizer(
            df=dataset.train,
        )

        def wordpiece_tokenizer(text):
            return wp_tokenizer.encode(text).tokens

        vectorizer = (
            CountVectorizer(tokenizer=wordpiece_tokenizer)
            if vectorizer == VectorizerEnum.BOW
            else TfidfVectorizer(tokenizer=wordpiece_tokenizer)
        )
    elif tokenizer is None:
        vectorizer = (
            CountVectorizer(stop_words=dataset.stopwords)
            if vectorizer == VectorizerEnum.BOW
            else TfidfVectorizer(stop_words=dataset.stopwords)
        )

    model = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    word_result: dict = classification_report(y_test, y_pred, output_dict=True)
    return word_result


# %% [markdown]
# # Neural Network

# %%


from copy import deepcopy


def neural_network(
    dataset: SplitSet,
    vectorizer: VectorizerEnum,
    tokenizer: Optional[TokenizerEnum] = None,
    clean_tweets: bool = True,
) -> dict:
    """
    Neural Network classifier for text classification.
    Args:
        dataset (SplitSet): The dataset to use for training and testing.
        tokenizer (TokenizerEnum): The tokenizer to use.
        If tokenizer is None, then no tokenization is applied. Only vectorization is applied.
        vectorizer (VectorizerEnum): The vectorizer to use.
    Returns:
        dict: A dictionary containing the results of the classification.
        Returns a classification report
    """
    pass

    data = deepcopy(dataset)
    train_df = data.train
    test_df = data.test
    # encode_labels(df)
    if clean_tweets:
        data.train["cleaned_tweet"] = data.train["tweet"].apply(clean_tweet)
        data.test["cleaned_tweet"] = data.test["tweet"].apply(clean_tweet)
    encode_labels(data.train)
    encode_labels(data.test)

    if tokenizer is not None:
        neural_input = get_wordpiece_tokeized_data(
            data,
            vocab_size=3700,
            tweet_column="cleaned_tweet",
            vectorizer_kwargs={"ngram": (1, 2), "max_features": None},
        )
    elif tokenizer == TokenizerEnum.SENTENCEPIECE:
        neural_input = get_sentencepiece_tokeized_data(
            data,
            vocab_size=3700,
            tweet_column="cleaned_tweet",
            vectorizer_kwargs={"ngram": (1, 2), "max_features": None},
        )

    model = Sequential()
    model.add(Input(shape=(neural_input.X_train.shape[1],)))

    # Dense layers for TF-IDF input
    # (512, 256, 128)
    # (8, 4, 2)
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Dense(3, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy"],
    )

    X = np.array(neural_input.get_dense_X_train())
    y = np.array(neural_input.y_train)

    model.fit(X, y, epochs=10, batch_size=64, verbose=1)

    # Evaluate the model
    # model.evaluate(neural_input.X_test, neural_input.y_test)
    y_pred = model.predict(
        np.array(neural_input.get_dense_X_test()),
    )
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(neural_input.y_test, y_pred_classes)
    print(f"Accuracy with filtered tweets {accuracy:.4f}")

    return classification_report(
        neural_input.y_test,
        y_pred_classes,
        target_names=["positive", "neutral", "negative"],
        output_dict=True,
    )


# %%
nn_yor_kwargs = {
    "nn_yor_wp_clean": {
        "dataset": YOR_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.WORDPIECE,
        "clean_tweets": True,
    },
    "nn_yor_wp_no_clean": {
        "dataset": YOR_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.WORDPIECE,
        "clean_tweets": False,
    },
    "nn_yor_sp_clean": {
        "dataset": YOR_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.SENTENCEPIECE,
        "clean_tweets": True,
    },
    "nn_yor_sp_no_clean": {
        "dataset": YOR_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.SENTENCEPIECE,
        "clean_tweets": False,
    },
}

nn_hau_kwargs = {
    "nn_hau_wp_clean": {
        "dataset": HAU_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.WORDPIECE,
        "clean_tweets": True,
    },
    "nn_hau_wp_no_clean": {
        "dataset": HAU_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.WORDPIECE,
        "clean_tweets": False,
    },
    "nn_hau_sp_clean": {
        "dataset": HAU_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.SENTENCEPIECE,
        "clean_tweets": True,
    },
    "nn_hau_sp_no_clean": {
        "dataset": HAU_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.SENTENCEPIECE,
        "clean_tweets": False,
    },
}

nn_pcm_kwargs = {
    "nn_pcm_wp_clean": {
        "dataset": PCM_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.WORDPIECE,
        "clean_tweets": True,
    },
    "nn_pcm_wp_no_clean": {
        "dataset": PCM_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.WORDPIECE,
        "clean_tweets": False,
    },
    "nn_pcm_sp_clean": {
        "dataset": PCM_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.SENTENCEPIECE,
        "clean_tweets": True,
    },
    "nn_pcm_sp_no_clean": {
        "dataset": PCM_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.SENTENCEPIECE,
        "clean_tweets": False,
    },
}

nn_ibo_kwargs = {
    "nn_ibo_wp_clean": {
        "dataset": IBO_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.WORDPIECE,
        "clean_tweets": True,
    },
    "nn_ibo_wp_no_clean": {
        "dataset": IBO_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.WORDPIECE,
        "clean_tweets": False,
    },
    "nn_ibo_sp_clean": {
        "dataset": IBO_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.SENTENCEPIECE,
        "clean_tweets": True,
    },
    "nn_ibo_sp_no_clean": {
        "dataset": IBO_DATASET,
        "vectorizer": VectorizerEnum.TFIDF,
        "tokenizer": TokenizerEnum.SENTENCEPIECE,
        "clean_tweets": False,
    },
}

# %%
nn_yor_wp = neural_network(**nn_yor_kwargs["nn_yor_wp_clean"])
nn_yor_wp_no_clean = neural_network(**nn_yor_kwargs["nn_yor_wp_no_clean"])
nn_yor_sp = neural_network(**nn_yor_kwargs["nn_yor_sp_clean"])
nn_yor_sp_no_clean = neural_network(**nn_yor_kwargs["nn_yor_sp_no_clean"])


nn_hau_wp = neural_network(**nn_hau_kwargs["nn_hau_wp_clean"])
nn_hau_wp_no_clean = neural_network(**nn_hau_kwargs["nn_hau_wp_no_clean"])
nn_hau_sp = neural_network(**nn_hau_kwargs["nn_hau_sp_clean"])
nn_hau_sp_no_clean = neural_network(**nn_hau_kwargs["nn_hau_sp_no_clean"])

nn_pcm_wp = neural_network(**nn_pcm_kwargs["nn_pcm_wp_clean"])
nn_pcm_wp_no_clean = neural_network(**nn_pcm_kwargs["nn_pcm_wp_no_clean"])
nn_pcm_sp = neural_network(**nn_pcm_kwargs["nn_pcm_sp_clean"])
nn_pcm_sp_no_clean = neural_network(**nn_pcm_kwargs["nn_pcm_sp_no_clean"])

nn_ibo_wp = neural_network(**nn_ibo_kwargs["nn_ibo_wp_clean"])
nn_ibo_wp_no_clean = neural_network(**nn_ibo_kwargs["nn_ibo_wp_no_clean"])
nn_ibo_sp = neural_network(**nn_ibo_kwargs["nn_ibo_sp_clean"])
nn_ibo_sp_no_clean = neural_network(**nn_ibo_kwargs["nn_ibo_sp_no_clean"])

# %%
compare_results(
    normal_result=nn_yor_wp,
    subword_result=nn_yor_sp,
)

compare_results(
    normal_result=nn_yor_sp,
    subword_result=nn_yor_wp,
)

compare_results(
    normal_result=nn_yor_wp_no_clean,
    subword_result=nn_yor_sp_no_clean,
)


# %%
lg_yor_tfidf = logistic_regression(dataset=YOR_DATASET, vectorizer=VectorizerEnum.TFIDF)
lg_yor_bow = logistic_regression(dataset=YOR_DATASET, vectorizer=VectorizerEnum.BOW)

lg_pcm_tfidf = logistic_regression(dataset=PCM_DATASET, vectorizer=VectorizerEnum.TFIDF)
lg_pcm_bow = logistic_regression(dataset=PCM_DATASET, vectorizer=VectorizerEnum.BOW)

lg_ibo_tfidf = logistic_regression(dataset=IBO_DATASET, vectorizer=VectorizerEnum.TFIDF)
lg_ibo_bow = logistic_regression(dataset=IBO_DATASET, vectorizer=VectorizerEnum.BOW)

lg_hau_tfidf = logistic_regression(dataset=HAU_DATASET, vectorizer=VectorizerEnum.TFIDF)
lg_hau_bow = logistic_regression(dataset=HAU_DATASET, vectorizer=VectorizerEnum.BOW)

# %%
# Compare Logistic Regression Results


# %%
nb_yor_tfidf = naive_bayes(dataset=YOR_DATASET, vectorizer=VectorizerEnum.TFIDF)
nb_yor_bow = naive_bayes(dataset=YOR_DATASET, vectorizer=VectorizerEnum.BOW)

nb_pcm_tfidf = naive_bayes(dataset=PCM_DATASET, vectorizer=VectorizerEnum.TFIDF)
nb_pcm_bow = naive_bayes(dataset=PCM_DATASET, vectorizer=VectorizerEnum.BOW)

nb_ibo_tfidf = naive_bayes(dataset=IBO_DATASET, vectorizer=VectorizerEnum.TFIDF)
nb_ibo_bow = naive_bayes(dataset=IBO_DATASET, vectorizer=VectorizerEnum.BOW)

nb_hau_tfidf = naive_bayes(dataset=HAU_DATASET, vectorizer=VectorizerEnum.TFIDF)
nb_hau_bow = naive_bayes(dataset=HAU_DATASET, vectorizer=VectorizerEnum.BOW)

# %%
# Compare Naive Bayes Results
