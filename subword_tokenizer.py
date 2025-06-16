import os
import re
from typing import Dict, Optional
import numpy as np
from pandas import DataFrame
from typing import Optional, Dict
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer as WPTokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase, NFD, StripAccents
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
import os

from custom_datasets import SplitSet
from custom_vectorizers import initialise_tfidf_vectorizer


def get_tokenizer(
    df: DataFrame,
    save_tokenizer: bool = False,
    vocab_size: int = 8000,
    tweet_column: str = "tweet",
) -> WPTokenizer:
    """
    Returns the WordPiece tokenizer.
    """
    tokenizer = WPTokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        # vocab_size=12000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        vocab_size=vocab_size,
        special_tokens=["[UNK]"],
    )

    # Collect all tweets into a single list for training
    tweets = df[tweet_column].tolist()
    tokenizer.train_from_iterator(tweets, trainer)
    if save_tokenizer:
        # Save the tokenizer to a file
        print("Saving tokenizer to data/wordpiece.json")
        if not os.path.exists("data"):
            os.makedirs("data")
        tokenizer.save("data/wordpiece.json")
    return tokenizer


def preprocess_tweet(tweet):
    # Remove all words that start with @ (e.g., @user, @someone123)
    return re.sub(r"@\w+", "", tweet).strip()


def wordpiece_tokenize_dataframe(
    df: DataFrame,
    tokenizer: WPTokenizer,
    preprocess_tweets: bool = False,
    tweet_column: str = "tweet",
) -> DataFrame:
    """
    Tokenizes the 'tweet' column of a DataFrame using a WordPiece tokenizer.
    Adds two new columns: 'tokenized_tweets' and 'token_ids'.
    """
    if preprocess_tweets:
        df[tweet_column] = df[tweet_column].apply(preprocess_tweet)
    df["tokenized_tweets"] = df[tweet_column].apply(
        lambda x: tokenizer.encode(x).tokens
    )
    df["token_ids"] = df[tweet_column].apply(lambda x: tokenizer.encode(x).ids)
    return df


class NeuralNetworkInput:
    def __init__(self, X_train, y_train, X_test, y_test, num_classes, num_features):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.num_features = num_features

    def get_dense_X_train(self):
        X_train = self.X_train.toarray()
        return X_train.astype(np.float32)

    def get_dense_X_test(self):
        X_test = self.X_test.toarray()
        return X_test.astype(np.float32)

    def get_input_shape(self):
        return (self.X_train.shape[1],)

    def get_num_classes(self):
        return self.num_classes

    def get_num_features(self):
        return self.num_features


def get_wordpiece_tokeized_data(
    data: SplitSet,
    vocab_size=None,
    vectorizer_kwargs: Optional[Dict] = None,
    tweet_column: str = "tweet",
) -> NeuralNetworkInput:
    tokenizer = get_tokenizer(
        df=data.train, vocab_size=vocab_size, tweet_column=tweet_column
    )

    # train_df = encode_labels(train_df)

    # Naive Bayes with wordpiece tokenized data
    wp_train_df = wordpiece_tokenize_dataframe(
        data.train,
        tokenizer,
    )
    wp_test_df = wordpiece_tokenize_dataframe(
        data.test,
        tokenizer,
    )

    wp_X_train_list = wp_train_df["tokenized_tweets"].tolist()
    wp_X_test_list = wp_test_df["tokenized_tweets"].tolist()

    # join sub lists into strings
    wp_X_train_list = [" ".join(tokens) for tokens in wp_X_train_list]
    wp_X_test_list = [" ".join(tokens) for tokens in wp_X_test_list]

    wp_y_train = wp_train_df["label_encoded"].tolist()
    wp_y_test = wp_test_df["label_encoded"].tolist()

    if vectorizer_kwargs is None:
        vectorizer_kwargs = {}
    tfidf_wp_train, vectorizer_wp = initialise_tfidf_vectorizer(
        wp_X_train_list, **vectorizer_kwargs
    )
    tfidf_wp_test = vectorizer_wp.transform(wp_X_test_list)

    wp_tfidf_features = tfidf_wp_train.shape[1]  # Number of TF-IDF features
    wp_num_classes = len(np.unique(wp_y_train))

    # X_train, X_test, y_train, y_test = train_test_split(
    #     tfidf_wp_train, wp_y_train, test_size=0.3, random_state=42
    # )
    return NeuralNetworkInput(
        # X_train, y_train, X_test, y_test, wp_num_classes, wp_tfidf_features
        tfidf_wp_train,
        wp_y_train,
        tfidf_wp_test,
        wp_y_test,
        wp_num_classes,
        wp_tfidf_features,
    )


def get_sentencepiece_tokeized_data(
    data: SplitSet,
    vocab_size=None,
    vectorizer_kwargs: Optional[Dict] = None,
    tweet_column: str = "tweet",
) -> NeuralNetworkInput:
    tokenizer = get_sentencepiece_tokenizer(
        df=data.train, vocab_size=vocab_size, tweet_column=tweet_column
    )

    # train_df = df

    # train_df = encode_labels(train_df)

    # Naive Bayes with wordpiece tokenized data
    sp_train_df = wordpiece_tokenize_dataframe(
        data.train,
        tokenizer,
    )
    sp_test_df = wordpiece_tokenize_dataframe(
        data.test,
        tokenizer,
    )

    sp_X_train_list = sp_train_df["tokenized_tweets"].tolist()
    sp_X_test_list = sp_test_df["tokenized_tweets"].tolist()

    # join sub lists into strings
    sp_X_train_list = [" ".join(tokens) for tokens in sp_X_train_list]
    sp_X_test_list = [" ".join(tokens) for tokens in sp_X_test_list]

    sp_y_train = sp_train_df["label_encoded"].tolist()
    sp_y_test = sp_test_df["label_encoded"].tolist()

    if vectorizer_kwargs is None:
        vectorizer_kwargs = {}
    tfidf_sp_train, vectorizer_wp = initialise_tfidf_vectorizer(
        sp_X_train_list, **vectorizer_kwargs
    )
    tfidf_sp_test = vectorizer_wp.transform(sp_X_test_list)

    wp_tfidf_features = tfidf_sp_train.shape[1]  # Number of TF-IDF features
    wp_num_classes = len(np.unique(sp_y_train))

    return NeuralNetworkInput(
        tfidf_sp_train,
        sp_y_train,
        tfidf_sp_test,
        sp_y_test,
        wp_num_classes,
        wp_tfidf_features,
    )


def get_sentencepiece_tokenizer(
    df: DataFrame, save_tokenizer: bool = False, tweet_column: str = "tweet"
) -> spm.SentencePieceProcessor:
    """
    Returns a SentencePiece tokenizer.
    """
    df[tweet_column].to_csv("tweets.txt", index=False, header=False)
    spm.SentencePieceTrainer.Train(
        input="tweets.txt", model_prefix="lang_model", vocab_size=8000, model_type="bpe"
    )

    sp = spm.SentencePieceProcessor()
    sp.load("lang_model.model")

    if save_tokenizer:
        print("Saving SentencePiece tokenizer to lang_model.model")
        if not os.path.exists("data"):
            os.makedirs("data")
        sp.save("data/lang_model.model")

    os.remove("tweets.txt")
    return sp


def sentencepiece_tokenize_dataframe(
    df: DataFrame, sp: spm.SentencePieceProcessor, preprocess_tweets: bool = False
) -> DataFrame:
    if preprocess_tweets:
        df["tweet"] = df["tweet"].apply(preprocess_tweet)

    df["tokenized_tweets"] = df["tweet"].apply(lambda x: sp.encode(x, out_type=str))
    df["token_ids"] = df["tweet"].apply(lambda x: sp.encode(x, out_type=int))
    return df
