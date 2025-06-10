import os
import re
from pandas import DataFrame
from tokenizers import Tokenizer as WPTokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase, NFD, StripAccents


def get_tokenizer(df: DataFrame, save_tokenizer: bool = False) -> WPTokenizer:
    """
    Returns the WordPiece tokenizer.
    """
    tokenizer = WPTokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=8000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    # Collect all tweets into a single list for training
    tweets = df["tweet"].tolist()
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
    df: DataFrame, tokenizer: WPTokenizer, preprocess_tweets: bool = False
) -> DataFrame:
    """
    Tokenizes the 'tweet' column of a DataFrame using a WordPiece tokenizer.
    Adds two new columns: 'tokenized_tweets' and 'token_ids'.
    """
    if preprocess_tweets:
        df["tweet"] = df["tweet"].apply(preprocess_tweet)
    df["tokenized_tweets"] = df["tweet"].apply(lambda x: tokenizer.encode(x).tokens)
    df["token_ids"] = df["tweet"].apply(lambda x: tokenizer.encode(x).ids)
    return df
