import os
import re
from pandas import DataFrame
from tokenizers import Tokenizer as WPTokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase, NFD, StripAccents
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
import os

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


def get_sentencepiece_tokenizer(
    df: DataFrame, save_tokenizer: bool = False
) -> spm.SentencePieceProcessor:
    """
    Returns a SentencePiece tokenizer.
    """
    df["tweet"].to_csv('tweets.txt', index=False, header=False)
    spm.SentencePieceTrainer.Train(input='tweets.txt', model_prefix='lang_model', vocab_size=8000, model_type='bpe')
    
    sp = spm.SentencePieceProcessor()
    sp.load('lang_model.model')
    
    if save_tokenizer:
        print("Saving SentencePiece tokenizer to lang_model.model")
        if not os.path.exists("data"):
            os.makedirs("data")
        sp.save('data/lang_model.model')
    
    os.remove('tweets.txt')
    return sp
    
def sentencepiece_tokenize_dataframe(
    df: DataFrame, sp: spm.SentencePieceProcessor, preprocess_tweets: bool = False
) -> DataFrame:
    if preprocess_tweets:
        df["tweet"] = df["tweet"].apply(preprocess_tweet)

    df["tokenized_tweets"]  = df["tweet"].apply(lambda x: sp.encode(x, out_type=str))
    df["token_ids"] = df["tweet"].apply(lambda x: sp.encode(x, out_type=int))
    return df
    

