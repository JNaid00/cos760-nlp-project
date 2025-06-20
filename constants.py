from enum import Enum

REPO_URL_NS = "https://github.com/hausanlp/NaijaSenti.git"
LOCAL_DIR_NS = "NaijaSenti"

REPO_URL_AS = "https://github.com/afrisenti-semeval/afrisent-semeval-2023.git"
LOCAL_DIR_AS = "afrisent-semeval-2023"

NS_LANGUAGES = ["hau", "ibo", "pcm", "yor"]


class TokenizerEnum(Enum):
    WORDPIECE = "wordpiece"
    BPE = "bpe"  # Need to look into
    SENTENCEPIECE = "sentencepiece"


class VectorizerEnum(Enum):
    TFIDF = "tfidf"
    BOW = "bow"  # Count Vectorizer
