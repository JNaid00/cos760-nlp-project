import os
import pandas as pd
import subprocess
from typing import Dict, List, Optional

from constants import LOCAL_DIR_NS, NS_LANGUAGES

REPO_URL_NS = "https://github.com/hausanlp/NaijaSenti.git"
LOCAL_DIR_NS = "NaijaSenti"

REPO_URL_AS = "https://github.com/afrisenti-semeval/afrisent-semeval-2023.git"
LOCAL_DIR_AS = "afrisent-semeval-2023"


def clone_repo(repo_url: str, local_dir: str) -> None:
    if os.path.isdir(local_dir):
        print("Repository exists. Updating...")
        subprocess.run(["git", "-C", local_dir, "pull", "origin", "main"], check=True)
    else:
        print("Repository not found. Cloning...")
        subprocess.run(["git", "clone", repo_url], check=True)
 
clone_repo(REPO_URL_NS, LOCAL_DIR_NS)
clone_repo(REPO_URL_AS, LOCAL_DIR_AS)
 
 
class SplitSet:
    """
    Holds the train, test, dev splits and stopwords for a single language.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        dev: pd.DataFrame,
        stopwords: Optional[List[str]] = None,
    ):
        self.train = train
        self.test = test
        self.dev = dev
        self.stopwords = stopwords if stopwords else []

    def summary(self):
        return {
            "train_size": len(self.train),
            "test_size": len(self.test),
            "dev_size": len(self.dev),
            "num_stopwords": len(self.stopwords),
        }


class MultiLangDataset:
    """
    Manages NLP datasets split by language. Each language contains train/test/dev and stopwords.
    """

    def __init__(self):
        self.languages: Dict[str, SplitSet] = {}

    def add_language(self, lang_code: str, split_set: SplitSet):
        self.languages[lang_code] = split_set

    def get(self, lang_code: str) -> Optional[SplitSet]:
        return self.languages.get(lang_code)

    def summary(self) -> Dict[str, Dict[str, int]]:
        return {lang: split.summary() for lang, split in self.languages.items()}

    def all_languages(self) -> List[str]:
        return list(self.languages.keys())


class Languages:
    """
    Contains the language codes for NaijaSenti dataset.
    """

    HAUSA = "hau"
    IGBO = "ibo"
    NIGERIAN_PIDGIN = "pcm"
    YORUBA = "yor"


def load_local_datasets(
    local_base_dir, languages=NS_LANGUAGES, splits=["dev", "test", "train"]
):
    dataset = MultiLangDataset()

    for lang in languages:
        split_data = {}
        for split in splits:
            path = os.path.join(local_base_dir, lang, f"{split}.tsv")
            try:
                df = pd.read_csv(path, sep="\t", encoding="utf-8")
                # dataset[lang][split] = df
                # dataset.add_language(lang, df)
                split_data[split] = df
            except Exception as e:
                print(f"Failed to load {path}: {e}")

        # Read in stopwords
        if local_base_dir.startswith(LOCAL_DIR_NS):
            path = os.path.join(f"{LOCAL_DIR_NS}/data/stopwords/{lang}.csv")
            try:
                stopwords_df = pd.read_csv(path, encoding="utf-8")
                split_data["stopwords"] = stopwords_df["word"].tolist()
            except Exception as e:
                print(f"Failed to load stopwords for {lang} from {path}: {e}")

        split_set = SplitSet(
            train=split_data.get("train", pd.DataFrame()),
            test=split_data.get("test", pd.DataFrame()),
            dev=split_data.get("dev", pd.DataFrame()),
            stopwords=split_data.get("stopwords", []),
        )
        dataset.add_language(lang, split_set)
    return dataset

ns_dataset: MultiLangDataset = load_local_datasets(local_base_dir=LOCAL_DIR_NS + '/data/annotated_tweets', languages=NS_LANGUAGES)
 
as_dataset: MultiLangDataset = load_local_datasets(local_base_dir=f'afrisent-semeval-2023/data', languages=NS_LANGUAGES,)
