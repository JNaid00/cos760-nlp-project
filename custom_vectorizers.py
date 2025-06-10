from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def initialise_tfidf_vectorizer(data, ngram=None, max_features=None):
    if ngram and max_features:
        vectorizer_tf = TfidfVectorizer(ngram_range=ngram, max_features=max_features)
    elif ngram and max_features is None:
        vectorizer_tf = TfidfVectorizer(ngram_range=ngram)
    elif ngram is None and max_features:
        vectorizer_tf = TfidfVectorizer(max_features=max_features)
    else:
        vectorizer_tf = TfidfVectorizer()
    # vectorizer_tfidf = TfidfVectorizer(ngram_range=ngram,max_features=max_features)
    vectorizer_tf.fit(data)
    X = vectorizer_tf.transform(data)
    return X, vectorizer_tf


def initialise_count_vectorizer(data, ngram=None, max_features=None):
    if ngram and max_features:
        vectorizer_count = CountVectorizer(ngram_range=ngram, max_features=max_features)
    elif ngram and max_features is None:
        vectorizer_count = CountVectorizer(ngram_range=ngram)
    elif ngram is None and max_features:
        vectorizer_count = CountVectorizer(max_features=max_features)
    else:
        vectorizer_count = CountVectorizer()

    vectorizer_count.fit(data)
    X = vectorizer_count.transform(data)
    return X, vectorizer_count
