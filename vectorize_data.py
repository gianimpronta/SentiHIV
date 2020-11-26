import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif


class Vectorizer:

    def __init__(self,
                 ngram_range=(1, 2),  # Unigrams and bigrams
                 top_k=20000,   #
                 token_mode='word',  # Could be 'words' or 'chars'
                 min_document_frequency=2  # only select words that appear at
                 # least twice in documents
                 ):

        # Limiting the number of features
        self.top_k = top_k

        kwargs = {
            'ngram_range': ngram_range,
            'dtype': np.float32,
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': token_mode,
            'min_df': min_document_frequency,
        }
        self._vectorizer = TfidfVectorizer(**kwargs)
        self._selector = None

    def fit(self, text, labels):

        tfidf = self._vectorizer.fit_transform(text)

        # Using F-test to select K best features
        self._selector = SelectKBest(f_classif,
                                     k=min(self.top_k, tfidf.shape[1]))
        self._selector.fit(tfidf, labels)

    def transform(self, text):
        tfidf = self._vectorizer.transform(text)
        selected = self._selector.transform(tfidf)
        selected = selected.astype('float32')
        return selected

    def fit_transform(self, text, labels):
        self.fit(text, labels)
        return self.transform(text)
