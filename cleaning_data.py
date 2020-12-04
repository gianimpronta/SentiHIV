import re

import nltk
import pandas as pd
import stanza
from sklearn.utils import shuffle

nltk.download('stopwords')
from nltk.corpus import stopwords


class NotSupportedError(Exception):
    pass


class DataCleaner:

    def __init__(self, language='pt', random_seed=123):
        if language not in ['pt', 'en']:
            raise NotSupportedError(
                "language not supported. valid languages: 'pt', 'en'.")
        self.language = language
        stanza.download(language)
        self.nlp = stanza.Pipeline(
            lang=language, processors='tokenize,mwt,pos,lemma')
        self.random_seed = random_seed

    @staticmethod
    def remove_urls(text):
        url_remover = re.compile(r'https?\w+')
        return url_remover.sub(r'', text)

    @staticmethod
    def remove_html(text):
        html_remover = re.compile(r'<.*?>')
        return html_remover.sub(r'', text)

    @staticmethod
    def remove_mentions(text):
        mention_remover = re.compile(r'@\w+')
        return mention_remover.sub(r'', text)

    @staticmethod
    def remove_numbers(text):
        number_remover = re.compile(r'\d+')
        return number_remover.sub(r'', text)

    @staticmethod
    def remove_hashtags(text):
        number_remover = re.compile(r'#\w+')
        return number_remover.sub(r'', text)

    @staticmethod
    def remove_punctuation(text):
        punct_remover = re.compile(r'[^\w\s\d]+')
        return punct_remover.sub(r'', text)

    @staticmethod
    def remove_excessive_whitespace(text):
        ws_remover = re.compile(r'\s+')
        return ws_remover.sub(r' ', str(text)).strip()

    @staticmethod
    def remove_stopwords(text, stop_words):
        return " ".join(
            [word for word in text.split(" ") if word not in stop_words])

    @staticmethod
    def remove_lonely_letter(text):
        return " ".join(
            [word for word in text.split(" ") if len(word) > 1])

    @staticmethod
    def lowering(text):
        return text.lower()

    @staticmethod
    def lemmatization(text, nlp):
        doc = nlp(text)
        return ' '.join(
            [f'{word.lemma}' for sent in doc.sentences for word in sent.words])

    def clean(self, df):
        try:
            df = pd.read_csv('dataset/clean_opcovidbr.csv', index_col=0)
        except FileNotFoundError:
            if self.language == 'pt':
                stop_words = set(stopwords.words('portuguese'))
            else:
                stop_words = set(stopwords.words('english'))

            if "twitter" in df.columns and "polarity" in df.columns:
                df = df.loc[:, ["twitter", "polarity"]]
                df = df[-df.polarity.isnull()]
                df["score"] = df.polarity
                df["score"] = df.score.apply(lambda x: 0 if x == -1 else 1)
                df["text"] = df.twitter

            if "OriginalTweet" in df.columns and "Sentiment" in df.columns:
                df = df.loc[:, ["OriginalTweet", "Sentiment"]]
                df = df[-df.Sentiment.isnull()]
                score = {
                    "Extremely Negative": 0,
                    "Negative": 0,
                    "Neutral": -1,
                    "Positive": 1,
                    "Extremely Positive": 1,
                }

                df["score"] = df.Sentiment.apply(lambda x: score[x])
                df = df[-(df.score == -1)]
                df["text"] = df.OriginalTweet

            df["text"] = df.text.apply(lambda x: DataCleaner.remove_mentions(x))
            df["text"] = df.text.apply(lambda x: DataCleaner.remove_html(x))
            df["text"] = df.text.apply(lambda x: DataCleaner.remove_numbers(x))
            df["text"] = df.text.apply(lambda x: DataCleaner.remove_hashtags(x))
            df["text"] = df.text.apply(lambda x: DataCleaner.lowering(x))
            df["text"] = df.text.apply(
                lambda x: DataCleaner.remove_stopwords(x, stop_words))
            df["text"] = df.text.apply(lambda x: DataCleaner.remove_punctuation(x))
            df["text"] = df.text.apply(lambda x: DataCleaner.remove_urls(x))
            df["text"] = df.text.apply(
                lambda x: DataCleaner.remove_excessive_whitespace(x))
            df["text"] = df.text.apply(
                lambda x: DataCleaner.remove_lonely_letter(x))

            df = df[df.text.apply(lambda x: len(x.split(" ")) > 2)]

            df["text"] = df.text.apply(
                lambda x: DataCleaner.lemmatization(x, self.nlp))

            df = shuffle(df, random_state=self.random_seed)
            df.to_csv('dataset/clean_opcovidbr.csv')

        return df
