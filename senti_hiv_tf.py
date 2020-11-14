"""Module to train n-gram model.

Vectorizes training and validation texts into n-grams and uses that for
training a n-gram model - a simple multi-layer perceptron model. We use n-gram
model for text classification when the ratio of number of samples to number of
words per sample for the given dataset is very small (<~1500).
"""
from __future__ import (
    absolute_import, absolute_import, division, division,
    print_function, print_function
)

import re

import nltk
nltk.download('stopwords')
import stanza
stanza.download('pt')

import pandas as pd

from nltk.corpus import stopwords

import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense, Dropout

# Vectorization parameters

FLAGS = None

# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 500


def read():
    df = pd.read_csv('dataset/Covid BR Tweets/opcovidbr.csv', index_col='Id')
    df = df[-df.polarity.isnull()]
    return train_test_split(df, test_size=0.2, random_state=123)


def remove_urls(text):
    url_remover = re.compile(r'https?://\S+|www\.\S+')
    return url_remover.sub(r'', text)


def remove_html(text):
    html_remover = re.compile(r'<.*?>')
    return html_remover.sub(r'', text)


def remove_mentions(text):
    mention_remover = re.compile(r'@\w+')
    return mention_remover.sub(r'', text)


def remove_numbers(text):
    number_remover = re.compile(r'\d+')
    return number_remover.sub(r'', text)


def remove_hashtags(text):
    number_remover = re.compile(r'#\w+')
    return number_remover.sub(r'', text)


def remove_punctuation(text):
    punct_remover = re.compile(r'[^\w\s\d]+')
    return punct_remover.sub(r'', text)


def remove_excessive_whitespace(text):
    ws_remover = re.compile(r'\s+')
    return ws_remover.sub(r' ', str(text)).strip()


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    # stop_words = set(stopwords.words('portuguese'))
    return " ".join(
        [word for word in text.split(" ") if word not in stop_words])


def lowering(text):
    return text.lower()


def lemmatization(text, nlp):
    doc = nlp(text)
    return ' '.join([f'{word.lemma}' for sent in doc.sentences for word in \
            sent.words])


def clean(df):
    df = df.loc[:, ["twitter", "polarity"]]
    df["score"] = df.polarity
    df["text"] = df.twitter
    df["text"] = df.text.apply(lambda x: remove_urls(x))
    df["text"] = df.text.apply(lambda x: remove_mentions(x))
    df["text"] = df.text.apply(lambda x: remove_html(x))
    df["text"] = df.text.apply(lambda x: remove_numbers(x))
    df["text"] = df.text.apply(lambda x: remove_hashtags(x))
    df["text"] = df.text.apply(lambda x: remove_punctuation(x))
    df["text"] = df.text.apply(lambda x: remove_excessive_whitespace(x))
    df["text"] = df.text.apply(lambda x: remove_stopwords(x))
    df["text"] = df.text.apply(lambda x: lowering(x))
    nlp = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos,lemma')
    df["text"] = df.text.apply(lambda x: lemmatization(x, nlp))

    # Removing messages that are too short.
    df = df[df.text.apply(lambda x: len(x) > 2)]

    df.loc[df.score == -1, 'score'] = 0

    return df


def cleaning():
    df_train, df_test = read()
    df_train = clean(df_train)
    df_test = clean(df_test)

    return df_train, df_test


def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as ngram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of uni-grams + bi-grams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
        'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': TOKEN_MODE,  # Split text into word tokens.
        'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train)
    x_val = selector.transform(x_val)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    return x_train, x_val


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = 2
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
            unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val = ngram_vectorize(train_texts, train_labels, val_texts)

    # Create model instance.
    model = mlp_model(layers=layers,
                      units=units,
                      dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:],
                      num_classes=num_classes)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
        x_train.toarray(),
        train_labels,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val.toarray(), val_labels),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('imdb_mlp_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


if __name__ == '__main__':
    df_train, df_test = cleaning()

    print("Reading")
    data = (df_train.text, df_train.score), (df_test.text, df_test.score)
    print("Training")
    train_ngram_model(data, epochs=1000)
