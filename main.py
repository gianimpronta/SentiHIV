from read_explore_data import read, explore, split_dataset
from cleaning_data import clean
from train_model import train_ngram_model
from vectorize_data import Vectorizer
from evaluate_model import evaluate

RANDOM_SEED = 1869


def main():
    print("Lendo dados\n")
    df = read()
    explore(df)

    print("\n\nLimpando dados\n")
    df = clean(df)
    explore(df)

    print("Dividindo o dataset em treino e teste ")
    df_train, df_test = split_dataset(df, RANDOM_SEED)

    print("\n\nVetorizando dados\n")
    vec = Vectorizer()
    x_train = vec.fit_transform(df_train.text, df_train.score)
    x_test = vec.transform(df_test.text)
    y_train = df_train.score
    y_test = df_test.score

    print("5 primeiras linhas vetorizadas do dataset de treino")
    print(x_train[:5])
    print("5 primeiras linhas vetorizadas do dataset de teste")
    print(x_test[:5])

    data = ((x_train, y_train), (x_test, y_test))

    print("Treinando modelo usando os seguintes parâmetros:")
    print("Layers: 2\nUnits: 64\nDropout_rate: 0.2")
    _, __, train_history = train_ngram_model(data)

    evaluate(train_history.model, x_test, y_test)


if __name__ == '__main__':
    main()
