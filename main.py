from read_explore_data import DatasetReader
from cleaning_data import DataCleaner
from train_model import ModelTrainer
from vectorize_data import Vectorizer
from evaluate_model import ModelEvaluator
import time
import random

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
from numpy.random import seed
seed(RANDOM_SEED)


class Main:

    @staticmethod
    def execute():
        tempo = {
            'leitura': 0,
            'limpeza': 0,
            'treinamento': 0,
            'total': 0
        }
        ini_leitura = time.time()
        dr = DatasetReader(random_seed=RANDOM_SEED)
        print("Lendo dados\n")
        dr.read()
        dr.explore()
        tempo['leitura'] = time.time()-ini_leitura

        print("\n\nLimpando dados\n")
        ini_limpeza = time.time()
        cleaner = DataCleaner(language='pt', random_seed=RANDOM_SEED)
        dr.df = cleaner.clean(dr.df)
        tempo['limpeza'] = time.time() - ini_limpeza
        dr.explore()

        ini_treinamento = time.time()
        print("Dividindo o dataset em treino e teste ")
        df_train, df_test = dr.split_dataset()

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
        print("Layers: 2\nUnits: 32\nDropout_rate: 0.4")
        trainer = ModelTrainer(layers=2, units=32, dropout_rate=0.4)
        _, __, train_history = trainer.train(data)
        tempo['treinamento'] = time.time() - ini_treinamento

        evaluator = ModelEvaluator(train_history.model)
        evaluator.evaluate(x_test, y_test)

        tempo['total'] = tempo['leitura'] + tempo['limpeza'] + \
                         tempo['treinamento']
        print(tempo)

    @staticmethod
    def execute_2():
        tempo = {
            'leitura': 0,
            'limpeza': 0,
            'treinamento': 0
        }
        ini_leitura = time.time()
        dr = DatasetReader(train_path='dataset/Corona_NLP_train.csv',
                           test_path='dataset/Corona_NLP_test.csv',
                           random_seed=RANDOM_SEED,
                           text_col="OriginalTweet",
                           label_col="Sentiment")
        print("Lendo dados\n")
        dr.read()
        dr.explore()
        tempo['leitura'] = time.time()-ini_leitura

        print("\n\nLimpando dados\n")
        ini_limpeza = time.time()
        cleaner = DataCleaner(language='en', random_seed=RANDOM_SEED)
        dr.df = cleaner.clean(dr.df)
        tempo['limpeza'] = time.time() - ini_limpeza
        dr.explore()

        ini_treinamento = time.time()
        print("Dividindo o dataset em treino e teste ")
        df_train, df_test = dr.split_dataset()

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
        print("Layers: 2\nUnits: 32\nDropout_rate: 0.4")
        trainer = ModelTrainer(layers=2, units=32, dropout_rate=0.4)
        _, __, train_history = trainer.train(data)
        tempo['treinamento'] = time.time() - ini_treinamento

        evaluator = ModelEvaluator(train_history.model)
        evaluator.evaluate(x_test, y_test)

        print(tempo)


if __name__ == '__main__':
    # Main().execute()
    Main().execute_2()
