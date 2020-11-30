from read_explore_data import DatasetReader
from cleaning_data import DataCleaner
from train_model import ModelTrainer
from vectorize_data import Vectorizer
from evaluate_model import ModelEvaluator

RANDOM_SEED = 1869


class Main:

    @staticmethod
    def execute():
        dr = DatasetReader(random_seed=RANDOM_SEED)
        print("Lendo dados\n")
        dr.read()
        dr.explore()

        print("\n\nLimpando dados\n")
        cleaner = DataCleaner('pt')
        dr.df = cleaner.clean(dr.df)
        dr.explore()

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

        evaluator = ModelEvaluator(train_history.model)
        evaluator.evaluate(x_test, y_test)


if __name__ == '__main__':
    Main().execute()
