import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


class DatasetReader:

    def __init__(self, train_path='dataset/Covid BR Tweets/opcovidbr.csv',
                 test_path=None, random_seed=1869, text_col='twitter',
                 label_col='polarity'):
        self._path = train_path
        self._test_path = test_path
        self.df = None
        self.test_df = None
        self.random_seed = random_seed
        self._text_col = text_col
        self._label_col = label_col

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df):
        self._df = new_df

    @property
    def test_df(self):
        return self._test_df

    @test_df.setter
    def test_df(self, new_df):
        self._test_df = new_df

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, new_seed):
        self._random_seed = new_seed

    def read(self):
        self.df = pd.read_csv(self._path, index_col=0)
        if self._test_path:
            self.test_df = pd.read_csv(self._test_path, index_col=0)
        df = pd.concat([self.df, self.test_df])
        return df

    def explore(self, df=None):
        if not df:
            df = self.df

        clean = 'score' in df.columns

        print('='*80)
        if clean:
            print("Dataset limpo")
        else:
            print("Dataset bruto")
        print('-'*80)
        print(f'Tamanho do dataset: {df.shape[0]} linhas e {df.shape[1]} colunas')
        print(f'Colunas: {", ".join(df.columns)}')

        s = 'Valores nulos por coluna:\n'
        for col in df.columns:
            s += f'\t{col}: {sum(df[col].isnull())}\n'
        print(s)

        if clean:
            print(
                f'Valores de score: {", ".join([str(_) for _ in df.score.unique()])}')
            print(f'Balanceamento de classes de score:\n')
            print(df.score.value_counts())
            print()
            print("Tamanho dos textos:")
            print(df.text.apply(lambda x: len(x)).describe())
        else:
            print(f'Valores de polaridade: '
                  f''
                  f'{", ".join([str(_) for _ in df[self._label_col].unique()])}\n')
            print(f'Balanceamento de classes de polaridade:\n')
            print(df[self._label_col].value_counts())
            print()
            print("Tamanho dos textos:")
            print(df[self._text_col].apply(lambda x: len(x)).describe())

        print("5 primeiras linhas:")
        print(df.head())
        print("5 últimas linhas:")
        print(df.tail())

        print('='*80)

    def split_dataset(self, random_state=None,
                      test_size=0.3, df=None, **kwargs):
        if not random_state:
            random_state = self.random_seed

        if not df:
            df = self.df
        return train_test_split(df, test_size=test_size,
                                random_state=random_state, **kwargs)


if __name__ == '__main__':
    dr = DatasetReader()
    dr.read()
    dr.explore()
    df_train, df_test = dr.split_dataset()

