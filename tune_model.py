import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator

from cleaning_data import DataCleaner
from read_explore_data import DatasetReader
from train_model import ModelTrainer
from vectorize_data import Vectorizer

RANDOM_SEED = 1869


class ModelTuner:

    def __init__(self):
        dr = DatasetReader(random_seed=RANDOM_SEED)
        print("Lendo dados\n")
        dr.read()
        dr.df = DataCleaner().clean(dr.df)
        df_train, df_test = dr.split_dataset()
        vec = Vectorizer()
        x_train = vec.fit_transform(df_train.text, df_train.score)
        x_test = vec.transform(df_test.text)
        y_train = df_train.score
        y_test = df_test.score
        self.data = ((x_train, y_train), (x_test, y_test))
        self.params = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def tune_ngram_model(self):
        """Tunes n-gram model on the given dataset.
        # Arguments
            data: tuples of training and test texts and labels.
        """
        # Select parameter values to try.
        num_layers = [1, 2, 3]
        num_units = [8, 16, 32, 64, 128]
        dropout_rates = [0.2, 0.3, 0.4, 0.5]
        # Save parameter combination and results.
        params = {
            'layers': [],
            'units': [],
            'dropout_rates': [],
            'acc': [],
            'loss': [],
            'auc': []
        }

        trainer = ModelTrainer()

        # Iterate over all parameter combinations.
        for drate in dropout_rates:
            for layers in num_layers:
                for units in num_units:
                    params['layers'].append(layers)
                    params['units'].append(units)
                    params['dropout_rates'].append(drate)

                    trainer.set_params(units=units, layers=layers,
                                       dropout_rate=drate)
                    accuracy, loss, hist = trainer.train(data=self.data)

                    print(('Accuracy: {accuracy}, Parameters: (layers={layers}, '
                           'units={units})').format(accuracy=accuracy,
                                                    layers=layers,
                                                    units=units))

                    for k, v in hist.history.items():
                        if not k.startswith('val'):
                            continue
                        k = k.split("_")[1]
                        params[k].append(v[-1])

        self.params = params
        return params

    def plot_params(self):
        df = pd.DataFrame(self.params)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        cols = ['units', 'layers', 'dropout_rate']
        metrics = ['acc', 'auc']
        # Plot the surface.
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                for metric in metrics:
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    surf = ax.plot_trisurf(df.iloc[:, i], df.iloc[:, j], df[metric],
                                           cmap=cm.coolwarm,
                                           linewidth=0, antialiased=True)
                    plt.show()

    def get_params(self):
        return self.params

    def print_params(self):
        print(pd.DataFrame(self.params))
