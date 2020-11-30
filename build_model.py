from keras import models
from keras.layers import Dropout, Dense


class ModelBuilder:

    def __init__(self, layers, units, dropout_rate, input_shape):
        self.layers = layers
        self.units = units
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape

    def build(self):
        # Sigmoid activation, because it's a binary output, 0 or 1
        op_units, op_activation = 1, 'sigmoid'
        model = models.Sequential()
        model.add(Dropout(rate=self.dropout_rate, input_shape=self.input_shape))

        for _ in range(self.layers - 1):
            model.add(Dense(units=self.units, activation='relu'))
            model.add(Dropout(rate=self.dropout_rate))

        model.add(Dense(units=op_units, activation=op_activation))
        return model
