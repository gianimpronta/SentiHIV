from keras import metrics

from build_model import mlp_model

import tensorflow as tf


def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    # Get the data.
    (x_train, y_train), (x_val, y_val) = data

    # Create model instance.
    model = mlp_model(layers=layers,
                      units=units,
                      dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:])

    # Compile model with learning parameters.
    loss = 'binary_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    # Accuracy and AUC metrics
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['acc', metrics.AUC()])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3)]

    # Train and validate model.
    history = model.fit(
        x_train.toarray(),
        y_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val.toarray(), y_val),
        verbose=1,
        batch_size=batch_size)

    # Print results.
    history2 = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history2['val_acc'][-1], loss=history2['val_loss'][-1]))

    # Save model.
    model.save('opcovidbr_mlp_model.h5')
    return history2['acc'][-1], history2['loss'][-1], history
