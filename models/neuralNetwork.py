import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

class neuralNetwork:


    def __init__(self):

        self._model = None
        self._seed = 42       

    def _build(self, input_dim: int):
        tf.random.set_seed(self._seed)
        m = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.30),

            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.30),

            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.20),

            layers.Dense(1, activation="sigmoid")
        ])
        m.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return m

    def train(self, X_train, y_train):
        if self._model is None:
            self._model = self._build(input_dim=X_train.shape[1])

        es  = callbacks.EarlyStopping(
                monitor="val_loss", patience=8,
                restore_best_weights=True, verbose=0
              )
        rlr = callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=4, min_lr=1e-5, verbose=0
              )

        self._model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=512,
            validation_split=0.15,
            callbacks=[es, rlr],
            verbose=0 
        )

    def predict(self, X_test):

        probs = self._model.predict(X_test, verbose=0).ravel()
        return (probs > 0.5).astype(int)