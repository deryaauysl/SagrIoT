"""Small Keras MLP for the 36-step soil-moisture change forecast."""
import numpy as np
import tensorflow as tf


def build(n_inputs, hidden, n_outputs=36, X_train=None, l2=0.0, dropout=0.0, skip=False):
    norm = tf.keras.layers.Normalization()
    if X_train is not None:
        norm.adapt(X_train)
    inp = tf.keras.Input((n_inputs,))
    z = norm(inp)
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    x = z
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation="relu", kernel_regularizer=reg)(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(n_outputs, kernel_regularizer=reg)(x)
    if skip:   # linear path + small nonlinear correction
        out = tf.keras.layers.Add()([out, tf.keras.layers.Dense(n_outputs, kernel_regularizer=reg)(z)])
    return tf.keras.Model(inp, out)


class MLPRegressor:
    """fit/predict wrapper so the MLP plugs into Learned like the ridge model."""

    def __init__(self, hidden=(32,), seed=0, epochs=150, patience=10, lr=1e-3, batch=256, l2=0.0, dropout=0.0, skip=False):
        self.hidden, self.seed, self.epochs, self.patience, self.lr, self.batch = hidden, seed, epochs, patience, lr, batch
        self.l2, self.dropout, self.skip = l2, dropout, skip

    def fit(self, X, Y, X_val, Y_val):
        tf.keras.utils.set_random_seed(self.seed)
        self.model = build(X.shape[1], self.hidden, Y.shape[1], X, self.l2, self.dropout, self.skip)
        self.model.compile(tf.keras.optimizers.Adam(self.lr), loss="mae")
        stop = tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
        h = self.model.fit(X, Y, validation_data=(X_val, Y_val), epochs=self.epochs,
                           batch_size=self.batch, callbacks=[stop], verbose=0)
        self.epochs_run, self.val_mae = len(h.history["loss"]), min(h.history["val_loss"])
        return self

    def params(self):
        return self.model.count_params() - self.model.layers[1].count_params()   # exclude normalization stats

    def predict(self, x):
        return self.model(x, training=False).numpy()
