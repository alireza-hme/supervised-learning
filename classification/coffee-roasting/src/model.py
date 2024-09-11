import tensorflow as tf
import numpy as np

from coffee_utils import load_coffee_data


X, Y = load_coffee_data()

# normalize data
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)
X_normal = norm_l(X)


# increase training set data
Xt = np.tile(X_normal, (1000, 1))
Yt = np.tile(Y, (1000, 1))


tf.random.set_seed(1234)  # applied to achieve consistent results

model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(3, activation="sigmoid", name="layer1"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="layer2"),
    ]
)

model.summary()
