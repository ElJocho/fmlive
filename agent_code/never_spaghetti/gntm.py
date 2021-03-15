from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import numpy as np

learning_rate = 0.0005

def init_model(gamestate):
    print("Nummer 5 lebt!")
    flat_gs = gamestate.flatten()
    n_inputs = flat_gs.shape[0]

    gntm = keras.models.Sequential([
        keras.layers.Dense(n_inputs, activation="elu", input_shape=n_inputs),
        keras.layers.Dense(6, activation="sigmoid"),
    ])
    opt = Adam(learning_rate)
    gntm.compile(loss='mse', optimizer=opt)

    target_f = gntm.predict(flat_gs)
    print(target_f)
    return gntm
