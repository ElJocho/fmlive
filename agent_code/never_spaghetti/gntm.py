from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, LeakyReLU
import numpy as np
import settings as s
learning_rate = 0.0005

MUTATION_RATE = 0.001


class GNTM:

    def __init__(self, file=None):

        #num_dim = s.COLS * s.ROWS # for grid only approach
        num_dim = 5*8 + 2


        num_hidden_layer = round((num_dim/3)*2)+6  #2/3 input layer plus output layer
        gntm = keras.Sequential()
        gntm.add(Dense(num_dim, input_shape=(num_dim,)))
        gntm.add(LeakyReLU(alpha=0.1))
        gntm.add(Dense(num_hidden_layer, input_shape=(num_dim,)))
        gntm.add(LeakyReLU(alpha=0.1))
        gntm.add(Dense(6, input_shape=(num_hidden_layer,)))
        gntm.add(Activation('sigmoid'))
        gntm.compile(loss='mse', optimizer='adam')
        if file is not None:
            gntm.load_weights(file)
        self.model = gntm

    def get_model(self):
        return self.model

    def get_predict(self, gamestate, bomb_logic_arr):
        flat_gamestate = np.append(gamestate.flatten(), bomb_logic_arr)
        flat_gamestate=np.atleast_2d(flat_gamestate)
        target_f = self.model.predict(flat_gamestate)

        return np.argmax(target_f[0])


    def cell_division(self):
        """Random Mutation in some genes."""
        weights = self.model.get_weights()
        for i in range(len(weights)):
            if i % 2 == 1:
                for j in range(len(weights[i])):
                    rand_val = np.random.uniform(0,1)
                    if(rand_val > 1 - MUTATION_RATE):
                        mutation = np.random.uniform(-.5, .5)
                        weights[i][j] += mutation
            else:
                for j in range(len(weights[i])):
                    for k in range(len(weights[i][j])):
                        rand_val = np.random.uniform(0,1)
                        if(rand_val > 1 - MUTATION_RATE):
                            mutation = np.random.uniform(-.5, .5)
                            weights[i][j][k] += mutation


        self.get_model().set_weights(weights)
        return self

    def set_weights(self, weights):
        self.model.set_weights(weights=weights)
