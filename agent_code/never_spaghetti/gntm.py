from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation
import numpy as np
import settings as s
learning_rate = 0.0005

class GNTM:

    def __init__(self, file=None):

        #num_dim = s.COLS * s.ROWS # for grid only approach
        num_dim = 5*8 + 2


        num_hidden_layer = round((num_dim/3)*2)+6  #2/3 input layer plus output layer
        gntm = keras.Sequential()
        gntm.add(Dense(num_dim, input_shape=(num_dim,)))
        gntm.add(Activation('relu'))
        gntm.add(Dense(num_hidden_layer, input_shape=(num_dim,)))
        gntm.add(Activation('relu'))
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
            for j in range(len(weights[i])):
                if(np.random.uniform(0,1) > .95):
                    change = np.random.uniform(-0.5,0.5)  # was .5
                    weights[i][j] += change
        self.get_model().set_weights(weights)
        return self


    def set_weights(self, weights):
        self.model.set_weights(weights=weights)