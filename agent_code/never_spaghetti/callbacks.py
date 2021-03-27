import os
from .translate_gamestate import translate_gamestate, direction_based_translation, bomb_logic
from .gntm import GNTM
from .train import setup_new_round

STARTING_POSTITIONS = {
    (1, 1): "ul",
    (1, 15): "bl",
    (15, 1): "ur",
    (15, 15): "br"
}

DIRECTION_SWITCHER = {
    "ul": {"UP": "UP", "RIGHT": "RIGHT", "LEFT": "LEFT", "DOWN": "DOWN"},
    "bl": {"UP": "LEFT", "RIGHT": "UP", "LEFT": "DOWN", "DOWN": "RIGHT"},
    "br": {"UP": "DOWN", "RIGHT": "LEFT", "LEFT": "RIGHT", "DOWN": "UP"},
    "ur": {"UP": "RIGHT", "RIGHT": "DOWN", "LEFT": "UP", "DOWN": "LEFT"},
}

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("gntm.keras"):
        self.logger.info("Setting up model from scratch.")
        self.model = GNTM()
    else:
        self.logger.info("Loading model from saved state.")
        self.model = GNTM("gntm.keras")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if game_state["step"] == 1:
        if self.train:
            with open("counter.txt") as infile:
                self.counter = int(infile.readline())
            self.model = self.current_pool[self.counter]

        self.loc = (game_state["self"][3][0], game_state["self"][3][1])
        self.loc = STARTING_POSTITIONS[self.loc]
        self.loc_past = self.loc
        try:
            self.loc_arr.append(self.loc)
        except:
            pass

    try:
        self.loc = STARTING_POSTITIONS[(game_state["self"][3][0], game_state["self"][3][1])]
    except:
        pass
    trans_state = direction_based_translation(game_state, self.loc)
    bomb_logic_arr = bomb_logic(game_state)
    self.logger.debug("Querying model for action.")
    wanted_action = ACTIONS[self.model.get_predict(trans_state, bomb_logic_arr)]
    if "WAIT" == wanted_action or "BOMB" == wanted_action:
        return wanted_action
    else:
        return DIRECTION_SWITCHER[self.loc][wanted_action]
