from collections import namedtuple, deque
from typing import List
import numpy as np
import json
import events as e
import os
from .pool_party import init_pool, mutate_models
from .translate_gamestate import direction_based_translation
from copy import deepcopy
file_path = os.path.dirname(os.path.realpath(__file__))
import statistics
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
NUMBER_OF_ROUNDS = 1

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    exterminate_old_models = False
    self.loc_arr = []
    self.max_models = 50
    self.current_pool=[]
    self.actions = []

    self.round_results = []
    if exterminate_old_models:
        self.counter = 0
        for f in os.listdir("pool"):
            if not f.endswith(".keras"):
                continue
            os.remove(os.path.join("pool", f))
        self.current_pool = init_pool(self.current_pool, num_models=self.max_models)
        self.fitness_list = [-999 for x in range(self.max_models)]
        with open(os.path.join(file_path, "fitness_coin_master.json"), "w") as infile:
            json.dump([0],infile)
    else:
        with open(os.path.join(file_path, "fitness.json"), "r") as infile:
            self.fitness_list = json.load(infile)


        self.current_pool = init_pool(num_models=self.max_models,current_pool=self.current_pool, from_file=True)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if old_game_state is not None:
        last_state = direction_based_translation(old_game_state, self.loc)
        if last_state[1 , 1::2].any() == 1 and self_action == "BOMB" and old_game_state["self"][2]:
            events.append(e.BOMB_NEXT_TO_CRATE)
        if e.BOMB_EXPLODED in events and (e.CRATE_DESTROYED not in events and e.KILLED_OPPONENT not in events):
            events.append(e.BOMB_USELESS)

        coin_dir = last_state[4,1::2]


        if self.loc == 'ur':
            temp = deepcopy(coin_dir[3])
            coin_dir[1:4] = coin_dir[0:3]
            coin_dir[0] = temp
        if self.loc == 'br':
            temp = deepcopy(coin_dir[2:4])
            coin_dir[2:4] = coin_dir[0:2]
            coin_dir[0:2] = temp
        if self.loc == 'bl':
            temp = deepcopy(coin_dir[0])
            coin_dir[0:3] = coin_dir[1:4]
            coin_dir[3] = temp
        if (coin_dir[0] > 0 and self_action == 'UP'):
            events.append(e.MOVED_TOWARDS_COIN)
        if (coin_dir[1] > 0 and self_action == 'RIGHT'):
            events.append(e.MOVED_TOWARDS_COIN)
        if (coin_dir[2] > 0 and self_action == 'DOWN'):
            events.append(e.MOVED_TOWARDS_COIN)
        if (coin_dir[3] > 0 and self_action == 'LEFT'):
            events.append(e.MOVED_TOWARDS_COIN)


    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    self.actions += events


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.info(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.actions += events
    self.round_results.append(calculate_fitness(self))

    # checking if all coins got collected
    last_state = direction_based_translation(last_game_state, self.loc)
    if all(np.array(last_state[4]) == 0):
        events.append(e.ALL_COINS_COLLECTED)

    if len(self.round_results) < NUMBER_OF_ROUNDS:
        self.actions.clear()
    else:
        self.fitness_list[self.counter] = statistics.mean(self.round_results)
        self.round_results.clear()
        if self.counter != self.max_models-1:
            setup_new_round(self, self.counter + 1)
        else:
            self.loc_arr = []
            print("")
            print(self.fitness_list, statistics.mean(self.fitness_list))
            self.current_pool = mutate_models(self, fitness=self.fitness_list, current_pool=self.current_pool)
            setup_new_round(self, self.counter-(self.max_models-1))
            self.fitness_list = [-999 for x in range(self.max_models)]
            print(f"Completed round!")


def setup_new_round(self, new_counter, preserve_own=False):
    """Calculate fitness of current Model then switch to new Model for next rounds."""
    self.actions.clear()
    if not preserve_own:
        self.counter = new_counter
        with open("counter.txt", "w") as outfile:
            outfile.write(str(self.counter))
    else:
        with open("counter.txt", "w") as outfile:
            outfile.write(str(new_counter))
def calculate_fitness(self):
    """
    Calculate fitness based on encountered actions.
    """
    fitness_influences = {
        e.COIN_COLLECTED: 100,
        e.INVALID_ACTION: -1,
        e.WAITED: -50,
        e.BOMB_DROPPED: -50,
        e.MOVED_TOWARDS_COIN: 1
        #e.KILLED_OPPONENT: 30,
        #e.KILLED_SELF: -4,
        #e.GOT_KILLED: -1,
        #e.BOMB_NEXT_TO_CRATE: 15,
        #e.BOMB_USELESS: -4.99
    }
    unique, counts = np.unique(np.array(self.actions), return_counts=True)
    print(self.loc, dict(zip(unique, counts)))
    fitness = 0
    for event in self.actions:
        if event in fitness_influences:
            fitness += fitness_influences[event]
    self.logger.info(f"Agent has a fitness of {fitness}")

    self.fitness_list[self.counter] = fitness
    with open(os.path.join(file_path, "fitness.json"), "w") as outpath:
        json.dump(self.fitness_list, outpath)

    return fitness