from collections import namedtuple, deque
from typing import List
import json
import events as e
import os
from .pool_party import init_pool, mutate_models
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


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    exterminate_old_models = False
    self.loc_arr = []
    self.max_models = 100
    self.current_pool=[]
    self.actions = []

    if exterminate_old_models:
        self.counter = 0
        for f in os.listdir("pool"):
            if not f.endswith(".keras"):
                continue
            os.remove(os.path.join("pool", f))
        self.current_pool = init_pool(self.current_pool, num_models=self.max_models)
        self.fitness_list = [-999 for x in range(self.max_models)]
    else:
        with open("counter.txt") as infile:
            self.counter = int(infile.readline())
        with open(os.path.join(file_path, "fitness.json"), "r") as infile:
            self.fitness_list = json.load(infile)

        self.current_pool = init_pool(num_models=self.max_models,current_pool=self.current_pool, from_file=True)
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    #init_pool(num_models=1)

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

    if self.counter != self.max_models-1:
        calculate_fitness(self)
        setup_new_round(self, self.counter+1)
    else:
        outcomes={"ur": [], "ul": [], "br": [], "bl": []}
        for i, elem in enumerate(self.loc_arr):
            outcomes[elem].append(self.fitness_list[i])
        for key, val in outcomes.items():
             print(key, ": ", round((len([values for values in val if values > 0])/len(val)*100), 1))
        self.loc_arr = []
        calculate_fitness(self)
        print("")
        print(self.fitness_list, statistics.mean(self.fitness_list))
        self.current_pool = mutate_models(self, fitness=self.fitness_list, current_pool=self.current_pool)
        setup_new_round(self, self.counter-(self.max_models-1))
        self.fitness_list = [-999 for x in range(self.max_models)]
        print(f"Completed round!")


def setup_new_round(self, new_counter):
    """Calculate fitness of current Model then switch to new Model for next rounds."""
    self.actions.clear()
    self.counter = new_counter
    with open("counter.txt", "w") as outfile:
        outfile.write(str(self.counter))
    self.model = self.current_pool[self.counter]


def calculate_fitness(self):
    """
    Calculate fitness based on encountered actions.
    """
    fitness_influences = {
        e.COIN_COLLECTED: 50,
        e.BOMB_DROPPED: 30,
        e.KILLED_OPPONENT: 30,
        e.KILLED_SELF: -80,
        e.GOT_KILLED: -1,
        e.CRATE_DESTROYED: 10,
        e.INVALID_ACTION: -.5,
        e.MOVED_UP: .1,
        e.MOVED_DOWN: .1,
        e.MOVED_LEFT: .1,
        e.MOVED_RIGHT: .1,
        #e.WAITED: -.5,
    }
    fitness = 0
    for event in self.actions:
        if event in fitness_influences:
            fitness += fitness_influences[event]
    self.logger.info(f"Agent has a fitness of {fitness}")

    self.fitness_list[self.counter] = fitness
    with open(os.path.join(file_path, "fitness.json"), "w") as outpath:
        json.dump(self.fitness_list, outpath)
