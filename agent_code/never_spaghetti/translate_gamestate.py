import numpy as np
from typing import Union
import settings as s

UNOCCUPIED = 0
PLAYER = 1
ENEMY = 2
WALL = 3
CRATE = 4
COIN = 5
BOMB = [-1, -2, -3, -4]
# todo if we have more than one turn where the explosions are deadly we need an array
EXPLOSION = -5
MAX_DIST = np.linalg.norm(np.array((s.COLS - 2, s.ROWS - 2)))


def direction_based_translation(gamestate: dict, starting_loc: str) -> np.array:
    """Translate gamestate to a 8*5 array which indicates directions of elements."""
    location = np.flip(np.array(gamestate["self"][3]))

    directions = get_directions(location)
    # walls n crates
    walls = np.zeros(8)
    i = 0

    for key, value in directions.items():
        if gamestate["field"][value] == -1:
            walls[i] = 1
        i += 1

    wall_ind = np.array(np.where(gamestate["field"] == 1))
    crates = separate_values(
        direction_sensor(location=location, objects=wall_ind.T, index=None)
    )

    bombs = direction_sensor(
        location=location, objects=gamestate["bombs"], index=0, bomb=True
    )
    enemies = direction_sensor(location, gamestate["others"], 3)
    coins = direction_sensor(location, gamestate["coins"], None)

    inputs = [walls, crates, bombs, enemies, coins]
    relative_inputs = []
    if starting_loc != "ul":
        for input in inputs:
            relative_inputs.append(rotate_fov(input, starting_loc))
    else:
        relative_inputs = inputs
    return np.append(
        [
            relative_inputs[0],
        ],
        relative_inputs[1:],
        axis=0,
    )


def direction_sensor(location: np.array, objects: list, index: Union[int, None], bomb: bool = False) -> np.array:
    """We fill it clockwise starting from upper left."""
    result = np.zeros(8)
    for object in objects:
        if index is None:
            x, y = object
        else:
            x, y = object[index]
        if y < location[0]:
            if x < location[1]:
                result[0] = max(result[0], relative_normed_dist(location, (y, x), bomb))
            elif x == location[1]:
                result[1] = max(result[1], relative_normed_dist(location, (y, x), bomb))
            else:
                result[2] = max(result[2], relative_normed_dist(location, (y, x), bomb))
        elif y == location[0]:
            if x < location[1]:
                result[7] = max(result[7], relative_normed_dist(location, (y, x), bomb))
            elif x == location[1]:
                pass
            else:
                result[3] = max(result[3], relative_normed_dist(location, (y, x), bomb))
        else:
            if x < location[1]:
                result[6] = max(result[6], relative_normed_dist(location, (y, x), bomb))
            elif x == location[1]:
                result[5] = max(result[5], relative_normed_dist(location, (y, x), bomb))
            else:
                result[4] = max(result[4], relative_normed_dist(location, (y, x), bomb))

    return result


def relative_normed_dist(a: tuple, b: tuple, bomb: bool = False) -> float:
    """Calculate a value which serves as the distance from self to object."""
    dist = np.linalg.norm(np.array(a) - np.array(b))
    if bomb:
        if dist <= 4:
            return 1
        else:
            return 1 - np.linalg.norm(np.array(a) - np.array(b)) / (MAX_DIST - 4)
    else:
        return 1 - np.linalg.norm(np.array(a) - np.array(b)) / MAX_DIST


def get_directions(location: np.array) -> dict:
    """Directions are based on compass directions."""
    j, i = location
    directions = {
        "nw": (i - 1, j - 1),
        "w": (i, j - 1),
        "sw": (i + 1, j - 1),
        "s": (i + 1, j),
        "se": (i + 1, j + 1),
        "e": (i, j + 1),
        "ne": (i - 1, j + 1),
        "n": (i - 1, j),
    }
    return directions


def rotate_fov(vec: np.array, starting_loc: str) -> np.array:
    """Rotate in a way that gntm always feels like it is in the upper left corner."""
    if starting_loc == "bl":
        rotate = 3
    elif starting_loc == "br":
        rotate = 2
    elif starting_loc == "ur":
        rotate = 1
    else:
        raise ValueError
    for _ in range(rotate):
        first_2 = np.copy(vec[:2])
        vec[:6] = vec[2:]
        vec[6:] = first_2
    return vec


def bomb_logic(gamestate: dict) -> np.array:
    """Can it lay a bomb and is it on a bomb."""
    result = np.zeros(2)
    if gamestate["self"][2]:
        result[0] = 1
    if (gamestate["self"][3][0], gamestate["self"][3][1]) in [
        bomb[0] for bomb in gamestate["bombs"]
    ]:
        result[1] = 1
    return result


def separate_values(crates: np.array) -> np.array:
    """Differentiate between next to a crate and close to a crate."""
    crates = np.where(crates >= 0.935, 1, crates)
    crates = np.where(crates < 1, crates / 2, crates)
    return crates
