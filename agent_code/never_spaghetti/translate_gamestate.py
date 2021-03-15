import numpy as np


UNOCCUPIED = 0
PLAYER = 1
ENEMY = 2
WALL = 3
CRATE = 4
COIN = 5
BOMB = [-1,-2,-3,-4]
# todo if we have more than one turn where the explosions are deadly we need an array
EXPLOSION = -5


def translate_gamestate(gamestate: dict) -> np.array:
    """Translate the given gamestate to a array with values for each occupied space."""
    board = np.where(gamestate["field"] == -1, WALL, gamestate["field"])
    board = np.where(board == 1, CRATE, board)
    for bomb in gamestate["bombs"]:
        board[bomb[0][0], bomb[0][1]] = BOMB[bomb[1]]
    explosion = gamestate["explosion_map"]
    board = np.where(explosion == 1, EXPLOSION, board)
    for enemy in gamestate["others"]:
        # todo currently we ignore their bomb capability
        if enemy[2]:
            board[enemy[3][0], enemy[3][1]] = ENEMY*10
        else:
            board[enemy[3][0], enemy[3][1]] = ENEMY
    if gamestate["self"][2]:
        board[gamestate["self"][3][0], gamestate["self"][3][1]] = PLAYER*10
    else:
        board[gamestate["self"][3][0], gamestate["self"][3][1]] = PLAYER

    return board
