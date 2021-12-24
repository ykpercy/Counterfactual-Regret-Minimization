import numpy as np
# Dudo definitions
NUM_SIDES = 6
NUM_ACTIONS = (2 * NUM_SIDES) + 1
DUDO = NUM_ACTIONS - 1  # index claims starting at 0,"dudo" action have index 12

claimNum = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
claimRank = [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1]


def claimHistoryToString():
    """
    Convert Dudo claim history to a String
    claim history as a boolean array of length NUM ACTIONS with a value at index i being true
    if and only if the corresponding claim/dudo action has been taken.
    """
    sb = []
    for a in range(NUM_ACTIONS-1):
        if bool(a):
            if len(sb) > 0:
                sb.append(',')
            sb.append(str(claimNum[a]))
            sb.append('*')
            sb.append(str(claimRank[a]))
    return ''.join(sb)


def infoSetToInteger(playerRoll):
    """
    Convert Dudo information set to an integer
    Args:
        playerRoll: int
    """
    infoSetNum = playerRoll
    for a in range(NUM_ACTIONS-2, -1, -1):
        infoSetNum = 2 * infoSetNum + np.int(bool(a))
    return infoSetNum


class DudoUtilities:
    pass




