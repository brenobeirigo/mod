from enum import IntEnum, unique


@unique
class CarStatus(IntEnum):
    IDLE = 0
    RECHARGING = 1
    ASSIGN = 2
    CRUISING = 3
    REBALANCE = 4
    RETURN = 5
    SERVICING = 6
