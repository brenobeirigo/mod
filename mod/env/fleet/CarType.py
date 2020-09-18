from enum import IntEnum, unique


@unique
class CarType(IntEnum):
    TYPE_FLEET = 0
    TYPE_HIRED = 1
    TYPE_VIRTUAL = 3
