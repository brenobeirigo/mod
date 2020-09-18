from mod.env.fleet.Car import Car


class VirtualCar(Car):
    def __init__(self, o):
        super().__init__(o)
        self.type = Car.TYPE_VIRTUAL