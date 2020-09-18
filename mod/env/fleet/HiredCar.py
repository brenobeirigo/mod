from mod.env.fleet.Car import Car


class HiredCar(Car):
    def __init__(
        self,
        o,
        contract_duration_h,
        current_step=0,
        current_arrival=0,
        duration_level=15,
    ):
        super().__init__(o)

        self.depot = o
        self.type = Car.TYPE_HIRED
        self.started_contract = False
        self.step = current_step
        self.arrival_time = current_arrival
        self.previous_arrival = current_arrival
        self.previous_step = current_step

        # Contract
        self.total_time = contract_duration_h * 60
        self.contract_duration = int(self.total_time // duration_level)
        self.duration_level = duration_level

    def update(self, step, time_increment=1):
        super().update(step, time_increment=time_increment)

        self.total_time = max(0, self.total_time - time_increment)
        self.contract_duration = int(self.total_time / self.duration_level)

    @property
    def attribute(self, level=0):
        if self.is_rebalancing():
            point = self.middle_point.id_level(level)
            # TODO check if this influeced ADP
            # print("ATTRIBUTE:", point, self.middle_point)
        else:
            point = self.point.id_level(level)
        return (
            point,
            1,
            self.contract_duration,
            self.type,
            self.origin.id_level(level),
        )

    def __repr__(self):
        return f"HiredCar{{id={self.id:02}, point={self.point}, origin={self.origin}}}"

    @property
    def label(self):
        return f"H{self.id:04}"

    def __str__(self):
        return f"H{self.id}[{self.contract_duration}] - {self.point}({self.origin})"