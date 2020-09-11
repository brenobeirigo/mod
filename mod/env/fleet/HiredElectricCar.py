from mod.env.fleet.Car import Car
from mod.env.fleet.ElectricCar import ElectricCar


class HiredElectricCar(ElectricCar):
    def __init__(
        self,
        o,
        battery_level_max,
        contract_duration_h,
        current_step=0,
        current_arrival=0,
        battery_level_miles_max=200,
        duration_level=15,
    ):
        super().__init__(o, battery_level_max, battery_level_miles_max)
        self.contract_duration = int(
            contract_duration_h * (60 // duration_level)
        )
        self.depot = o
        self.type = Car.TYPE_HIRED
        self.started_contract = False
        self.step = current_step
        self.arrival_time = current_arrival
        self.previous_arrival = current_arrival
        self.previous_step = current_step

        # Contract
        self.total_time = contract_duration_h * 60
        self.duration_level = duration_level

    def update(self, step, time_increment=1):
        super().update(step, time_increment=time_increment)

        self.total_time = max(0, self.total_time - time_increment)
        self.contract_duration = int(self.total_time / self.duration_level)

    def __repr__(self):
        return (
            f"HiredElectricCar{{id={self.id:02}, "
            f"(point, battery)=({self.point},{self.battery_level})}}"
        )

    @property
    def label(self):
        return f"EH{self.id:04}"

    def __str__(self):
        return f"EH{self.id}[{self.contract_duration}] - {self.point}"