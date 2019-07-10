from amod.Car import Car

class ElectricCar(Car):

    def __init__(self, o, battery_level_max, battery_level_miles_max=200):

        super().__init__(o)

        # Needs to reset
        self.battery_level_miles = battery_level_miles_max
        self.battery_level_max = battery_level_max
        self.battery_level_miles_max = battery_level_miles_max
        self.recharging_cost = 0
        self.battery_level = battery_level_max
        self.recharge_count = 0

    def attribute_level(self, level):
        return (self.point.id_level(level), self.battery_level)

    def update_trip(
        self,
        duration_service,
        distance_traveled,
        revenue,
        trip,
        time_increment=15,
    ):
        """Update car settings after being matched with a passenger.

        Arguments:
            duration_service {int} -- How long to pick up and deliver
            distance_traveled {float} -- Total distance to pickup and deliver
            revenue {float} -- Revenue accrued by doing task
            trip {Trip} -- Trip car is servicing
        """

        super().update_trip(
            duration_service,
            distance_traveled,
            revenue,
            trip,
            time_increment=time_increment,
        )

        self.battery_level_miles -= distance_traveled

        self.battery_level = int(
            round(
                self.battery_level_miles
                / self.battery_level_miles_max
                * self.battery_level_max
            )
        )

    def reset(self, battery_level):
        super().reset()
        self.battery_level = battery_level
        self.recharge_count = 0
        self.recharging_cost = 0

    def has_power(self, distance):
        """Check if car has power to travel distane

        Arguments:
            distance {float} -- Distance in miles

        Returns:
            boolean -- True, if vehicle can travel distance
        """
        return self.battery_level_miles - distance > 0


    def get_full_recharging_miles(self):

        # Amount to recharge (miles)
        recharge_need = self.battery_level_miles_max - self.battery_level_miles

        return recharge_need

    def move(
        self,
        duration_service,
        distance_traveled,
        revenue,
        destination,
        trip=None,
        time_increment=15,
    ):
        """Update car settings after being matched with a passenger.

        Arguments:
            duration_service {int} -- How long to pick up and deliver
            distance_traveled {float} -- Total distance to pickup and deliver
            revenue {float} -- Revenue accrued by doing task
            trip {Trip} -- Trip car is servicing
        """

        self.battery_level_miles -= distance_traveled

        self.battery_level = int(
            round(
                self.battery_level_miles
                / self.battery_level_miles_max
                * self.battery_level_max
            )
        )

        super().move(
            duration_service,
            distance_traveled,
            revenue,
            destination,
            trip=trip,
            time_increment=time_increment,
        )

    @property
    def attribute(self, level=0):
        return (
            self.point.id_level(level),
            self.battery_level,
            self.contract_duration,
            self.type,
        )
    
    
    def update_recharge(self, duration, cost, extra_dist, time_increment=15):
        """Recharge car.

        Parameters
        ----------
        duration : int
            Minutes recharging
        cost : float
            Cost of recharging
        extra_dist : int
            Extra distance car can travel after recharging.
        time_increment : int, optional
            Time increment, by default 15
        """

        # Store previous car info
        self.previous_arrival = self.arrival_time

        # Sum increment to battery level (max = battery size)
        self.battery_level_miles = min(
            self.battery_level_miles + extra_dist, self.battery_level_miles_max
        )

        # Get new battery level
        self.battery_level = int(
            self.battery_level_miles
            / self.battery_level_miles_max
            * self.battery_level_max
        )

        # Update car's recharging cost
        self.recharging_cost += cost

        # when duration recharging is lower than time increment, car
        # still moves to next time step.
        self.arrival_time += max(duration, time_increment)
        self.step += max(int(duration / time_increment), 1)

        # Cars tahte are busy fulfilling trips or recharging are not considered
        # to be reassigned for a decision
        self.status = Car.RECHARGING

        # How many times has the car recharged?
        self.recharge_count += 1

    def __str__(self):
        return f"V{self.id}[{self.battery_level}] - {self.point}"

    def __repr__(self):
        return (
            f"Car{{id={self.id:02}, "
            f"(point, battery)=({self.point},{self.battery_level})}}"
        )
    
    def need_recharge(self, threshold):
        battery_ratio = self.battery_level_miles / self.battery_level_miles_max
        if battery_ratio < threshold:
            return True
        return False

    def status_log(self):

        if self.trip:
            trip = (
                (
                    f" - Trip: [{self.trip.o.id:>4},{self.trip.d.id:>4}] "
                    f"(dropoff={self.trip.dropoff_time:04})"
                )
                if self.trip is not None
                else ""
            )

        status = (
            f"{self.label}[{self.status:>15}]"
            f" - Previous arrival: {self.previous_arrival:>5}"
            f" - Arrival: {self.arrival_time:>5}"
            f"(step={self.step:>5})"
            f" - Traveled: {self.distance_traveled:>6.2f}"
            f" - Battery: {self.battery_level:2}/{self.battery_level_max}"
            # f" - Revenue: {self.revenue:>6.2f}"
            # f" - #Trips: {self.n_trips:>3}"
            # f" - #Previous: {self.previous.id:>4}"
            # f" - #Waypoint: {self.waypoint.id:>4}"
            # f" - Attribute: ({self.point},{self.battery_level})"
            # f"[{self.battery_level_miles:>6.2f}/"
            # f"{self.battery_level_miles_max}]"
            f"{trip}"
        )

        return status
