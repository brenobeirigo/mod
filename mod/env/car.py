class Car:

    # All cars
    count = 0

    IDLE = 0
    RECHARGING = 1
    ASSIGN = 2
    CRUISING = 3
    REBALANCE = 4

    COMPANY_OWNED_ORIGIN = "FREE"
    COMPANY_OWNED_CONTRACT_DURATION = "INF"

    TYPE_FLEET = 0
    TYPE_HIRED = 1
    TYPE_TO_HIRE = 2

    # List of car types (each type is associated to different estimates)
    car_types = [TYPE_FLEET, TYPE_HIRED]

    DISCARD = "-"
    INFINITE_CONTRACT_DURATION = "Inf"  # - 2

    status_label_dict = {
        IDLE: "Idle",
        RECHARGING: "Recharging",
        ASSIGN: "With passenger",
        CRUISING: "Cruising",
        REBALANCE: "Rebalancing",
    }

    type_label_dict = {
        TYPE_FLEET: "AV",
        TYPE_HIRED: "FV",
        TYPE_TO_HIRE: "HIRE",
    }

    status_list = [
        IDLE,
        # RECHARGING,
        ASSIGN,
        REBALANCE,
        CRUISING,
    ]

    adp_label_dict = {DISCARD: "-", INFINITE_CONTRACT_DURATION: "Inf"}

    def __init__(self, o):
        self.id = Car.count
        self.point = o
        self.waypoint = None
        self.previous = o
        self.origin = o
        self.type = Car.TYPE_FLEET

        # TODO fix battery level
        self.battery_level = 1

        self.trip = None
        self.point_list = [self.point]

        Car.count += 1
        self.arrival_time = 0
        self.previous_arrival = 0
        self.step = 0
        self.revenue = 0
        self.n_trips = 0
        self.distance_traveled = 0

        # Vehicle starts free to operate
        self.status = Car.IDLE
        self.current_trip = None

        # Regular cars are always available
        self.contract_duration = Car.INFINITE_CONTRACT_DURATION

    @property
    def label(self):
        return f"C{self.id:04}"

    @property
    def attribute(self, level=0):
        return (
            self.point.id_level(level),
            1,
            self.contract_duration,
            self.type,
            Car.DISCARD,
        )

    def attribute_level(self, level):
        return (self.point.id_level(level),)

    @property
    def busy(self):
        """Return False if car is idle"""
        if self.status == Car.IDLE:
            return False
        return True

    def status_log(self):

        if self.trip:
            trip = (
                (
                    f" - Trip: [{self.trip.o.id:>4},{self.trip.d.id:>4}] "
                    f"(pk_step={self.trip.pk_step:>5}, dropoff={self.trip.dropoff_time:>6.2f})"
                )
                if self.trip is not None
                else ""
            )
        else:
            trip = ""

        status = (
            f"{self.label}[{Car.status_label_dict[self.status]:>15}]"
            f" - Previous arrival: {self.previous_arrival:>6.2f}"
            f" - Arrival: {self.arrival_time:>6.2f}"
            f"(step={self.step:>5})"
            f" - Traveled: {self.distance_traveled:>6.2f}"
            # f" - Revenue: {self.revenue:>6.2f}"
            # f" - #Trips: {self.n_trips:>3}"
            # f" - #Previous: {self.previous.id:>4}"
            # f" - #Waypoint: {self.waypoint.id:>4}"
            f"{trip}"
        )

        return status

    def update(self, step, time_increment=15):
        """Run every time_step to free vehicles that finished their task
        and update arrival times of idle vehicles.

        Parameters
        ----------
        step : int
            Current time step
        time_increment : int, optional
            Duration of time steps, by default 15
        """
        # TODO check consistency of steps
        if self.trip:
            if step >= self.trip.pk_step:
                self.status = Car.ASSIGN

        # If car finished its task, it is currently idle
        if self.arrival_time <= step * time_increment:
            self.status = Car.IDLE
            self.trip = None
            self.previous = self.point
            self.previous_arrival = self.arrival_time
            self.waypoint = None

            # Update route
            if self.point != self.point_list[-1]:
                self.point_list.append(self.point)

            # Car is free to service users
            self.arrival_time = step * time_increment
            self.step = step

        if not self.busy:
            self.arrival_time = step * time_increment

    def same_region_point(self, pos, level=0):
        return self.point.id_level(level) == self.point.id_level(level)

    def move(
        self,
        duration_service,
        duration_service_steps,
        distance_traveled,
        cost,
        destination,
    ):
        """Update car settings after being matched with a passenger.

        Parameters
        ----------
        duration_service : int
            Total duration to rebalance (in minutes)
        distance_traveled : float
            Distance travelede during rebalancing (in meters)
        cost : float
            Rebalancing cost
        destination : Point
            Rebalancing target location
        time_increment : int, optional
            Duration (in minutes) of each time increment, by default 15
        """

        self.previous = self.point

        self.point = destination

        self.distance_traveled += distance_traveled

        self.revenue += cost

        self.previous_arrival = self.arrival_time

        self.arrival_time += duration_service

        self.step += max(duration_service_steps, 1)

        self.status = Car.REBALANCE

    def update_trip(
        self,
        pk_duration,
        total_duration,
        distance_traveled,
        revenue,
        trip,
        duration_pickup_step=0,
        duration_total_step=0,
    ):
        """Update car settings after being matched with a passenger.
        
        Parameters
        ----------
        duration_service : int
            Cruising time + servicing time (in minutes)
        distance_traveled : float
            Cruising distance + serviciing distance (in meters)
        revenue : float
            Revenue accrued by doing task
        trip : Trip
            Trip assigned to car
        time_increment : int, optional
            Duration (in minutes) of each time increment, by default 15
        """

        self.previous = self.point

        self.previous_arrival = self.arrival_time

        self.point = trip.d

        self.waypoint = trip.o

        self.distance_traveled += distance_traveled

        self.revenue += revenue

        # Guarantee arrival time consistency
        self.arrival_time += total_duration

        trip.dropoff_time = self.arrival_time

        trip.picked_by = self

        self.trip = trip

        self.n_trips += 1

        pk_step = max(0, duration_pickup_step)

        if pk_step == 0:
            # If pk_step is zero, car is already carrying customer
            # in the subsequent time step
            self.status = Car.ASSIGN
        else:
            self.status = Car.CRUISING

        self.trip.pk_step = self.step + pk_step

        # If service duration is lower than time increment, car have
        # to be free in the next time step
        self.step += max(duration_total_step, 1)

        if self.arrival_time > self.step:
            print("What!")

    def reset(self):
        self.point = self.origin
        self.waypoint = None
        self.point_list = [self.point]
        self.arrival_time = 0
        self.previous_arrival = 0
        self.revenue = 0
        self.distance_traveled = 0
        self.trip = None
        self.current_trip = None
        self.n_trips = 0
        self.count = 0
        self.previous = self.point
        self.revenue = 0

    def __str__(self):
        return f"V{self.id} - {self.point}"

    def __repr__(self):
        return f"Car{{id={self.id:02}, " f"point={self.point}}}"


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
        pk_duration,
        total_duration,
        distance_traveled,
        revenue,
        trip,
        duration_pickup_step=0,
        duration_total_step=0,
    ):
        """Update car settings after being matched with a passenger.

        Arguments:
            duration_service {int} -- How long to pick up and deliver
            distance_traveled {float} -- Total distance to pickup and deliver
            revenue {float} -- Revenue accrued by doing task
            trip {Trip} -- Trip car is servicing
        """

        super().update_trip(
            total_duration,
            pk_duration,
            distance_traveled,
            revenue,
            trip,
            duration_pickup_step=duration_pickup_step,
            duration_total_step=duration_total_step,
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

        super().move(duration_service, distance_traveled, revenue, destination)

    @property
    def attribute(self, level=0):
        return (
            self.point.id_level(level),
            self.battery_level,
            self.contract_duration,
            self.type,
            # Origin is irrelevant for company vehicles
            Car.DISCARD,
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
                    f"(pk_step={self.trip.pk_step:>5}, dropoff={self.trip.dropoff_time:>6.2f})"
                )
                if self.trip is not None
                else ""
            )

        status = (
            f"{self.label}[{Car.status_label_dict[self.status]:>15}]"
            f" - Previous arrival: {self.previous_arrival:>6.2f}"
            f" - Arrival: {self.arrival_time:>6.2f}"
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
        self.type = Car.TYPE_TO_HIRE
        self.started_contract = False
        self.step = current_step
        self.arrival_time = current_arrival
        self.previous_arrival = current_arrival

        # Contract
        self.total_time = contract_duration_h * 60
        self.contract_duration = self.total_time // duration_level
        self.duration_level = duration_level

    def update(self, step, time_increment=1):
        super().update(step, time_increment=time_increment)

        self.total_time = max(0, self.total_time - time_increment)
        self.contract_duration = int(self.total_time / self.duration_level)

    @property
    def attribute(self, level=0):
        return (
            self.point.id_level(level),
            1,
            self.contract_duration,
            self.type,
            self.origin.id_level(level),
        )

    def __repr__(self):
        return f"HiredCar{{id={self.id:02}, " f"point={self.point}}}"

    @property
    def label(self):
        return f"H{self.id:04}"

    def __str__(self):
        return f"H{self.id}[{self.contract_duration}] - {self.point}"


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
        self.contract_duration = contract_duration_h * (60 // duration_level)
        self.depot = o
        self.type = Car.TYPE_TO_HIRE
        self.started_contract = False
        self.step = current_step
        self.arrival_time = current_arrival
        self.previous_arrival = current_arrival

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
