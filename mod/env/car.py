class Car:

    # All cars
    count = 0

    IDLE = "Idle"
    RECHARGING = "Recharging"
    ASSIGN = "With passenger"
    REBALANCE = "Rebalancing"

    COMPANY_OWNED_ORIGIN = "FREE"
    COMPANY_OWNED_CONTRACT_DURATION = "INF"

    TYPE_FLEET = "AV"
    TYPE_HIRED = "FV"

    # List of car types (each type is associated to different estimates)
    car_types = [TYPE_FLEET, TYPE_HIRED]

    status_list = [IDLE, RECHARGING, ASSIGN, REBALANCE]

    def __init__(self, o, battery_level_max, battery_level_miles_max=200):
        self.id = Car.count
        self.point = o
        self.waypoint = None
        self.previous = o
        self.origin = o
        self.type = Car.TYPE_FLEET

        # Needs to reset
        self.battery_level_miles = battery_level_miles_max
        self.battery_level_max = battery_level_max
        self.battery_level_miles_max = battery_level_miles_max
        self.trip = None
        self.point_list = [self.point]

        Car.count += 1
        self.arrival_time = 0
        self.previous_arrival = 0
        self.step = 0
        self.revenue = 0
        self.n_trips = 0
        self.recharging_cost = 0
        self.distance_traveled = 0
        self.battery_level = battery_level_max
        self.recharge_count = 0

        # Vehicle starts free to operate
        self.status = Car.IDLE
        self.current_trip = None

        self.contract_duration = 32

    @property
    def attribute(self, level=0):
        return (
            self.point.id_level(level),
            self.battery_level,
            self.contract_duration,
        )

    @property
    def attribute2(self, level=0):
        return (
            Car.COMPANY_OWNED_ORIGIN,
            self.point.id_level(level),
            self.battery_level,
            Car.COMPANY_OWNED_CONTRACT_DURATION,
        )

    def attribute_level(self, level):
        return (self.point.id_level(level), self.battery_level)

    @property
    def attribute_point(self):
        return (self.point, self.battery_level)

    @property
    def busy(self):
        """Return False if car is idle"""
        if self.status == Car.IDLE:
            return False
        return True

    def status_log(self):

        trip = (
            (
                f" - Trip: [{self.trip.o.id},{self.trip.d.id}] "
                f"(dropoff={self.trip.dropoff_time:04})"
            )
            if self.trip is not None
            else ""
        )

        status = (
            f"C{self.id:04}[{self.status:>15}]"
            f" - Previous arrival: {self.previous_arrival:>5}"
            f" - Arrival: {self.arrival_time:>5}"
            f"(step={self.step:>5})"
            f" - Battery: {self.battery_level:2}/{self.battery_level_max}"
            # f"[{self.battery_level_miles:>6.2f}/"
            # f"{self.battery_level_miles_max}]"
            # f" - Traveled: {self.distance_traveled:>6.2f}"
            # f" - Revenue: {self.revenue:>6.2f}"
            # f" - #Trips: {self.n_trips:>3}"
            # f" - #Previous: {self.previous.id:>4}"
            # f" - #Waypoint: {self.waypoint.id:>4}"
            # f" - Attribute: ({self.point},{self.battery_level})"
            # f"{trip}"
        )

        return status

    def need_recharge(self, threshold):
        battery_ratio = self.battery_level_miles / self.battery_level_miles_max
        if battery_ratio < threshold:
            return True
        return False

    def update(self, step, time_increment=15):
        """Run every time_step to free vehicles that
        finished their task and update arrival times of
        idle vehicles

        Arguments:
            t {int} -- current time steps

        Keyword Arguments:
            time_increment {int} -- duration of time steps (default: {15})
        """

        # print("updating according to current time ", t)
        # If vehicle is idle, update current arrival time
        # if self.status == Car.IDLE:
        #     # print(f'car {self} is idle!')
        #     self.arrival_time = step * time_increment
        #     self.step = step

        # If car finished its task, it is currently idle
        if self.arrival_time <= step * time_increment:
            # print(f'car {self} is NO LONGER idle!')
            self.status = Car.IDLE
            self.trip = None
            self.previous = self.point
            self.previous_arrival = self.arrival_time
            self.waypoint = None

            # Update route
            if self.point != self.point_list[-1]:
                self.point_list.append(self.point)

            self.previous_battery_level = self.battery_level

            # Car is free to service users
            self.arrival_time = step * time_increment
            self.step = step

        if not self.busy:
            self.arrival_time = step * time_increment

    def has_power(self, distance):
        """Check if car has power to travel distane
        
        Arguments:
            distance {float} -- Distance in miles
        
        Returns:
            boolean -- True, if vehicle can travel distance
        """
        return self.battery_level_miles - distance > 0

    def same_region_point(self, pos, level=0):
        return self.point.id_level(level) == self.point.id_level(level)

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

        self.previous = self.point

        self.previous_battery_level = self.battery_level

        self.point = destination

        self.battery_level_miles -= distance_traveled

        self.distance_traveled += distance_traveled

        self.revenue += revenue

        self.previous_arrival = self.arrival_time

        self.arrival_time += max(duration_service, time_increment)

        self.step += max(int(duration_service / time_increment), 1)

        self.battery_level = int(
            round(
                self.battery_level_miles
                / self.battery_level_miles_max
                * self.battery_level_max
            )
        )

        self.previous_battery_level = self.battery_level
        # Cars that are busy fulfilling trips or recharging
        # are not considered to be reassigned for a decision

        if self.trip:

            self.status = Car.ASSIGN

            self.n_trips += 1

            self.trip = trip
        else:
            self.status = Car.REBALANCE

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

        self.previous = self.point

        self.previous_arrival = self.arrival_time

        self.previous_battery_level = self.battery_level

        self.point = trip.d

        self.waypoint = trip.o

        self.battery_level_miles -= distance_traveled

        self.distance_traveled += distance_traveled

        self.revenue += revenue

        self.arrival_time += max(duration_service, time_increment)

        trip.dropoff_time = self.arrival_time
        trip.picked_by = self

        # If service duration is lower than time increment, car have
        # to be free in the next time step
        self.step += max(int(duration_service / time_increment), 1)

        self.battery_level = int(
            round(
                self.battery_level_miles
                / self.battery_level_miles_max
                * self.battery_level_max
            )
        )

        self.previous_battery_level = self.battery_level
        # Cars that are busy fulfilling trips or recharging
        # are not considered to be reassigned for a decision
        self.status = Car.ASSIGN

        self.n_trips += 1

        self.trip = trip

    def get_full_recharging_miles(self):

        # Amount to recharge (miles)
        recharge_need = self.battery_level_miles_max - self.battery_level_miles

        return recharge_need

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
        self.previous_battery_level = self.battery_level
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

    def reset(self, battery_level):
        self.point = self.origin
        self.waypoint = None
        self.point_list = [self.point]
        self.arrival_time = 0
        self.previous_arrival = 0
        self.revenue = 0
        self.distance_traveled = 0
        self.battery_level = battery_level
        self.trip = None
        self.current_trip = None
        self.n_trips = 0
        self.count = 0
        self.previous = self.point
        self.recharge_count = 0
        self.recharging_cost = 0
        self.revenue = 0

    def __str__(self):
        return f"V{self.id}[{self.battery_level}] - {self.point}"

    def __repr__(self):
        return (
            f"Car{{id={self.id:02}, "
            f"(point, battery)=({self.point},{self.battery_level})}}"
        )


class HiredCar(Car):
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
        self.start_end_point = o
        self.type = Car.TYPE_HIRED
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

    @property
    def attribute2(self, level=0):
        return (
            self.start_end_point.id_level(level),
            self.point.id_level(level),
            self.battery_level,
            self.contract_duration,
        )

    def can_service(self, trip, time_step, get_distance, get_travel_time):
        dist_to_origin = get_distance(self.point, self.trip.o)
        dist_od = get_distance(self.point.o, self.trip.d)
        dist_d_start = get_distance(self.trip.d, self.start_end_point)

        total_dist = dist_to_origin + dist_od + dist_d_start

        # Next arrival
        duration_min = get_travel_time(total_dist)

    def __repr__(self):
        return (
            f"HiredCar{{id={self.id:02}, "
            f"(point, battery)=({self.point},{self.battery_level})}}"
        )

    def __str__(self):
        return f"V{self.id}[{self.contract_duration}] - {self.point}"

