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
    TYPE_TO_HIRE = "HIRE"

    # List of car types (each type is associated to different estimates)
    car_types = [TYPE_FLEET, TYPE_HIRED]

    status_list = [IDLE, RECHARGING, ASSIGN, REBALANCE]

    INFINITE_CONTRACT_DURATION = "Inf"

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
            # f" - Revenue: {self.revenue:>6.2f}"
            # f" - #Trips: {self.n_trips:>3}"
            # f" - #Previous: {self.previous.id:>4}"
            # f" - #Waypoint: {self.waypoint.id:>4}"
            f"{trip}"
        )

        return status

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

        self.point = destination

        self.distance_traveled += distance_traveled

        self.revenue += revenue

        self.previous_arrival = self.arrival_time

        self.arrival_time += max(duration_service, time_increment)

        self.step += max(int(duration_service / time_increment), 1)

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

        self.point = trip.d

        self.waypoint = trip.o

        
        self.distance_traveled += distance_traveled

        self.revenue += revenue

        self.arrival_time += max(duration_service, time_increment)

        trip.dropoff_time = self.arrival_time
        trip.picked_by = self

        # If service duration is lower than time increment, car have
        # to be free in the next time step
        self.step += max(int(duration_service / time_increment), 1)

        # Cars that are busy fulfilling trips or recharging
        # are not considered to be reassigned for a decision
        self.status = Car.ASSIGN

        self.n_trips += 1

        self.trip = trip


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
        return (
            f"Car{{id={self.id:02}, "
            f"point={self.point}}}"
        )