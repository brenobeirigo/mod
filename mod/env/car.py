import collections
import mod.util.log_util as log_util


class Car:

    # All cars
    count = 0

    # Cars cannot visit the last tabu locations
    SIZE_TABU = 5

    IDLE = 0
    RECHARGING = 1
    ASSIGN = 2
    CRUISING = 3
    REBALANCE = 4
    RETURN = 5
    SERVICING = 6

    statuses = [
        IDLE,
        RECHARGING,
        ASSIGN,
        CRUISING,
        REBALANCE,
        RETURN,
        SERVICING,
    ]

    COMPANY_OWNED_ORIGIN = "FREE"
    COMPANY_OWNED_CONTRACT_DURATION = "INF"

    TYPE_FLEET = 0
    TYPE_HIRED = 1
    TYPE_VIRTUAL = 3

    DISCARD_BATTERY = 1

    A_LOCATION = 0
    A_BATTERY = 1
    A_CONTRACT = 2
    A_TYPE = 3
    A_STATION = 4
    A_ARRIVAL = 5

    # List of car types (each type is associated to different estimates)
    car_types = [TYPE_FLEET, TYPE_HIRED]

    DISCARD = "-"
    INFINITE_CONTRACT_DURATION = "Inf"  # - 2

    status_label_dict = {
        IDLE: "Parked",
        RECHARGING: "Recharging",
        ASSIGN: "With passenger",
        CRUISING: "Driving to pick up",
        REBALANCE: "Repositioning",
        RETURN: "Returning",
        SERVICING: "Servicing passenger",
    }

    type_label_dict = {TYPE_FLEET: "AV", TYPE_HIRED: "FV"}

    status_list = [
        IDLE,
        # RECHARGING,
        ASSIGN,
        REBALANCE,
        CRUISING,
        RETURN,
    ]

    adp_label_dict = {DISCARD: "-", INFINITE_CONTRACT_DURATION: "Inf"}

    def m_data(self):
        return (
            "## "
            f"middle={self.middle_point}, "
            f"elapsed_distance={self.elapsed_distance:6.2f}, "
            f"time_o_m={self.time_o_m:6.2f}, "
            f"distance_o_m={self.distance_o_m:6.2f}, "
            f"elapsed={self.elapsed:6.2f}, "
            f"remaining_distance={self.remaining_distance:6.2f}, "
            f"step_m={self.step_m:>4}"
        )

    def __init__(self, o):
        self.id = Car.count
        # Current node or destination
        self.point = o
        self.waypoint = None
        # Last point visited
        self.previous = o
        # Starting point
        self.origin = o

        # Middle point data
        self.middle_point = o
        self.elapsed_distance = 0
        self.time_o_m = 0
        self.distance_o_m = 0
        self.elapsed = 0
        self.remaining_distance = 0
        self.step_m = 0

        self.type = Car.TYPE_FLEET
        self.idle_step_count = 0
        self.interrupted_rebalance_count = 0

        self.tabu = collections.deque([o.id], Car.SIZE_TABU)

        # TODO fix battery level
        self.battery_level = Car.DISCARD_BATTERY

        self.trip = None
        self.point_list = [self.point]

        Car.count += 1
        self.arrival_time = 0
        self.previous_arrival = 0
        self.previous_step = 0
        self.step = 0
        self.revenue = 0
        self.n_trips = 0
        self.distance_traveled = 0

        # Vehicle starts free to operate
        self.status = Car.IDLE
        self.current_trip = None

        self.time_status = dict()
        for s in Car.statuses:
            self.time_status[s] = list()

        # Regular cars are always available
        self.contract_duration = Car.INFINITE_CONTRACT_DURATION

    @property
    def label(self):
        return f"C{self.id:04}"

    @property
    def attribute(self, level=0):
        """Car attributes at aggregation level "level" for location.
        When car is rebalancing, returns the location correspond to the
        middle point.
        
        Parameters
        ----------
        level : int, optional
            Location RC level, by default 0 (disaggregate)
        
        Returns
        -------
        tuple
            (point(level), battery, contract_duration, type, home station)
        """
        if self.status == Car.REBALANCE:
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
            Car.DISCARD,
            # self.step,
        )

    def interrupt_rebalance(self):
        """Make car idle to pickup request"""
        self.interrupted_rebalance_count += 1
        self.point = self.middle_point
        self.arrival_time = self.previous_arrival + self.time_o_m
        self.distance_traveled -= self.remaining_distance

        # Cut time spent rebalancing
        self.time_status[self.status][-1] = self.time_o_m
        # Approximated, since car will be actually avaliable in "elapsed"
        # time units.
        self.step = self.step_m
        self.status = Car.IDLE

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
                    f"(pk_step={self.trip.pk_step:>5}, "
                    f"pk_delay={self.trip.pk_delay:>6.2f}, "
                    f"dropoff={self.trip.dropoff_time:>6.2f}, "
                    f"placement={self.trip.placement}, "
                    f"elapsed={self.trip.elapsed_sec:>3}, "
                    f"class={self.trip.sq_class}, "
                    f"max_wait={self.trip.max_delay_from_placement:>6.2f})"
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
            f"(previous={self.previous_step:>5},step={self.step:>5})"
            f" - Traveled: {self.distance_traveled:>6.2f}"
            f" -> ({self.previous},{self.middle_point},{self.point})"
            f" - interruped={self.interrupted_rebalance_count:>4}"
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
            self.previous_step = self.step
            self.waypoint = None

            # If car was rebalancing, update middle point information
            self.middle_point = self.point
            self.elapsed = 0
            self.elapsed_distance = 0
            self.time_o_m = 0
            self.distance_o_m = 0
            self.remaining_distance = 0
            self.step_m = step

            # Update route
            if self.point != self.point_list[-1]:
                self.point_list.append(self.point)

            # Time car spent PARKED since arrival
            self.time_status[self.status].append(
                step * time_increment - self.arrival_time
            )
            # print(self.time_status[self.status])

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
        return_trip=False,
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
        # Car will not move back to the last places it has visited
        self.tabu.append(self.point.id)

        # self.idle_step_count+=1
        self.idle_step_count = 0

        self.previous = self.point

        self.point = destination

        self.distance_traveled += distance_traveled

        self.revenue += cost

        self.previous_arrival = self.arrival_time

        self.previous_step = self.step

        self.arrival_time += duration_service

        self.step += max(duration_service_steps, 1)

        if return_trip:
            self.status = Car.RETURN
        else:
            self.status = Car.REBALANCE

        self.time_status[self.status].append(duration_service)

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

        self.previous_step = self.step

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
            # in the subsequent time step. That is because car is
            # EXACTLY in customer location.
            self.status = Car.ASSIGN
        else:
            self.status = Car.CRUISING

        self.time_status[Car.ASSIGN].append(total_duration - pk_duration)
        self.time_status[Car.CRUISING].append(pk_duration)

        self.trip.pk_step = self.step + pk_step

        # How long to pick up the user
        self.trip.pk_delay = pk_duration

        # If service duration is lower than time increment, car have
        # to be free in the next time step
        self.step += max(duration_total_step, 1)

        # if self.arrival_time > self.step:
        #     print("What!", self.arrival_time, self.step)

        # After trip, vehicle is free again to rebalance
        self.tabu = collections.deque([self.point.id], Car.SIZE_TABU)

        self.idle_step_count = 0

    def reset(self):
        self.point = self.origin
        self.waypoint = None
        self.point_list = [self.point]
        self.arrival_time = 0
        self.previous_arrival = 0
        self.previous_step = 0
        self.revenue = 0
        self.distance_traveled = 0
        self.trip = None
        self.current_trip = None
        self.n_trips = 0
        self.count = 0
        self.previous = self.point
        self.revenue = 0
        self.time_status = dict()
        for s in Car.statuses:
            self.time_status[s] = list()

    def __str__(self):
        return f"V{self.id} - {self.point}"

    def __repr__(self):
        return (
            f"Car{{id={self.id:02},"
            f"previous={self.previous}, "
            f"middle={self.middle_point}, "
            f"point={self.point}, "
            f"status={self.status}}}"
        )


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
        return_trip=False,
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
            return_trip=return_trip,
        )

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

        self.previous_step = self.step

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
                    f"(pk_step={self.trip.pk_step:>5}, "
                    f"pk_delay={self.trip.pk_delay:>6.2f}, "
                    f"dropoff={self.trip.dropoff_time:>6.2f})"
                )
                if self.trip is not None
                else ""
            )

        status = (
            f"{self.label}[{Car.status_label_dict[self.status]:>15}]"
            f"{log_util.format_tuple(self.attribute)}"
            f" - steps=[{self.previous_step} ({self.previous_arrival:>6.2f} min),"
            f" {self.step:>5} ({self.arrival_time:>6.2f} min)]"
            f" - traveled: {self.distance_traveled:>6.2f} km"
            f""
            # f" - battery: {self.battery_level:2}/{self.battery_level_max}"
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


class VirtualCar(Car):
    def __init__(self, o):
        super().__init__(o)
        self.type = Car.TYPE_VIRTUAL


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
        return (
            self.point.id_level(level),
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
