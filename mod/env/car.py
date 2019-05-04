class Car:
       
    # All cars
    count = 0

    IDLE = 'Idle'
    RECHARGING = 'Recharging'
    ASSIGN = 'With passenger'
    REBALANCE = 'Rebalancing'
    status_list = [IDLE, RECHARGING, ASSIGN, REBALANCE]
    
    def __init__(self, o, battery_level_max, battery_level_miles_max = 200):
        self.id = Car.count
        self.point = o
        self.previous = o
        self.origin = o

        # Needs to reset
        self.battery_level_miles = battery_level_miles_max
        self.battery_level_max = battery_level_max
        self.battery_level_miles_max = battery_level_miles_max
        self.trip = None

        Car.count+=1
        self.arrival_time = 0
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
        self.update_attribute()
    
    @property
    def busy(self):
        #print("busy check", self.status)
        if self.status == Car.IDLE:
            return False
        return True

    def status_log(self):
        trip = (
            f' - Trip: [{self.trip.o.id},{self.trip.d.id}]'
            if self.trip is not None else ''
        )
        status = (
            f'C{self.id:04}[{self.status:>15}]'
            f' - Arrival: {self.arrival_time:>5}'
            f'(step={self.step:>5})'
            f' - Battery: {self.battery_level:2}/{self.battery_level_max}'
            f'[{self.battery_level_miles:>6.2f}/'
            f'{self.battery_level_miles_max}]'
            f' - Traveled: {self.distance_traveled:>6.2f}'
            f' - Revenue: {self.revenue:>6.2f}'
            f' - #Trips: {self.n_trips:>3}'
            f' - Attribute: {self.attribute}'
            f'{trip}'
            
        )
        return status

    def update_attribute(self):
        self.attribute = (self.point.id, self.battery_level)

    def need_recharge(self,threshold):
        battery_ratio = self.battery_level_miles/self.battery_level_miles_max
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
        
        #print("updating according to current time ", t)
        # If vehicle is idle, update current arrival time
        if self.status == Car.IDLE:
            #print(f'car {self} is idle!')
            self.arrival_time = step*time_increment
            self.step = step
        
        # If car finished its task, it is currently idle
        if self.arrival_time <= (step+1)*time_increment:
            #print(f'car {self} is NO LONGER idle!')
            self.status = Car.IDLE
            self.trip = None
            self.previous = self.point
            self.previous_battery_level = self.battery_level

    def has_power(self, distance):
        """Check if car has power to travel distane
        
        Arguments:
            distance {float} -- Distance in miles
        
        Returns:
            boolean -- True, if vehicle can travel distance
        """
        return self.battery_level_miles - distance > 0

    def update_trip(
        self, duration_service, distance_traveled,
        revenue, trip, time_increment=15
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
        
        self.point = trip.d
        
        self.battery_level_miles -= distance_traveled
        
        self.distance_traveled += distance_traveled
        
        self.revenue += revenue
        
        
        self.arrival_time += duration_service

        self.step += int(duration_service/time_increment)

        self.battery_level = int(round(
            self.battery_level_miles
            /self.battery_level_miles_max
            *self.battery_level_max
        ))
        
        self.previous_battery_level = self.battery_level
        # Cars that are busy fulfilling trips or recharging
        # are not considered to be reassigned for a decision
        self.status = Car.ASSIGN

        self.n_trips+=1

        self.trip = trip
        
        self.update_attribute()

    def get_full_recharging_miles(self):

        # Amount to recharge (miles)
        recharge_need = (
            self.battery_level_miles_max
            - self.battery_level_miles
        )

        return recharge_need

    def update_recharge(self, duration_recharging, cost, time_increment=15):
        """[summary]
        
        Arguments:
            recharge_rate {int} -- miles/hour
            cost {[type]} -- [description]
            next_point {[type]} -- [description]
        """

        self.previous_battery_level = self.battery_level

        # Update final battery level
        self.battery_level_miles = self.battery_level_miles_max
        self.battery_level = self.battery_level_max
        
        self.recharging_cost += cost
        self.arrival_time += duration_recharging
        self.step += int(duration_recharging/time_increment)
        self.arrival_time = int(self.arrival_time)



        # Cars tahte are busy fulfilling trips or recharging are not considered
        # to be reassigned for a decision
        self.status = Car.RECHARGING
        
        self.recharge_count+=1
        
        self.update_attribute()
    
    def reset(self, battery_level):
        self.point = self.origin
        self.arrival_time = 0
        self.revenue = 0
        self.distance_traveled = 0
        self.battery_level = battery_level
        self.attribute = (self.origin.id, battery_level)
        self.trip = None
        self.current_trip = None
        self.n_trips = 0
        self.count = 0
        self.previous = self.point
        self.recharge_count = 0
        self.recharging_cost = 0
        self.revenue = 0

    def __str__(self):
        return f'V{self.id}[{self.battery_level}] - {self.point}'
    
    def __repr__(self):
        return f'Car{{id={self.id:02}, (point, battery)={self.attribute}}}'
