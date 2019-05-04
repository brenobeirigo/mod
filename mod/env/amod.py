from mod.env.car import Car
from mod.env.trip import Trip
from mod.env.network import Point, get_point_list
import itertools as it
from bidict import bidict
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from operator import itemgetter
import scipy
import numpy as np
import random
from pprint import pprint

class ActionSpaceSparse(object):

    def __init__(self):
        pass
    
    
    def get_action_tuple(self, cars):

        actions = defaultdict(int)

        for c in cars:

            # What is the decision taken by vehicle c
            decision = ActionSpace.get_decision(c)

            # TODO is it really previous???
            actions[((c.previous.id, c.previous_battery_level), decision)]+=1

        comb = tuple([a+(v,) for a,v in actions.items()])
        return comb

class ActionSpace(object):

    def __init__(self, car_attributes, decisions):
        self.car_attributes = car_attributes
        self.decisions = decisions
        
        # Declare the complete action space
        self.actions = list(it.product(
            self.car_attributes,
            self.decisions
        ))

        self.size = len(self.actions)

        # Associate each single action (a, d) to an id
        self.action_id_map = bidict(
            {a:k for k, a in enumerate(self.actions)}
        )
    
    def get_single_action_tuple(self, id):
        return self.action_id_map.inverse[id]

    def get_single_action_id(self, action):
        return self.action_id_map[action]
    
    def get_action_tuple(self, cars):

        actions = defaultdict(int)

        for c in cars:

            # What is the decision taken by vehicle c
            decision = ActionSpace.get_decision(c)

            actions[(c.attribute, decision)]+=1

        comb = tuple([a+(v,) for a,v in actions.items()])
        return comb


    def get_action_vector(self, cars):

        actions = np.zeros(self.size, dtype=int)

        for c in cars:

            decision = ActionSpace.get_decision(c)

            action_id = self.action_id_map[(c.attribute, decision)]
            actions[action_id]+=1
 
        return actions

    @staticmethod
    def get_decision(car):
        """Return the decision tuple derived from car.
        

        Arguments:
            car {Car} -- Vehicle after decision is made
        
        Returns:
            tuple -- decision code, from zone, to zone

        E.g.:
        decision (0, 1, 2) - Car is servicing customer (flag=0)
        going from zone 1 to zone 2
        """

        if car.status == Car.IDLE:
            decision = (
                Amod.TRIP_STAY_DECISION,
                car.point.id,
                car.point.id
            )
        elif car.status == Car.ASSIGN:
            decision = (
                Amod.TRIP_STAY_DECISION,
                car.trip.o.id,
                car.trip.d.id
            )
        elif car.status == Car.RECHARGING or car.status == Car.REBALANCE:
            decision = (
                Amod.RECHARGE_REBALANCE_DECISION,
                car.previous.id,
                car.point.id
            )
        return decision


    @staticmethod
    def read_single_action_tuple(action):
        car_attribute, decision, number_time_decision = action
        # Car attributes
        point_id, battery_level = car_attribute
        
        # Decision attributes
        action_code, from_point_id, to_point_id = decision

        # Decision category
        decision_category = (
            'servicing/staying' if action_code == Amod.TRIP_STAY_DECISION else
            'recharging/rebalancing'
        )

        print(
            f'- [{number_time_decision:>3} car(s)]'
            f' in [point = {point_id:>5}]'
            f' and [battery level = {battery_level:>3}]'
            f' is {decision_category} from->to'
            f' [origin = {from_point_id:>5}, destination = {to_point_id:>5}]'
        )

class Amod:
    # Decision codes
    
    # In a zoned environment with (z1, z2) cells signals:
    #  - trip from z1 to z2
    #  - stay in zone z1 = z2
    TRIP_STAY_DECISION = 0

    # In a zoned environment with (z1, z2) cells signals:
    #  - rebalance from z1 to z2
    #  - recharge in zone z1 = z2
    RECHARGE_REBALANCE_DECISION = 1

    def __init__(
        self, config, car_positions = []):
        """Start AMoD environment
        
        Arguments:
            rows {int} -- Number of zone rows
            cols {int} -- Number of zone columns
            fleet_size {int} -- Number of cars
            battery_size {int} -- Battery max. capacity
            actions {list} -- Possible actions from each state
            car_positions -- Where each vehicle is in the beginning.
        """
        self.config = config
        self.time_steps = config.time_steps
        
        # Defining the operational map
        self.n_zones = self.config.rows*self.config.cols
        zones = np.arange(self.n_zones)
        self.zones = zones.reshape((self.config.rows, self.config.cols))
        
        # Defining map points
        self.points = get_point_list(self.config.rows,self.config.cols)
        self.point_list_ids = [l.id for l in self.points]
        
        # Battery levels -- l^{d}
        self.battery_levels = config.battery_levels
        self.battery_size_miles = config.battery_size_miles
        self.fleet_size = config.fleet_size
        self.car_origin_points = [
            point for point in random.choices(
                self.points,
                k = self.fleet_size
            )
        ]

        # Start Q table
        self.Q = defaultdict(lambda: defaultdict(float))

        # If no list predefined positions
        if not car_positions:
            # Creating random fleet starting from random points
            self.cars = [
                Car(
                    point,
                    self.battery_levels,
                    battery_level_miles_max=self.battery_size_miles
                )
                for point in self.car_origin_points
            ]
        else:
            # Creating random fleet starting from random points
            self.cars = [
                Car(
                    point,
                    self.battery_levels,
                    battery_level_miles_max=self.battery_size_miles
                )
                for point in car_positions
            ]

    ################################################################
    # Network ######################################################
    ################################################################
    def get_travel_time(self,distance):
        """Travel time in minutes given distance in miles"""
        travel_time_h = distance / self.config.speed_mph
        travel_time_min = travel_time_h * 60
        return travel_time_min

    def get_travel_time_od(self,o, d):
        """Travel time in minutes given Euclidean distance in miles
        between origin o and destination d"""
        distance = self.get_distance(o,d)
        return self.get_travel_time(distance)

    def get_distance(self,o,d):
        """Receives two points of a gridmap and returns
        Euclidian distance.
        
        Arguments:
            o {Point} -- Origin point
            d {Point} -- Destination point
        
        Returns:
            float -- Euclidian distance
        """
        return self.config.zone_widht*np.linalg.norm(
            np.array([o.x, o.y])
            -np.array([d.x, d.y])
        )

    def pickup2(self,trip, car, update = False):
        """Insert trip into car and update car status.
        
        Arguments:
            trip {Trip} -- Trip matched to vehicle
            car {Car} -- Car in which trip will be inserted
        
        Return:
            float -- Reward gained by car after servicing trip
        """
        # Distance car has to travel to service trip
        distance_trip = self.get_distance(trip.o, trip.d)


        # Distance to pickup passanger
        distance_pickup = self.get_distance(car.point, trip.o)
        
        # Total distance
        total_distance = (
            distance_pickup
            + distance_trip
        )

        # Reward
        reward = self.config.calculate_fare(distance_trip)

        # Next arrival
        duration_min = int(round(
            self.get_travel_time(total_distance)
        ))

        if update:
            # Update car data
            car.update_trip(
                duration_min,
                total_distance,
                reward,
                trip
            )

            # Associate car to trip
            trip.picked_by = car
        # print(
        #     f'Trip total distance ({trip.o}->{trip.d}) = {total_distance:.2f}'
        #     f' miles ({distance_pickup:.2f}+{distance_trip:.2f})'
        #     f' (~{duration_min}min) - Fares: ${reward:.2f}'
        # )

        return duration_min, total_distance, reward

    def get_dict_cars_per_attribute(self, cars):
        
        dict_cars_per_attribute = defaultdict(list)
        for c in cars:
            dict_cars_per_attribute[c.attribute].append(c)
        
        return dict_cars_per_attribute

    def get_cars_state_list(self, cars):
        return [c.attribute for c in cars]

    def get_car_state_tuple_from_cars(self, cars):
        # Rt = the change in the number of cars due to information
        # arriving between t-1 and t (e.g., cars entering/leaving 
        # the service, or random delays in the arrival of cars moving
        # from one point to another


        # Get state of all free cars (e.g., [(p1,b1), (p2,b2), ...])
        car_states = [c.attribute for c in self.cars if not c.busy]

        # Rta = number of resources with attribute vector a at time t
        resources_per_attribute = Counter(car_states)
        
        car_state_tuple = list()
        for a, count in resources_per_attribute.items():
            car_state_tuple.append((a + (count,)))
        
        car_state_tuple.sort(key=itemgetter(0,1))

        return tuple(car_state_tuple)
    

    def full_recharge(self, car):
        """Recharge car fully and update its parameters.
        Car will be available only when recharge is complete.
        
        Arguments:
            car {Car} -- Car to be recharged
        
        Returns:
            float -- Cost of recharge
        """
        # How many rechargeable miles vehicle have
        miles = car.get_full_recharging_miles()
        
        # How much time vehicle needs to recharge
        time_min, steps = self.config.get_full_recharging_time(miles)

        # Total cost of rechargin
        cost = self.config.calculate_cost_recharge(time_min)

        # Update vehicle status to recharging
        car.update_recharge(time_min, cost)

        return cost

    def get_car_state_tuple_from_cars2(self, cars):
        # Rt = the change in the number of cars due to information
        # arriving between t-1 and t (e.g., cars entering/leaving 
        # the service, or random delays in the arrival of cars moving
        # from one point to another


        # Get state of all free cars (e.g., [(p1,b1), (p2,b2), ...])
        car_states = [c.attribute for c in self.cars if not c.busy]

        car_per_attribute = self.get_dict_cars_per_attribute(cars)


        # Rta = number of resources with attribute vector a at time t
        resources_per_attribute = Counter(car_states)
        
        car_state_tuple = list()
        for a, list_cars in car_per_attribute.items():
            car_state_tuple.append((a + (len(list_cars),)))
        
        car_state_tuple.sort(key=itemgetter(0,1))

        return tuple(car_state_tuple), car_per_attribute

    def get_car_state_vector_from_cars(self, cars):
        # Rt = the change in the number of cars due to information
        # arriving between t-1 and t (e.g., cars entering/leaving 
        # the service, or random delays in the arrival of cars moving
        # from one point to another

        # Rt = resource state vector at time t
        car_state_vector = np.zeros(len(self.car_attributes), dtype=int)

        # Get state of all cars (e.g., [(p1,b1), (p2,b2), ...])
        car_states = self.get_cars_state_list(self.cars)

        # Rta = number of resources with attribute vector a at time t
        resources_per_attribute = Counter(car_states)
        
        for a, count in resources_per_attribute.items():
            car_state_vector[self.dict_car_states[a]] = count

        return car_state_vector

    def get_current_action(self, cars):

        actions = np.zeros(self.action_space.size, dtype=int)

        for c in cars:

            #print("Car status:", c.status, [Car.IDLE, Car.ASSIGN, Car.RECHARGING, Car.REBALANCE])

            if c.status == Car.IDLE:
                decision = (Amod.TRIP_STAY_DECISION, c.point.id, c.point.id)
            elif c.status == Car.ASSIGN:
                decision = (Amod.TRIP_STAY_DECISION, c.trip.o.id, c.trip.d.id)
            elif c.status == Car.RECHARGING or c.status == Car.REBALANCE:
                decision = (Amod.RECHARGE_REBALANCE_DECISION, c.previous.id, c.point.id)

            action_id = self.action_space.dict[(c.attribute, decision)]
            actions[action_id]+=1
 
        return actions

    def get_trip_state_vector(self, trips_t):
        """Get trip state vector from list of trips.
        
        Arguments:
            trips_t {list} -- List of trips
        
        Returns:
            array -- List of trip count per (origin, destination) pair

        Example:

        map = [[1,2],
               [3,4]]

        Possible trips:
            [(1,2) (1,3) (1,4)
             (2,1) (2,3) (2,4)
             (3,1) (3,2) (3,4)
             (4,1) (4,2) (4,3)]

        >> trips = [Trip(1,2),Trip(1,2), Trip(3,4)]
        >> get_trip_state_vector(trips)
        >>  [2 0 0
             0 0 0
             0 0 1
             0 0 0]

        """

        #Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
        trip_per_attribute = Counter([t.attribute for t in trips_t])

        # Dt = number of new trips that  first became known to the
        # system between time t-1 and t
        trip_state_vector = np.zeros(len(self.trip_attributes), dtype=int)
        
        # How many trips per attribute?
        for b, count in trip_per_attribute.items():
            trip_state_vector[self.dict_trip_states[b]] = count
        
        return trip_state_vector

    def get_trip_state_tuple(self, trips_t):
        """Get trip state vector from list of trips.
        
        Arguments:
            trips_t {list} -- List of trips
        
        Returns:
            array -- List of trip count per (origin, destination) pair

        Example:

        map = [[1,2],
               [3,4]]

        Possible trips:
            [(1,2) (1,3) (1,4)
             (2,1) (2,3) (2,4)
             (3,1) (3,2) (3,4)
             (4,1) (4,2) (4,3)]

        >> trips = [Trip(1,2),Trip(1,2), Trip(3,4)]
        >> get_trip_state_tuple(trips)
        >>  ((1,2,2), (3,4,1))

        """

        #Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
        trip_per_attribute = Counter([t.attribute for t in trips_t])
        
        trip_state_tuple = list()
        # How many trips per attribute?
        for b, count in trip_per_attribute.items():
            # (b, count) = (from point_id, to point_id, count)
            trip_state_tuple.append((b + (count,)))
        
        trip_state_tuple.sort(key=itemgetter(0,1))
        return tuple(trip_state_tuple)

    
    def get_trip_state_tuple2(self, trips_t):
        """Get trip state vector from list of trips.
        
        Arguments:
            trips_t {list} -- List of trips
        
        Returns:
            array -- List of trip count per (origin, destination) pair

        Example:

        map = [[1,2],
               [3,4]]

        Possible trips:
            [(1,2) (1,3) (1,4)
             (2,1) (2,3) (2,4)
             (3,1) (3,2) (3,4)
             (4,1) (4,2) (4,3)]

        >> trips = [Trip(1,2),Trip(1,2), Trip(3,4)]
        >> get_trip_state_tuple(trips)
        >>  ((1,2,2), (3,4,1))

        """

        #Dictionary of #trips per trip attribute,i.e., (o.id, d.id)
        dict_trip_per_attribute = defaultdict(list)

        for t in trips_t:
            dict_trip_per_attribute[t.attribute].append(t)

        trip_state_tuple = list()
        # How many trips per attribute?
        for b, trip_list in dict_trip_per_attribute.items():
            # (b, count) = (from point_id, to point_id, count)
            count = len(trip_list)
            # print('Count:', count)
            trip_state_tuple.append((b + (count,)))
        
        trip_state_tuple.sort(key=itemgetter(0,1))
        return tuple(trip_state_tuple), dict_trip_per_attribute
    

    def start_action_space(self):

        self.action_space = ActionSpace(
            self.car_attributes,
            self.decisions
        )
    
    def start_action_space_sparse(self):

        self.action_space = ActionSpaceSparse()


    def get_state(self, time_step, cars, trips):
        trip_state = self.get_trip_state_vector(trips)
        car_state = self.get_car_state_vector_from_cars(cars)
        t_state = np.array([time_step], dtype=int)
        return np.concatenate((t_state, car_state, trip_state))

    def get_state_sparse(self, time_step, cars, trips):
        trip_state = self.get_trip_state_tuple(trips)
        car_state = self.get_car_state_tuple_from_cars(cars)
        
        if not car_state or not trip_state:
            return None
        return (time_step, car_state, trip_state)

    def get_state_sparse2(self, time_step, cars, trips):
        trip_state, trips_per_attribute = self.get_trip_state_tuple2(trips)
        if not trip_state:
            return None, None, None

        car_state, cars_per_attribute = self.get_car_state_tuple_from_cars2(cars)
        if not car_state:
            return None, None, None
        
        return (time_step, car_state, trip_state), cars_per_attribute, trips_per_attribute

    def get_action(self, cars):
        trip_state = self.get_trip_state_vector(trips)
        car_state = self.get_car_state_vector_from_cars(cars)
        t_state = np.array([time_step], dtype=int)
        return np.concatenate((t_state, car_state, trip_state))


    def print_dimension(self, max_number_trips, max_time_steps):
        """Print problem's dimension in terms of number of states and actions.
        
        Arguments:
            max_number_trips {int} -- Maximum number of tris occurring in
                a time step. 
            max_time_steps {int} -- The duration of an episode
        """


        n_zones = self.config.rows*self.config.cols

        # Car attributes (zone id, battery level) (A)
        car_attributes = n_zones*self.config.battery_levels

        # Outcome space - Expectation over a vector of random variables
        # The resource state vector at time t
        cars_per_attribute =  scipy.misc.comb(
            car_attributes + len(self.cars) - 1,
            len(self.cars)
        )

        # How many (o,d) possibilities (B)
        trip_attributes = n_zones*n_zones

        # Dtb is the number of trips with attribute b, Dt = (Dtb)b in B
        trips_per_attribute =  scipy.misc.comb(
            trip_attributes + max_number_trips -1,
            max_number_trips
        )

        # St = (Rt, Dt)
        physical_state_count = cars_per_attribute * trips_per_attribute


        print(
            '##### Problem dimension ############################################'
            f'\n - Number of zones (A): {n_zones}'
            f'\n - Car attributes (A): {car_attributes}'
            f'\n - Trip attributes (B): {trip_attributes}' 
            f'\n - Resource state vector (Rt): {physical_state_count}'
            f'\n - Trip state vector (Dt): {trips_per_attribute}'
            f'\n - Physical state (St): {physical_state_count}'
            f'\n - States in episode ({max_time_steps} steps): {physical_state_count*max_time_steps}'
        )
    
    # def get_trip_attribute(self):
    #     # B = set of trip attributes (o,d)
    #     return 
    #     self.trip_attributes = list(it.product(
    #         self.point_list_ids,
    #         self.point_list_ids)
    #     )
    def print_state_space(self):
        print("################ State space ####################")
        
        print(
            f'# Car attributes -- [format=(zone_id, battery_level),'
            f' total={len(self.car_attributes)}]:'
        )
        pprint(dict(self.dict_car_states))

        print(
            f'\n# Trip attributes -- [format=(from_zone_id, to_zone_id),'
            f' total={len(self.trip_attributes)}]:'
        )
        pprint(dict(self.dict_trip_states))

    def start_decision_space(self):

        # All car states
        car_state_vector = np.zeros(len(self.car_attributes), dtype=int)

        # Decisions to cover trips (and stay where o=d)
        trip_stay = [
            (Amod.TRIP_STAY_DECISION,) + d
            for d in self.trip_attributes
        ]
        # Decisions to stay
        #stay = [(l.id, l.id) for l in self.points]
        
        # Decisions to move empty/rebalance (and recharge where o=d)
        recharge_rebalance = [
            (Amod.RECHARGE_REBALANCE_DECISION,) + d
            for d in self.trip_attributes
        ]
        # Decisions to recharge
        # recharge = [(l.id, l.id) for l in self.points]
        
        # Decision variables
        self.decisions = trip_stay + recharge_rebalance #+ recharge + stay

        # Every trip t atribute (t.o, t.d)
        #  is associated to an index
        self.dict_decisions = bidict(
            {
                d:k 
                for k, d in enumerate(self.decisions)
            }
        )
    
    def start_state_action_space(self):
        
        # A = set of all possible car attributes
        self.car_attributes = list(it.product(
            self.point_list_ids,
            np.arange(self.battery_size+1)))
        
        self.dict_car_states = bidict(
            {a:k for k, a in enumerate(self.car_attributes)}
            )

        # B = set of trip attributes (o,d)
        self.trip_attributes = list(it.product(
            self.point_list_ids,
            self.point_list_ids)
        )

        # Every trip t atribute (t.o, t.d)
        #  is associated to an index
        self.dict_trip_states = bidict(
            {
                trip_a:k 
                for k, trip_a in enumerate(self.trip_attributes)
            }
        )
        

        # Every trip atribute (zone_o, zone_d)
        #  is associated to an index
        trip_index_attribute_dict = bidict(
            {
                trip:k for k,trip in enumerate(self.trip_attributes)
            }
        )

    def print_environment(self):
        """Print environment zones, points, and cars"""
        print("\nZones:")
        pprint(self.zones)

        print('\nLocations:')
        pprint(self.points)

        print("\nFleet:")
        pprint(self.cars)
    
    def get_fleet_status(self):
        """Number of cars per status and total battery level
        in miles.
        
        Returns:
            dict, float -- #cars per status, total battery level
        """
        status_count = defaultdict(int)
        total_battery_level = 0
        for c in self.cars:
            total_battery_level+=c.battery_level_miles
            status_count[c.status]+=1
        return status_count, total_battery_level

    def print_current_stats(self):
        count_status = defaultdict(int)
        for c in self.cars:
            print(c.status_log())
            count_status[c.status]+=1
        
        pprint(dict(count_status)) 
        
    def reset(self):
        # Rt = resource state vector at time t
        #resource_state_vector = np.zeros(len(self.car_attributes))
        
        # Dt = trip state vector at time t
        #trip_state_vector = np.zeros(len(self.trip_attributes))

        # Back to initial state
        #self.state = (resource_state_vector, trip_state_vector)

        # Return cars to initial state
        # for c in self.cars:
        #     c.reset(self.battery_size)

        # Creating random fleet starting from random points
        self.cars = [
            Car(
                point,
                self.config.battery_levels,
                battery_level_miles_max=self.config.battery_size_miles
            )
            for point in self.car_origin_points
        ]

    def step(self, action):
        #     reward = self.P[self.s][action][0][2]
        #     done = self.P[self.s][action][0][3]
        #     info = {'prob':self.P[self.s][action][0][0]}
        #     self.s = self.P[self.s][action][0][1]
        #     return (self._convert_state(self.s), reward, done, info)   
        pass

    def transition_function(self, state, action, trips):
        """Return next state and reward when action is applied to state
        
        Arguments:
            state {tuple} -- car point id and battery level
            action {string} -- assign, move empty, recharge, or hold
            trips {list} -- active trips at current time step
        
        Returns:
            tuple -- (next state, reward)
        """
        print("Transition", state, action)
        car_point_id, battery_level = state
        
        if action == 'assign':
            # - Time to reach the new point
            # - The new point it moved to
            # - the batter level remaining after reaching a new point
            new_state = state

            # Get trip in the car surroundings
            trip = get_trip(self.points, car_point_id, trips, self.config.pickup_zone_range, self.zones)
            
            # Distance until next user
            distance = self.get_distance(trip.o, trip.d)
            # Reward
            reward = self.config.trip_base_fare + self.config.trip_cost_mile*distance
            
            return new_state, reward, distance

        elif action == 'move_empty':
            # - Time to reach the new point
            # - The new point it moved to
            # - the batter level remaining after reaching a new point
            new_state = state
            return new_state, 0, 0
        
        elif action == 'recharge':
            # - Time car took for recharging (decision variable)
            # - New battery level
            new_state = state
            recharging_time = 0
            
            reward = -(
                self.config.recharge_base_fare
                + (
                    self.config.recharge_cost_mile
                    *self.config.recharge_rate
                    *recharging_time
                )
            )

            return new_state, reward, 0
        
        elif action == 'hold':
            # - Time of the availability of the car (typically next epoch)
            new_state = state
            return new_state, 0, 0
    
    def step2(self, state, time_step, cars_per_attribute, trips_per_attribute, action, trips):

        total_reward = 0
        serviced = list()
        denied = list()

        for a in action:

            
        
            car_attribute, decision, number_time_decision = a
            
            # Decision attributes
            action_code, from_point_id, to_point_id = decision
            
            # Trip attribute
            trip_attribute = (from_point_id, to_point_id)
            list_trips_in_decision = trips_per_attribute[trip_attribute]
            cars_with_attribute = cars_per_attribute[car_attribute]

            # print(f'\n## ACTION {a} ###########################')
            # pprint(list_trips_in_decision)
            # pprint(cars_with_attribute)

            

            for n, c in enumerate(cars_with_attribute):
                #c.update(time_step, time_increment=config.TIME_INCREMENT)
                
                # Only 'number_time_decision' cars will execute decision
                # determined in action 'a'
                if n >= number_time_decision:
                    break
                

                if action_code == Amod.RECHARGE_REBALANCE_DECISION:
                    if from_point_id == to_point_id:
                        # print("RECHARGING")
                        # Recharge
                        # Recharge vehicle
                        cost_recharging = cu.recharge(c)
                        
                        # Subtract cost of recharging
                        total_reward-=cost_recharging

                    else:
                        # Rebalance
                        # print("REBALANCING")
                        pass
                elif action_code == Amod.TRIP_STAY_DECISION:
                    if from_point_id == to_point_id:
                        # print("STAYING")
                        # Stay
                        pass
                    else:
                        # print("TRIP")
                        # Get a trip to apply decision
                        trip = list_trips_in_decision.pop()                      
                        
                        duration_min, total_distance, car_reward = cu.pickup2(trip, c)
                        
                        c.update_trip(
                            duration_min,
                            total_distance,
                            car_reward,
                            trip
                        )

                        serviced.append(trip)

                        total_reward+=car_reward

            # Remove cars already used to fulfill decisions
            cars_with_attribute = cars_with_attribute[number_time_decision:]

        return total_reward, serviced, list(it.chain.from_iterable(trips_per_attribute.values()))


class StepLog:
    def __init__(self, env):
        self.env = env
        self.reward_list = list()
        self.serviced_list = list()
        self.rejected_list = list()
        self.total_list = list()
        self.car_statuses = defaultdict(list)
        self.total_battery = list()
        self.n = 0

    def add_record(self, reward, serviced, rejected):
        self.n +=1
        self.reward_list.append(reward)
        self.serviced_list.append(len(serviced))
        self.rejected_list.append(len(rejected))
        total = len(serviced) + len(rejected)
        self.total_list.append(total)

        # Get number of cars per status in a time step 
        # and aggregate battery level
        dict_status, battery_level =  self.env.get_fleet_status()
        
        # Fleet aggregate battery level
        self.total_battery.append(battery_level)

        # Number of vehicles per status
        for k in Car.status_list:
            self.car_statuses[k].append(dict_status.get(k,0))
        
    def overall_log(self):

        # Get number of times recharging for each vehicle
        recharge_list = []
        for c in self.env.cars:
            recharge_list.append(c.recharge_count)

        s = sum(self.serviced_list)
        t = sum(self.total_list)
        print(f'        Service rate: {s}/{t} ({s/t:.2%})')
        print(f'Fleet recharge count: {sum(recharge_list)}')

        
    def plot_timestep_status(self):
        steps = np.arange(self.n)
        k, s = list(zip(*self.car_statuses.items()))
        plt.stackplot(steps, s, labels=k)
        # for k, s in self.car_statuses.items():
        #     plt.stackplot(np.arange(self.n), s, label=k)
        plt.legend()
        plt.show()
    
    def plot_trip_coverage_battery_level(self):
        
        max_battery_level = (
            len(self.env.cars)
            *(self.env.cars[0].battery_level_miles_max
            *self.env.config.battery_size_kwh_mile)
        )

        # Closest power of 10
        max_battery_level_10 =  10**round(np.math.log10(max_battery_level))


        list_battery_level_kwh = (
            np.array(self.total_battery)*self.env.config.battery_size_kwh_mile
        )

        steps = np.arange(self.n)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Time (15min)')
        ax1.set_ylabel('Trips')
        ax1.plot(steps, self.total_list, label='Trips Requested', color='b')
        ax1.plot(steps, self.serviced_list, label='Trips Taken', color='g')
        ax1.legend()
        ax2 = ax1.twinx()
        ax2.plot(steps,  list_battery_level_kwh, label = 'Battery Level', color='r')
        ax2.set_ylabel('Total Battery Level (KWh)')
        
        # Configure ticks x axis
        x_ticks = 6
        x_stride = 20
        max_x = np.math.ceil(self.n/x_stride)*x_stride
        xticks = np.arange(0, max_x + x_stride, x_stride)
        #print(self.n, xticks)
        plt.xticks(xticks)

        # Configure ticks y axis (battery level)
        y_ticks = 5 # apart from 0
        y_stride = max_battery_level_10/y_ticks
        yticks = np.arange(0, max_battery_level_10 + y_stride, y_stride)
        plt.yticks(yticks)

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        ax2.legend()
        plt.show()
    
    
        