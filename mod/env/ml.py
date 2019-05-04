import numpy as np

from collections import defaultdict, Counter
from operator import itemgetter
from bidict import bidict
from mod.env.amod import Amod
from mod.env.match import fcfs
import scipy

class ActionSpace(object):

    def __init__(self, get_decision):
        self.get_decision = get_decision
    
    
    def get_action(self, cars):

        actions = defaultdict(int)

        for c in cars:

            # What is the decision taken by vehicle c
            decision = self.get_decision(c)

            # TODO is it really previous???
            actions[((c.previous.id, c.previous_battery_level), decision)]+=1

        comb = tuple([a+(v,) for a,v in actions.items()])
        return comb
    
    def get_single_action_tuple(self, id):
        return self.action_id_map.inverse[id]

    def get_single_action_id(self, action):
        return self.action_id_map[action]
    
    def get_action_tuple(self, cars):

        actions = defaultdict(int)

        for c in cars:

            # What is the decision taken by vehicle c
            decision = self.get_decision(c)

            actions[(c.attribute, decision)]+=1

        comb = tuple([a+(v,) for a,v in actions.items()])
        return comb


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

#######################################################################
################ STATE HELPERS ########################################
#######################################################################

def get_dict_cars_per_attribute(cars, level=0):
        
    dict_cars_per_attribute = defaultdict(list)

    for c in cars:
        dict_cars_per_attribute[c.attribute(level)].append(c)
    
    return dict_cars_per_attribute

# def get_car_state(cars, level=0):
#     # Rt = the change in the number of cars due to information
#     # arriving between t-1 and t (e.g., cars entering/leaving 
#     # the service, or random delays in the arrival of cars moving
#     # from one point to another


#     # Get state of all free cars (e.g., [(p1,b1), (p2,b2), ...])
#     car_states = [c.attribute for c in cars if not c.busy]

#     # Rta = number of resources with attribute vector a at time t
#     resources_per_attribute = Counter(car_states)
    
#     car_state_tuple = list()
#     for a, count in resources_per_attribute.items():
#         car_state_tuple.append((a + (count,)))
    
#     car_state_tuple.sort(key=itemgetter(0,1))

#     return tuple(car_state_tuple)


def get_demand_state(trips_t, level=0):
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
        dict_trip_per_attribute[t.attribute(level)].append(t)

    trip_state_tuple = list()
    # How many trips per attribute?
    for b, trip_list in dict_trip_per_attribute.items():
        # (b, count) = (from point_id, to point_id, count)
        count = len(trip_list)
        # print('Count:', count)
        trip_state_tuple.append((b + (count,)))
    
    # Sort by from_id and then by to_id
    trip_state_tuple.sort(key=itemgetter(0,1))

    return tuple(trip_state_tuple), dict_trip_per_attribute

def get_fleet_state(cars, level=0):
    # Rt = the change in the number of cars due to information
    # arriving between t-1 and t (e.g., cars entering/leaving 
    # the service, or random delays in the arrival of cars moving
    # from one point to another

    car_per_attribute = get_dict_cars_per_attribute(cars, level=level)

    car_state_tuple = list()
    for a, list_cars in car_per_attribute.items():
        car_state_tuple.append((a + (len(list_cars),)))
    
    # Rta = number of resources with attribute vector a at time t
    #resources_per_attribute = Counter(car_states)
    car_state_tuple.sort(key=itemgetter(0,1))

    return tuple(car_state_tuple), car_per_attribute

def get_state(time_step, cars, trips, level=0):
    trip_state, trips_per_attribute = get_demand_state(trips, level)
    if not trip_state:
        return None, None, None

    car_state, cars_per_attribute = get_fleet_state(cars, level)
    if not car_state:
        return None, None, None
    
    return (time_step, car_state, trip_state), cars_per_attribute, trips_per_attribute

class Adp:

    
    def print_dimension(self, max_number_trips, total_time_steps):
        """Print problem's dimension in terms of number of states and actions.
        
        Arguments:
            max_number_trips {int} -- Maximum number of tris occurring in
                a time step. 
            total_time_steps {int} -- The duration of an episode
        """


        n_zones = self.mod.config.rows*self.mod.config.cols

        # Car attributes (zone id, battery level) (A)
        car_attributes = n_zones*self.mod.config.battery_levels

        # Outcome space - Expectation over a vector of random variables
        # The resource state vector at time t
        cars_per_attribute =  scipy.misc.comb(
            car_attributes + len(self.mod.cars) - 1,
            len(self.mod.cars)
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
            f'\n - Number of zones (|Z|): {n_zones}'
            f'\n - Car attributes (|A|): {car_attributes}'
            f'\n - Trip attributes (|B|): {trip_attributes}' 
            f'\n - Resource state vector (Rt): {physical_state_count}'
            f'\n - Trip state vector (Dt): {trips_per_attribute}'
            f'\n - Physical state (St): {physical_state_count}'
            f'\n - States in episode ({total_time_steps} steps): {physical_state_count*total_time_steps}'
        )


    def iterate(self, step, trips, epsilon = 0.1, gamma = 0.8, alpha = 0.1):

        g = 0
        """Service all trips in a given step, update the environment
        and store the iteration result in the 'step_log' object.
        
        Arguments:
            amod {Amod} -- Environment
            trips {list} -- List of trips
            step {int} -- Current step
        
        Returns:
            [type] -- [description]
        """

        # print(
        #     f'---- Time step {step:>4} '
        #     f'---- # Trips = {len(trips):>4} '
        #     f'---- # States visited:{len(amod.Q.keys())}'
        # )
        
        # Current state, return None if all cars are busy or there are no trips
        state, dict_car_attribute, dict_trip_attribute = get_state(
            step,
            self.mod.cars,
            trips
        )
        
        action = self.action_space.get_action(self.mod.cars)

        reward, list_serviced, list_rejected = fcfs(
            self.mod,
            trips,
            step
        )
        
        # Get the best action so far
        # Get actions taken from current state
        action_dict = self.table[g].get(state, None)
        
        # Update the new value
        self.table[g][state][action] = (
            (1 - alpha) * reward
            + alpha * (
                self.table[g][state].get(action, 0)
                if state in self.table[g] else 0
            )
        )

        return reward, list_serviced, list_rejected


    def __init__(self, env):
        
        self.mod = env
        
        # Each level has its own table
        self.table = [
            defaultdict(lambda: defaultdict(float))
            for g in range(self.mod.config.aggregation_levels)
        ]

        self.action_space = ActionSpace(Amod.get_decision)
        
    def value(self, state, action, level=0):
        return self.table[level][state][action]
    
    def set_value(self, state, action, value, level=0):
        self.table[level][state][action]=value