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
        
        # Defining map points with aggregation_levels
        self.points = get_point_list(
            self.config.rows,
            self.config.cols,
            levels=self.config.aggregation_levels
        )

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

    def pickup(self,trip, car):
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

        # if update:
        #     # Update car data
        #     car.update_trip(
        #         duration_min,
        #         total_distance,
        #         reward,
        #         trip
        #     )

        #     # Associate car to trip
        #     trip.picked_by = car
        # print(
        #     f'Trip total distance ({trip.o}->{trip.d}) = {total_distance:.2f}'
        #     f' miles ({distance_pickup:.2f}+{distance_trip:.2f})'
        #     f' (~{duration_min}min) - Fares: ${reward:.2f}'
        # )

        return duration_min, total_distance, reward

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

    @staticmethod
    def get_decision(car, level):
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
                car.point.id_level(level),
                car.point.id_level(level)
            )
        elif car.status == Car.ASSIGN:
            decision = (
                Amod.TRIP_STAY_DECISION,
                car.trip.o.id_level(level),
                car.trip.d.id_level(level)
            )
        elif car.status == Car.RECHARGING or car.status == Car.REBALANCE:
            decision = (
                Amod.RECHARGE_REBALANCE_DECISION,
                car.previous.id_level(level),
                car.point.id_level(level)
            )
        return decision

    
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

    def update_fleet_status(self, step):
        # Update fleet status
        for car in self.cars:
            car.update(step, time_increment=self.config.time_increment)