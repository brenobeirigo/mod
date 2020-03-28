import os
import sys
import json
from datetime import datetime, date, timedelta
from scipy.stats import gamma, norm, truncnorm
import pandas as pd
import numpy as np
import random
from collections import namedtuple
import hashlib
from copy import deepcopy
from pprint import pprint

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

# #################################################################### #
# ## FOLDERS ######################################################### #
# #################################################################### #

# Scenarios
# INSTANCE = "nyc"
INSTANCE = "rot"
FOLDER_INSTANCE = root + f"/data/input/{INSTANCE}"
FOLDER_TRAINING_TRIPS = f"{FOLDER_INSTANCE}/trips/train/"
FOLDER_TESTING_TRIPS = f"{FOLDER_INSTANCE}/trips/test/"
FOLDER_FAV_ORIGINS = f"{FOLDER_INSTANCE}/fav/"
FOLDER_TUNING = f"{FOLDER_INSTANCE}/tuning/"
FOLDER_OD_DATA = f"{FOLDER_INSTANCE}/od_data/"

# All folders
FOLDERS = [
    FOLDER_INSTANCE,
    FOLDER_TRAINING_TRIPS,
    FOLDER_TESTING_TRIPS,
    FOLDER_FAV_ORIGINS,
    FOLDER_TUNING,
    FOLDER_OD_DATA,
]

# Create all folders
for f in FOLDERS:
    if not os.path.exists(f):
        os.makedirs(f)

# #################################################################### #
# ## FILE PATHS####################################################### #
# #################################################################### #

# Spatiotemporal probability
FIST_CLASS_PROB = f"{FOLDER_INSTANCE}1st_class_prob_info.npy"

# Load all trip paths
PATHS_TRAINING_TRIPS = [
    f"{FOLDER_TRAINING_TRIPS}{t}"
    for t in os.listdir(FOLDER_TRAINING_TRIPS)
    if t.endswith(".csv")
]

# Load all test paths
PATHS_TESTING_TRIPS = [
    f"{FOLDER_TESTING_TRIPS}{t}"
    for t in os.listdir(FOLDER_TESTING_TRIPS)
    if t.endswith(".csv")
]

print(
    f"{len(PATHS_TRAINING_TRIPS)} trip files"
    f" and {len(PATHS_TESTING_TRIPS)} test files loaded."
)

# Car statuses
IDLE = 0
RECHARGING = 1
ASSIGN = 2
CRUISING = 3
REBALANCE = 4
RETURN = 5
SERVICING = 6

status_label_dict = {
    IDLE: "Parked",
    RECHARGING: "Recharging",
    ASSIGN: "With passenger",
    CRUISING: "Driving to pick up",
    REBALANCE: "Rebalancing",
    SERVICING: "Servicing passenger",
    RETURN: "Return",
}

# Output folder
FOLDER_OUTPUT = root + "/data/output/"

# Plot folders
FOLDER_SERVICE_PLOT = root + "/data/output/service_plot/"
FOLDER_FLEET_PLOT = root + "/data/output/fleet_plot/"
FOLDER_EPISODE_TRACK = root + "/data/output/track_episode/"

# Map projections for visualization
PROJECTION_MERCATOR = "MERCATOR"
PROJECTION_GPS = "GPS"
FLEET_START_LAST = "FLEET_START_LAST"
FLEET_START_SAME = "FLEET_START_SAME"
FLEET_START_RANDOM = "FLEET_START_RANDOM"

# #################################################################### #
# SCENARIOS ########################################################## #
# #################################################################### #

# Trip ODs are uniformly distributed on the map at random
SCENARIO_BALANCED = "BALANCED"

# Trip origins are concentrated to production areas and destinations
# to attraction areas
SCENARIO_UNBALANCED = "UNBALANCED"

# Trip origins are uniformly distributed while the destinations are
# fixed to one point (e.g. an access station).
SCENARIO_FIRST_MILE = "FIRST_MILE"

# Trip origins are fixed to one point (e.g., station) and destinations
# are uniformly distributed
SCENARIO_LAST_MILE = "LAST_MILE"

# Uses the real-world New York city data from 2011-02-01
SCENARIO_NYC = "NYC"

# ADP update methods
AVERAGED_UPDATE = "AVERAGED_UPDATE"
WEIGHTED_UPDATE = "WEIGHTED_UPDATE"


class Config:

    SQ_CLASS_1 = "A"
    SQ_CLASS_2 = "B"

    # This configuration refers to which test case?
    TEST_LABEL = "TEST_LABEL"
    # Determined in tuning
    TUNE_LABEL = "TUNE_LABEL"

    SPEED = "SPEED"
    FLEET_SIZE = "FLEET_SIZE"
    FLEET_START = "FLEET_START"
    CAR_SIZE_TABU = "CAR_SIZE_TABU"

    BATTERY_SIZE_DISTANCE = "BATTERY_SIZE_DISTANCE"

    BATTERY_SIZE = "BATTERY_SIZE"
    BATTERY_LEVELS = "BATTERY_LEVELS"
    BATTERY_SIZE_KWH_DISTANCE = "BATTERY_SIZE_KWH_DISTANCE"
    BATTERY_DISTANCE_LEVEL = "BATTERY_DISTANCE_LEVEL"
    MEAN_TRIP_DISTANCE = "MEAN_TRIP_DISTANCE"
    SD_TRIP_DISTANCE = "SD_TRIP_DISTANCE"
    MINIMUM_TRIP_DISTANCE = "MINIMUM_TRIP_DISTANCE"
    MAXIMUM_TRIP_DISTANCE = "MAXIMUM_TRIP_DISTANCE"
    TRIP_BASE_FARE = "TRIP_BASE_FARE"
    TRIP_DISTANCE_RATE_KM = "TRIP_DISTANCE_RATE_KM"
    TRIP_TOLERANCE_DELAY_MIN = "TRIP_TOLERANCE_DELAY_MIN"
    TRIP_MAX_PICKUP_DELAY = "TRIP_MAX_PICKUP_DELAY"
    TRIP_CLASS_PROPORTION = "TRIP_CLASS_PROPORTION"
    USE_CLASS_PROB = "USE_CLASS_PROB"

    TRIP_COST_DISTANCE = "TRIP_COST_DISTANCE"
    TOTAL_TRIPS = "TOTAL_TRIPS"
    MIN_TRIPS = "MIN_TRIPS"
    MAX_TRIPS = "MAX_TRIPS"
    PICKUP_ZONE_RANGE = "PICKUP_ZONE_RANGE"
    MATCHING_DELAY = "MATCHING_DELAY"
    REACHABLE_NEIGHBORS = "REACHABLE_NEIGHBORS"

    # In general, aggregation of attribute vectors is performed using a
    # collection of aggregation functions, G(g) : A → A(g), where A(g)
    # represents the gth level of aggregation of the attribute space A.
    AGGREGATION_LEVELS = "AGGREGATION_LEVELS"
    LEVEL_DIST_LIST = "LEVEL_LIST"
    LEVEL_TIME_LIST = "LEVEL_TIME_LIST"
    LEVEL_CONTRACT_DURATION = "LEVEL_CONTRACT_DURATION"
    LEVEL_CAR_TYPE = "LEVEL_CAR_TYPE"
    LEVEL_CAR_ORIGIN = "LEVEL_CAR_ORIGIN"
    INCUMBENT_AGGREGATION_LEVEL = "INCUMBENT_AGGREGATION_LEVEL"
    ADP_IGNORE_ZEROS = "ADP_IGNORE_ZEROS"

    ZONE_WIDTH = "ZONE_WIDTH"
    VALID_ZONES = "VALID_ZONES"
    ROWS = "ROWS"
    COLS = "COLS"
    ORIGIN_CENTERS = "ORIGIN_CENTERS"
    ORIGIN_CENTER_ZONE_SIZE = "ORIGIN_CENTER_ZONE_SIZE"
    DESTINATION_CENTERS = "DESTINATION_CENTERS"

    # Recharging
    ENABLE_RECHARGING = "ENABLE_RECHARGING"
    RECHARGE_THRESHOLD = "RECHARGE_THRESHOLD"
    RECHARGE_BASE_FARE = "RECHARGE_BASE_FARE"
    RECHARGE_COST_DISTANCE = "RECHARGE_COST_DISTANCE"
    RECHARGE_RATE = "RECHARGE_RATE"
    PARKING_RATE_MIN = "PARKING_RATE_MIN"
    COST_RECHARGE_SINGLE_INCREMENT = "COST_RECHARGE_SINGLE_INCREMENT"
    TIME_INCREMENT = "TIME_INCREMENT"
    TOTAL_TIME = "TOTAL_TIME"
    OFFSET_REPOSITIONING_MIN = "OFFSET_REPOSITIONING_MIN"
    OFFSET_TERMINATION_MIN = "OFFSET_TERMINATION_MIN"
    TIME_PERIODS = "TIME_PERIODS"

    # SAVING DATA
    USE_SHORT_PATH = "USE_SHORT_PATH"
    SAVE_TRIP_DATA = "SAVE_TRIP_DATA"
    SAVE_FLEET_DATA = "SAVE_FLEET_DATA"

    # FLEET ECONOMICS
    OPERATION_YEARS = "OPERATION_YEARS"
    OPERATED_DAYS_YEAR = "OPERATED_DAYS_YEAR"
    CAR_BASE_COST = "CAR_BASE_COST"
    MAINTENANCE_INSURANCE = "MAINTENANCE_INSURANCE"
    BATTERY_COST = "BATTERY_COST"

    # LEARNING
    STEPSIZE = "STEPSIZE"
    DISCOUNT_FACTOR = "DISCOUNT_FACTOR"
    HARMONIC_STEPSIZE = "HARMONIC_STEPSIZE"
    STEPSIZE_RULE = "STEPSIZE_RULE"
    STEPSIZE_CONSTANT = "STEPSIZE_FIXED"
    UPDATE_METHOD = "UPDATE_METHOD"  # AVERAGED, WEIGTHED

    # Network
    STEP_SECONDS = "STEP_SECONDS"  # In km/h
    N_CLOSEST_NEIGHBORS = "N_CLOSEST_NEIGHBORS"
    N_CLOSEST_NEIGHBORS_EXPLORE = "N_CLOSEST_NEIGHBORS_EXPLORE"
    NEIGHBORHOOD_LEVEL = "NEIGHBORHOOD_LEVEL"
    REBALANCE_LEVEL = "REBALANCE_LEVEL"
    PENALIZE_REBALANCE = "PENALIZE_REBALANCE"
    REBALANCE_REACH = "REBALANCE_REACH"
    REBALANCE_MULTILEVEL = "REBALANCE_MULTILEVEL"
    MATCHING_LEVELS = "MATCHING_LEVELS"
    CENTROID_LEVEL = "CENTROID_LEVEL"  # ODs are centroids

    # Model constraints
    SQ_GUARANTEE = "SQ_GUARANTEE"
    MAX_CARS_LINK = "MAX_CARS_LINK"
    TIME_MAX_CARS_LINK = "TIME_MAX_CARS_LINK"
    LINEARIZE_INTEGER_MODEL = "LINEARIZE_INTEGER_MODEL"
    USE_ARTIFICIAL_DUALS = "USE_ARTIFICIAL_DUALS"
    # Mathing methods

    # Match cars with immediate neigbors at chosen level
    MATCH_NEIGHBORS = "MATCH_NEIGHBORS"

    # Match cars within the same center at chosen level
    MATCH_CENTER = "MATCH_CENTER"

    # Match cars by distance (car can reach travel)
    MATCH_DISTANCE = "MATCH_DISTANCE"

    MATCH_METHOD = "MATCH_METHOD"
    MATCH_LEVEL = "MATCH_LEVEL"
    MATCH_MAX_NEIGHBORS = "MATCH_MAX_NEIGHBORS"
    LEVEL_RC = "LEVEL_RC"

    IDLE_ANNEALING = "IDLE_ANNEALING"

    # Method
    MYOPIC = "MYOPIC"
    POLICY_RANDOM = "POLICY_RANDOM"
    ACTIVATE_THOMPSON = "ACTIVATE_THOMPSON"
    SAVE_PROGRESS = "SAVE_PROGRESS"
    METHOD_ADP_TRAIN = "adp/train"
    METHOD_ADP_TEST = "adp/test"
    METHOD_RANDOM = "random"
    METHOD_REACTIVE = "reactive"
    METHOD_MYOPIC = "myopic"
    METHOD = "METHOD"
    ITERATIONS = "ITERATIONS"

    # DEMAND
    DEMAND_CENTER_LEVEL = "DEMAND_CENTER_LEVEL"
    DEMAND_TOTAL_HOURS = "DEMAND_TOTAL_HOURS"
    DEMAND_EARLIEST_HOUR = "DEMAND_EARLIEST_HOUR"
    DEMAND_RESIZE_FACTOR = "DEMAND_RESIZE_FACTOR"
    DEMAND_MAX_STEPS = "DEMAND_MAX_STEPS"
    EARLIEST_STEP_MIN = "EARLIEST_STEP_MIN"
    DEMAND_SCENARIO = "DEMAND_SCENARIO"
    TIME_INCREMENT_TIMEDELTA = "TIME_INCREMENT_TIMEDELTA"
    DEMAND_EARLIEST_DATETIME = "DEMAND_EARLIEST_DATETIME"
    DEMAND_SAMPLING = "DEMAND_SAMPLING"
    DEMAND_CLASSED = "DEMAND_CLASSED"
    ALLOW_USER_BACKLOGGING = "ALLOW_USER_BACKLOGGING"
    MAX_IDLE_STEP_COUNT = "MAX_IDLE_STEP_COUNT"
    TRIP_REJECTION_PENALTY = "TRIP_REJECTION_PENALTY"
    UNIVERSAL_SERVICE = "UNIVERSAL_SERVICE"

    # NETWORK INFO
    NAME = "NAME"
    REGION = "REGION"
    NODE_COUNT = "NODE_COUNT"
    EDGE_COUNT = "EDGE_COUNT"
    CENTER_COUNT = "CENTER_COUNT"

    # HIRING
    PROFIT_MARGIN = "PROFIT_MARGIN"
    CONTRACT_DURATION_LEVEL = "CONTRACT_DURATION_LEVEL"
    CONGESTION_PRICE = "CONGESTION_PRICE"
    MEAN_CONTRACT_DURATION = "MEAN_CONTRACT_DURATION"
    MIN_CONTRACT_DURATION = "MIN_CONTRACT_DURATION"
    MAX_CONTRACT_DURATION = "MAX_CONTRACT_DURATION"
    FAV_FLEET_SIZE = "FAV_FLEET_SIZE"
    DEPOT_SHARE = "DEPOT_SHARE"
    FAV_DEPOT_LEVEL = "FAV_DEPOT_LEVEL"
    SEPARATE_FLEETS = "SEPARATE_FLEETS"
    FAV_AVAILABILITY_FEATURES = "FAV_AVAILABILITY_FEATURES"
    FAV_EARLIEST_FEATURES = "FAV_EARLIEST_FEATURES"
    # Max. contract duration = MAX_TIME_PERIODS

    # Max. number of rebalancing targets
    MAX_TARGETS = "MAX_TARGETS"

    def __init__(self, config):

        self.current_iteration = 0
        self.current_step = 0

        self.config = config

    # ################################################################ #
    # ## Area ######################################################## #
    # ################################################################ #

    @property
    def origin_centers(self):
        return self.config[Config.ORIGIN_CENTERS]

    @property
    def method(self):
        return self.config[Config.METHOD]

    @property
    def main_path(self):
        return f"{FOLDER_OUTPUT}{self.label}/"

    @property
    def output_path(self):
        return f"{FOLDER_OUTPUT}{self.label}/{self.method}/"

    @property
    def sampled_tripdata_path(self):
        samples_path = (
            f"{FOLDER_OUTPUT}{self.label}/{self.method}/trip_samples_data/"
        )
        # Creates directories
        if not os.path.exists(samples_path):
            os.makedirs(samples_path)
        return samples_path

    @property
    def fleet_data_path(self):
        fleet_data_path = (
            f"{FOLDER_OUTPUT}{self.label}/{self.method}/fleet_data/"
        )
        # Creates directories
        if not os.path.exists(fleet_data_path):
            os.makedirs(fleet_data_path)
        return fleet_data_path

    @property
    def destination_centers(self):
        return self.config[Config.DESTINATION_CENTERS]

    @property
    def origin_center_zone_size(self):
        return self.config[Config.ORIGIN_CENTER_ZONE_SIZE]

    @property
    def demand_center_level(self):
        # E.g., levels 1, 2, 3 = 60, 120, 180
        # if level_origins = 3
        return self.config[Config.DEMAND_CENTER_LEVEL]

    ####################################################################
    ### Battery info ###################################################
    ####################################################################

    @property
    def enable_recharging(self):
        """Battery charging will be considered in the ADP"""
        return self.config[Config.ENABLE_RECHARGING]

    @property
    def recharge_base_fare(self):
        """Trip base fare in dollars"""
        return self.config["RECHARGE_BASE_FARE"]

    @property
    def min_battery_level(self):
        """Trip base fare in dollars"""
        return (
            self.config["RECHARGE_THRESHOLD"] * self.config["BATTERY_LEVELS"]
        )

    @property
    def recharge_cost_distance(self):
        """Trip base fare in dollars"""
        return self.config["RECHARGE_COST_DISTANCE"]

    @property
    def recharge_rate(self):
        """Trip base fare in dollars"""
        return self.config["RECHARGE_RATE"]

    @property
    def cost_recharge_single_increment(self):
        """Trip base fare in dollars"""
        return self.config[Config.COST_RECHARGE_SINGLE_INCREMENT]

    @property
    def recharge_threshold(self):
        """Minimum battery charge percentage (float in [0,1] interval) """
        return self.config["RECHARGE_THRESHOLD"]

    def calculate_cost_recharge(self, recharging_time_min):
        recharging_time_h = recharging_time_min / 60.0
        return self.config["RECHARGE_BASE_FARE"] + (
            self.config["RECHARGE_COST_DISTANCE"]
            * self.config["RECHARGE_RATE"]
            * recharging_time_h
        )

    def get_parking_cost(self):
        """Return the cost of travelling 'distance' meters"""
        return (
            self.config[Config.RECHARGE_COST_DISTANCE]
            * self.time_increment
            * self.speed
            / 60
        )

    @property
    def parking_cost_step(self):
        return self.config[Config.PARKING_RATE_MIN] * self.time_increment

    def get_travel_cost(self, distance_km):
        """Return the cost of travelling 'distance' meters"""
        return self.config[Config.RECHARGE_COST_DISTANCE] * distance_km

    def calculate_dist_recharge(self, recharging_time_min):
        recharging_time_h = recharging_time_min / 60.0
        return self.config["RECHARGE_RATE"] * recharging_time_h

    def get_full_recharging_time(self, distance):
        """Get recharge time in relation to recharge distance
        according to recharge rate in miles/hour

        Arguments:
            distance {float} -- miles

        Returns:
            int, int --recharge time in minutes and time steps
        """

        hours_recharging = distance / self.config["RECHARGE_RATE"]

        minutes_recharging = hours_recharging * 60

        time_steps_recharging = (
            minutes_recharging / self.config["TIME_INCREMENT"]
        )

        # print(
        #     f'RECHARGING(miles:{distance:>3.2f}'
        #     f' - h:{hours_recharging:>3.2f}'
        #     f' - m:{minutes_recharging:>3.2f})'
        #     f' - steps:{time_steps_recharging:>3.2f})'
        # )

        return minutes_recharging, int(round(time_steps_recharging))

    ####################################################################
    # Battery ##########################################################
    ####################################################################

    @property
    def battery_size_distances(self):
        """Battery size in number of miles """
        return self.config["BATTERY_SIZE_DISTANCE"]

    @property
    def battery_levels(self):
        """Number of discrete levels"""
        return self.config["BATTERY_LEVELS"]

    @property
    def battery_distance_level(self):
        """Number of discrete levels"""
        return self.config[Config.BATTERY_DISTANCE_LEVEL]

    @property
    def battery_size_kwh_distance(self):
        """Maximum battery size in miles"""
        return self.config["BATTERY_SIZE_KWH_DISTANCE"]

    @property
    def recharge_time_single_level(self):
        return self.config["RECHARGE_TIME_SINGLE_LEVEL"]

    ####################################################################
    # Trip #############################################################
    ####################################################################

    @property
    def trip_base_fare(self):
        """Trip base fare in dollars"""
        return self.config["TRIP_BASE_FARE"]

    @property
    def trip_max_pickup_delay(self):
        return self.config[Config.TRIP_MAX_PICKUP_DELAY]

    @property
    def trip_distance_rate_km(self):
        return self.config[Config.TRIP_DISTANCE_RATE_KM]

    @property
    def trip_tolerance_delay(self):
        return self.config[Config.TRIP_TOLERANCE_DELAY_MIN]

    @property
    def trip_class_proportion(self):
        return self.config[Config.TRIP_CLASS_PROPORTION]

    @property
    def use_class_prob(self):
        """Load 1st class probability from FIST_CLASS_PROB"""
        return self.config[Config.USE_CLASS_PROB]

    @property
    def trip_cost_fare(self):
        """Trip cost per mile in dollars"""
        return self.config["TRIP_COST_DISTANCE"]

    @property
    def trip_rejection_penalty(self):
        return self.config[Config.TRIP_REJECTION_PENALTY]

    @property
    def pickup_zone_range(self):
        """Duration of the time steps in (min)"""
        return self.config["PICKUP_ZONE_RANGE"]

    @property
    def allow_user_backlogging(self):
        return self.config["ALLOW_USER_BACKLOGGING"]

    @property
    def matching_delay(self):
        """Matching delay in minutes
        """
        return self.config["MATCHING_DELAY"]

    @property
    def adp_ignore_zeros(self):
        """Method can ignore/use duals which are zero.
        """
        return self.config["ADP_IGNORE_ZEROS"]

    @property
    def max_idle_step_count(self):
        return self.config["MAX_IDLE_STEP_COUNT"]

    @property
    def consider_rebalance(self):
        if self.myopic or self.policy_reactive:
            return False
        return True

    @property
    def myopic(self):
        return self.config[Config.METHOD] == Config.METHOD_MYOPIC

    @property
    def train(self):
        return self.config[Config.METHOD] == Config.METHOD_ADP_TRAIN

    @property
    def test(self):
        return self.config[Config.METHOD] == Config.METHOD_ADP_TEST

    @property
    def save_progress(self):
        if self.config[Config.METHOD] == Config.METHOD_ADP_TRAIN:
            return self.config[Config.SAVE_PROGRESS]
        else:
            return None

    @property
    def policy_random(self):
        return self.config[Config.METHOD] == Config.METHOD_RANDOM

    @property
    def policy_reactive(self):
        return self.config[Config.METHOD] == Config.METHOD_REACTIVE

    @property
    def time_increment(self):
        """Duration of the time steps in (min)"""
        return self.config["TIME_INCREMENT"]

    @property
    def time_increment_timedelta(self):
        return self.config[Config.TIME_INCREMENT_TIMEDELTA]

    @property
    def demand_earliest_datetime(self):
        return self.config[Config.DEMAND_EARLIEST_DATETIME]

    @property
    def speed(self):
        """Speed in mph"""
        return self.config["SPEED"]

    @property
    def zone_widht(self):
        """Zone width in miles"""
        return self.config["ZONE_WIDTH"]

    @property
    def time_steps(self):
        """Time steps in minutes"""
        return self.config["TIME_PERIODS"]

    @property
    def time_steps_until_termination(self):
        """Time steps in minutes"""
        return self.config["TIME_PERIODS_TERMINATION"]

    @property
    def rows(self):
        """Number of rows in zone"""
        return self.config["ROWS"]

    @property
    def cols(self):
        """Number of colums in zone"""
        return self.config["COLS"]

    @property
    def fleet_size(self):
        """Number of cars"""
        return self.config["FLEET_SIZE"]

    @property
    def aggregation_levels(self):
        """Number of aggregation levels"""
        return self.config["AGGREGATION_LEVELS"]

    @property
    def n_aggregation_levels(self):
        """Number of aggregation levels"""
        return len(self.config["AGGREGATION_LEVELS"])

    @property
    def incumbent_aggregation_level(self):
        """Trip base fare in dollars"""
        return self.config[Config.INCUMBENT_AGGREGATION_LEVEL]

    @property
    def stepsize(self):
        """Trip base fare in dollars"""
        return self.config[Config.STEPSIZE]

    ####################################################################
    ### Demand #########################################################
    ####################################################################

    @property
    def universal_service(self):
        # True if all users must be picked up
        return self.config[Config.UNIVERSAL_SERVICE]

    @property
    def demand_scenario(self):
        """Minimum number of trips (15min) """
        return self.config[Config.DEMAND_SCENARIO]

    @property
    def min_trips(self):
        """Minimum number of trips (15min) """
        return self.config["MIN_TRIPS"]

    @property
    def max_trips(self):
        """Maximum number of trips (15min)"""
        return self.config["MAX_TRIPS"]

    def get_steps_from_m(self, m):
        return m / self.time_increment

    def get_steps_from_h(self, hour):
        return hour * 60 / self.time_increment

    def get_step(self, hour):
        return int(
            self.offset_repositioning_steps
            + (hour - self.demand_earliest_hour) * 60 / self.time_increment
        )

    @property
    def offset_repositioning_steps(self):
        """Number of time steps with no trips before 
        demand (for reposition)"""
        return int(
            self.config["OFFSET_REPOSITIONING_MIN"]
            / self.config["TIME_INCREMENT"]
        )

    @property
    def reposition_h(self):
        return self.config[Config.OFFSET_REPOSITIONING_MIN] / 60.0

    @property
    def offset_termination_steps(self):
        """Number of time steps with no trips after demand (so
        that all passengers can be delivered)"""
        return int(
            self.config["OFFSET_TERMINATION_MIN"]
            / self.config["TIME_INCREMENT"]
        )

    def resize_zones(self, factor):
        # Each zone has width = 0.5 miles
        self.config["ZONE_WIDTH"] = int(self.config["ZONE_WIDTH"] * factor)
        self.config["ROWS"] = int(self.config["ROWS"] * factor)
        self.config["COLS"] = int(self.config["COLS"] * factor)
        self.config["VALID_ZONES"] = int(
            self.config["VALID_ZONES"] * (factor * factor)
        )

    def __str__(self):
        return self.config.__str__()

    def calculate_fare(self, distance_trip, sq_class=None):
        base = self.config[Config.TRIP_BASE_FARE][sq_class]
        distance_fare = self.config[Config.TRIP_COST_DISTANCE] * distance_trip
        total = base + distance_fare
        # print(f'{base:6.2f} + {distance_fare:6.2f} = {total:6.2f}')
        return total

    def get_path_od_fares(self, extension="npy"):
        """Path of saved fares per sq_class, o, d"""
        base_fares = "_".join(
            [
                f"{sq}_{base:.2f}"
                for sq, base in self.config[Config.TRIP_BASE_FARE].items()
            ]
        )
        return (
            FOLDER_OD_DATA
            + f"od_base_{base_fares}_rate_{self.config[Config.TRIP_COST_DISTANCE]:.2f}.{extension}"
        )

    @property
    def sl_config_label(self):
        def proportion(sq):
            # Proportion comes from prbability file
            return (
                "P"
                if self.use_class_prob
                else f"{self.config[Config.TRIP_CLASS_PROPORTION][sq]:.2f}"
            )

        """Path of saved fares per sq_class, o, d"""
        sl_config_label = "_".join(
            [
                (
                    f"{sq}_{base:.2f}_"
                    f"{self.config[Config.TRIP_MAX_PICKUP_DELAY][sq]:.2f}_"
                    f"{self.config[Config.TRIP_TOLERANCE_DELAY_MIN][sq]:.2f}_"
                    f"{self.config[Config.TRIP_REJECTION_PENALTY][sq]:.2f}_"
                    f"{proportion(sq)}"
                )
                for sq, base in self.config[Config.TRIP_BASE_FARE].items()
            ]
        )

        return sl_config_label

    @property
    def sl_config_dict(self):
        sl_config_dict = {}
        for sq, base in self.config[Config.TRIP_BASE_FARE].items():
            sl_config_dict[f"{sq}_trip_base_fare"] = base
            sl_config_dict[f"{sq}_trip_distance_rate_km"] = self.config[
                Config.TRIP_DISTANCE_RATE_KM
            ][sq]
            sl_config_dict[f"{sq}_trip_max_pickup_delay"] = self.config[
                Config.TRIP_MAX_PICKUP_DELAY
            ][sq]
            sl_config_dict[f"{sq}_trip_tolerance_delay_min"] = self.config[
                Config.TRIP_TOLERANCE_DELAY_MIN
            ][sq]
            sl_config_dict[f"{sq}_trip_rejection_penalty"] = self.config[
                Config.TRIP_REJECTION_PENALTY
            ][sq]
            sl_config_dict[f"{sq}_trip_class_proportion"] = self.config[
                Config.TRIP_CLASS_PROPORTION
            ][sq]

        return sl_config_dict

    @property
    def sl_label(self):
        paper_label = dict()
        paper_label["A"] = "1"
        paper_label["B"] = "2"
        sl_config_label = "_".join(
            [
                (
                    f"{paper_label[sq]}"
                    " ("
                    f"{self.config[Config.TRIP_MAX_PICKUP_DELAY][sq]:.0f}"
                    f"{(f' + {self.config[Config.TRIP_TOLERANCE_DELAY_MIN][sq]:.0f}' if self.config[Config.TRIP_TOLERANCE_DELAY_MIN][sq]>0 else '')}"
                    ")"
                    f"{(' [P]' if self.config[Config.TRIP_REJECTION_PENALTY][sq] > 0 else '')}"
                )
                for sq, base in self.config[Config.TRIP_BASE_FARE].items()
                if self.config[Config.TRIP_CLASS_PROPORTION][sq] > 0
            ]
        )

        return sl_config_label

    def get_path_od_penalties(self, extension="npy"):
        """Path of saved fares per sq_class, o, d"""
        sl_config_label = "_".join(
            [
                f"{sq}_{base:.2f}_{self.config[Config.TRIP_MAX_PICKUP_DELAY][sq]:.2f}_{self.config[Config.TRIP_TOLERANCE_DELAY_MIN][sq]:.2f}"
                for sq, base, in self.config[Config.TRIP_BASE_FARE].items()
            ]
        )
        return (
            FOLDER_OD_DATA
            + f"od_penalties_{sl_config_label}_rate_{self.config[Config.TRIP_COST_DISTANCE]:.2f}.{extension}"
        )

    def get_path_od_costs(self, extension="npy"):
        """Path of saved costs per o, d"""
        return (
            FOLDER_OD_DATA
            + f"od_costs_km_{self.config[Config.RECHARGE_COST_DISTANCE]:.2f}.{extension}"
        )

    def update(self, dict_update_base):

        # print("Update")

        # pprint(dict_update_base)

        # Copy dictionary before updating element types
        dict_update = deepcopy(dict_update_base)
        # Guarantee elements are tuples
        if Config.REBALANCE_LEVEL in dict_update:
            dict_update[Config.REBALANCE_LEVEL] = tuple(
                dict_update[Config.REBALANCE_LEVEL]
            )

        if Config.N_CLOSEST_NEIGHBORS in dict_update:
            dict_update[Config.N_CLOSEST_NEIGHBORS] = tuple(
                dict_update[Config.N_CLOSEST_NEIGHBORS]
            )

        if Config.N_CLOSEST_NEIGHBORS_EXPLORE in dict_update:
            dict_update[Config.N_CLOSEST_NEIGHBORS_EXPLORE] = tuple(
                dict_update[Config.N_CLOSEST_NEIGHBORS_EXPLORE]
            )

        if Config.AGGREGATION_LEVELS in dict_update:
            dict_update[Config.AGGREGATION_LEVELS] = tuple(
                dict_update[Config.AGGREGATION_LEVELS]
            )

        # TODO check data structure
        try:

            if Config.TRIP_MAX_PICKUP_DELAY in dict_update:
                dict_update[Config.TRIP_MAX_PICKUP_DELAY] = {
                    kv[0]: kv[1]
                    for kv in dict_update[Config.TRIP_MAX_PICKUP_DELAY]
                }
            if Config.TRIP_DISTANCE_RATE_KM in dict_update:
                dict_update[Config.TRIP_DISTANCE_RATE_KM] = {
                    kv[0]: kv[1]
                    for kv in dict_update[Config.TRIP_DISTANCE_RATE_KM]
                }
            if Config.TRIP_TOLERANCE_DELAY_MIN in dict_update:
                dict_update[Config.TRIP_TOLERANCE_DELAY_MIN] = {
                    kv[0]: kv[1]
                    for kv in dict_update[Config.TRIP_TOLERANCE_DELAY_MIN]
                }
            if Config.TRIP_CLASS_PROPORTION in dict_update:
                dict_update[Config.TRIP_CLASS_PROPORTION] = {
                    kv[0]: kv[1]
                    for kv in dict_update[Config.TRIP_CLASS_PROPORTION]
                }
            if Config.TRIP_REJECTION_PENALTY in dict_update:
                dict_update[Config.TRIP_REJECTION_PENALTY] = {
                    kv[0]: kv[1]
                    for kv in dict_update[Config.TRIP_REJECTION_PENALTY]
                }
            if Config.TRIP_BASE_FARE in dict_update:
                dict_update[Config.TRIP_BASE_FARE] = {
                    kv[0]: kv[1] for kv in dict_update[Config.TRIP_BASE_FARE]
                }
        except:
            pass

        self.config.update(dict_update)

        self.config["BATTERY_SIZE_KWH_DISTANCE"] = (
            self.config["BATTERY_SIZE"] / self.config["BATTERY_SIZE_DISTANCE"]
        )

        self.config["BATTERY_SIZE_KWH_DISTANCE"] = (
            self.config["BATTERY_SIZE"] / self.config["BATTERY_SIZE_DISTANCE"]
        )

        # # Total number of time periods
        # self.config["TIME_PERIODS"] = int(
        #     self.config["OFFSET_REPOSITIONING_MIN"]
        #     + self.config["TOTAL_TIME"] * 60 / self.config["TIME_INCREMENT"]
        #     + self.config["OFFSET_TERMINATION_MIN"]
        # )

        #       Total number of time periods
        self.config["TIME_PERIODS"] = int(
            (
                self.config["OFFSET_REPOSITIONING_MIN"]
                + self.config[Config.DEMAND_TOTAL_HOURS] * 60
                + self.config["OFFSET_TERMINATION_MIN"]
            )
            / self.config["TIME_INCREMENT"]
        )

        self.config["TIME_PERIODS_TERMINATION"] = int(
            (
                self.config["OFFSET_REPOSITIONING_MIN"]
                + self.config[Config.DEMAND_TOTAL_HOURS] * 60
            )
            / self.config["TIME_INCREMENT"]
        )

        self.config[Config.BATTERY_DISTANCE_LEVEL] = (
            self.config[Config.BATTERY_SIZE_DISTANCE]
            / self.config[Config.BATTERY_LEVELS]
        )

        self.config["BATTERY_SIZE_KWH_DISTANCE"] = (
            self.config["BATTERY_SIZE"] / self.config["BATTERY_SIZE_DISTANCE"]
        )

        self.config["BATTERY_SIZE_DISTANCE_LEVEL"] = (
            self.config["BATTERY_SIZE_DISTANCE"]
            / self.config["BATTERY_LEVELS"]
        )

        self.config[
            Config.COST_RECHARGE_SINGLE_INCREMENT
        ] = self.calculate_cost_recharge(self.time_increment)

        self.config[Config.DEMAND_MAX_STEPS] = int(
            self.config[Config.DEMAND_TOTAL_HOURS] * 60 / self.time_increment
        )
        self.config[Config.EARLIEST_STEP_MIN] = int(
            self.config[Config.DEMAND_EARLIEST_HOUR] * 60 / self.time_increment
        )

        self.config[Config.TIME_INCREMENT_TIMEDELTA] = timedelta(
            minutes=self.config[Config.TIME_INCREMENT]
        )

        self.config[Config.DEMAND_EARLIEST_DATETIME] = (
            datetime.strptime("2011-02-01 00:00", "%Y-%m-%d %H:%M")
            + timedelta(hours=self.config[Config.DEMAND_EARLIEST_HOUR])
            - timedelta(minutes=self.config[Config.OFFSET_REPOSITIONING_MIN])
        )

        # Convert levels to tuples to facilitate pickle
        self.config[Config.AGGREGATION_LEVELS] = [
            tuple(a) for a in self.config[Config.AGGREGATION_LEVELS]
        ]

    @property
    def exp_settings(self):
        label = self.label
        return FOLDER_OUTPUT + label + "/exp_settings.json"

    @property
    def short_path(self):
        return self.config[Config.USE_SHORT_PATH]

    @property
    def save_trip_data(self):
        return self.config[Config.SAVE_TRIP_DATA]

    @property
    def save_fleet_data(self):
        return self.config[Config.SAVE_FLEET_DATA]

    @property
    def label_md5(self):
        return hashlib.md5(self.label.encode()).hexdigest()

    @property
    def label(self, name=""):
        # Implemented by childreen
        pass

    def save(self, file_path=None):
        label = self.label
        self.config["label"] = self.label
        self.config["label_md5"] = self.label_md5

        if self.short_path:
            label = self.label_md5

        folder = FOLDER_OUTPUT + str(label)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        def convert_times(t):
            if isinstance(t, (date, datetime)):
                return t.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(t, timedelta):
                return t.seconds

        if not file_path:
            file_path = self.exp_settings

        with open(file_path, "w") as f:
            json.dump(self.config, f, indent=4, default=convert_times)

    def log_path(self, iteration=""):
        return self.folder_adp_log + f"{iteration:04}.log"

    @property
    def iteration_step_seed(self):
        seed = self.current_iteration * self.time_steps + self.current_step
        print(seed, self.current_iteration, self.time_steps, self.current_step)
        return seed

    @property
    def iterations(self):
        return self.config[Config.ITERATIONS]

    @property
    def step_seconds(self):
        """Speed in kmh"""
        return self.config["STEP_SECONDS"]

    ## Demand ######################################################## #
    @property
    def demand_total_hours(self):
        return self.config[Config.DEMAND_TOTAL_HOURS]

    @property
    def demand_sampling(self):
        return self.config[Config.DEMAND_SAMPLING]

    @property
    def demand_is_classed(self):
        return self.config[Config.DEMAND_CLASSED]

    @property
    def demand_earliest_hour(self):
        return self.config[Config.DEMAND_EARLIEST_HOUR]

    @property
    def offset_termination_min(self):
        return self.config[Config.OFFSET_TERMINATION_MIN]

    @property
    def offset_termination_hour(self):
        return self.config[Config.OFFSET_TERMINATION_MIN] / 60

    @property
    def offset_repositioning_min(self):
        return self.config[Config.OFFSET_REPOSITIONING_MIN]

    @property
    def latest_hour(self):
        return (
            self.demand_earliest_hour
            + self.demand_total_hours
            + self.offset_termination_hour
        )

    @property
    def idle_annealing(self):
        return self.config[Config.IDLE_ANNEALING]

    @property
    def demand_resize_factor(self):
        return self.config[Config.DEMAND_RESIZE_FACTOR]

    @property
    def demand_max_steps(self):
        return self.config[Config.DEMAND_MAX_STEPS]

    @property
    def demand_earliest_step_min(self):
        return self.config[Config.EARLIEST_STEP_MIN]

    @property
    def car_size_tabu(self):
        return self.config[Config.CAR_SIZE_TABU]

    def get_time(self, steps, format="%H:%M"):
        """Return time corresponding to the steps elapsed since the
        the first time step"""
        t = (
            self.demand_earliest_datetime
            + steps * self.time_increment_timedelta
        )
        return t.strftime(format)


class ConfigStandard(Config):
    def __init__(self, config=None):

        if not config:
            config = dict()

        super().__init__(config)

        self.config = dict()

        self.config[Config.ITERATIONS] = 500

        ################################################################
        # Car ##########################################################
        ################################################################

        # Speed cars (mph) - 20MPH
        self.config["SPEED"] = 17

        # Total fleet
        self.config["FLEET_SIZE"] = 1500

        self.config[Config.FLEET_START] = FLEET_START_LAST

        ################################################################
        # Battery ######################################################
        ################################################################

        self.config["BATTERY_SIZE_DISTANCE"] = 200  # miles
        self.config["BATTERY_SIZE"] = 66  # kWh
        self.config["BATTERY_LEVELS"] = 20  # levels

        # How many KWh per mile?
        self.config["BATTERY_SIZE_KWH_DISTANCE"] = (
            self.config["BATTERY_SIZE"] / self.config["BATTERY_SIZE_DISTANCE"]
        )

        self.config["BATTERY_SIZE_DISTANCE_LEVEL"] = (
            self.config["BATTERY_SIZE_DISTANCE"]
            / self.config["BATTERY_LEVELS"]
        )

        # How many miles each level has?
        self.config[Config.BATTERY_DISTANCE_LEVEL] = (
            self.config[Config.BATTERY_SIZE_DISTANCE]
            / self.config[Config.BATTERY_LEVELS]
        )

        ################################################################
        # Time  ########################################################
        ################################################################

        # Lenght of time incremnts (min) - default is 15min
        self.config[Config.TIME_INCREMENT] = 15
        self.config[Config.TIME_INCREMENT_TIMEDELTA] = timedelta(
            minutes=self.config[Config.TIME_INCREMENT]
        )

        self.config[Config.DEMAND_EARLIEST_DATETIME] = datetime.strptime(
            "2011-11-02 00:00", "%Y-%m-%d %H:%M"
        )

        # Total horizon (h)
        self.config["TOTAL_TIME"] = 24

        # Offset at the beginning to reposition vehicles
        self.config["OFFSET_REPOSITIONING_MIN"] = 3

        # Offset at the end to guarantee trips terminate
        self.config["OFFSET_TERMINATION_MIN"] = 11

        # Total number of time periods
        self.config["TIME_PERIODS"] = int(
            (
                self.config["OFFSET_REPOSITIONING_MIN"]
                + self.config["TOTAL_TIME"] * 60
                + self.config["OFFSET_TERMINATION_MIN"]
            )
            / self.config["TIME_INCREMENT"]
        )

        # Total number of time periods
        self.config["TIME_PERIODS_TERMINATION"] = int(
            (
                self.config["OFFSET_REPOSITIONING_MIN"]
                + self.config["TOTAL_TIME"] * 60
            )
            / self.config["TIME_INCREMENT"]
        )

        # Step in seconds
        self.config[Config.STEP_SECONDS] = 60

        ################################################################
        # Map settings #################################################
        ################################################################

        # How many surrounding zones cars check for new costumers
        self.config["PICKUP_ZONE_RANGE"] = 2

        # How many aggregation levels (level 0, i.e., no aggregation)
        # included
        self.config["AGGREGATION_LEVELS"] = [0, 60, 120, 300]

        # Attributes are based on a single aggregation level
        self.config[Config.INCUMBENT_AGGREGATION_LEVEL] = 2

        # Each zone has width = 0.5 miles
        self.config["ZONE_WIDTH"] = 0.5

        # The New Jersey is divided into 201 bt 304 rectangular
        # zones of width 0.5 miles with 21634 valid zones
        self.config["VALID_ZONES"] = 21634
        self.config["ROWS"] = 201
        self.config["COLS"] = 304

        # Origin centers and number of surrounding layers
        self.config[Config.ORIGIN_CENTERS] = 4
        self.config[Config.DESTINATION_CENTERS] = 4
        self.config[Config.ORIGIN_CENTER_ZONE_SIZE] = 3

        self.config["RECHARGE_THRESHOLD"] = 0.1  # 10%
        self.config["RECHARGE_BASE_FARE"] = 1  # dollar
        self.config[Config.RECHARGE_COST_DISTANCE] = 0.1  # dollar
        # self.config[Config.PARKING_RATE_MIN] = 1.50/60
        self.config[Config.PARKING_RATE_MIN] = 0  # = rebalancing 1 min
        self.config["RECHARGE_RATE"] = 300  # miles/hour
        self.config[
            Config.COST_RECHARGE_SINGLE_INCREMENT
        ] = self.calculate_cost_recharge(self.time_increment)

        # How much time does it take (min) to recharge one single level?
        self.config["RECHARGE_TIME_SINGLE_LEVEL"] = int(
            60
            * self.config["BATTERY_SIZE_DISTANCE_LEVEL"]
            / self.config["RECHARGE_RATE"]
        )

        ################################################################
        # Fleet economics ##############################################
        ################################################################
        self.config[Config.OPERATION_YEARS] = 4
        self.config[Config.OPERATED_DAYS_YEAR] = 340
        self.config[Config.CAR_BASE_COST] = 40000  # Dollars

        # The cost starts with $240/kWh for the first 16.67 kWhs
        # (which corresponds to a 50 miles range) and then cost
        # increases by 20% for the next 16.67 kWhs.

        # Let bsize = 16.67 * i for i = {1, 2, . . . , 10}, then the
        # battery cost is:

        # c^bat(b^size) = $240*(1 + 0.2*(i−1))*(16.67*i).
        self.config[Config.BATTERY_COST] = 240

        ################################################################
        # Demand characteristics #######################################
        ################################################################

        self.config["MEAN_TRIP_DISTANCE"] = 24.8  # miles
        self.config["SD_TRIP_DISTANCE"] = 7  # TODO it was guessed
        self.config["MINIMUM_TRIP_DISTANCE"] = 2  # 5th percentile is 6
        self.config["MAXIMUM_TRIP_DISTANCE"] = 65  # 95th percentile is 57.5
        # Simulation parameters
        self.config["TRIP_BASE_FARE"] = 2.4  # dollar
        # TODO can vary according to:
        # - Where trip originates
        # - time of the day
        # - surge pricing
        self.config["TRIP_COST_DISTANCE"] = 1  # dollar

        # Total number of trips (min, max) = (40, 640) in period
        self.config["TOTAL_TRIPS"] = 32874
        self.config["MIN_TRIPS"] = 40
        self.config["MAX_TRIPS"] = 640

        # DEMAND DATA ##################################################
        # Data correspond to 1 day NY demand
        self.config[Config.DEMAND_TOTAL_HOURS] = 24
        self.config[Config.DEMAND_EARLIEST_HOUR] = 0
        self.config[Config.DEMAND_RESIZE_FACTOR] = 1
        self.config[Config.DEMAND_MAX_STEPS] = int(
            self.config[Config.DEMAND_TOTAL_HOURS] * 60 / self.time_increment
        )
        self.config[Config.EARLIEST_STEP_MIN] = int(
            self.config[Config.DEMAND_EARLIEST_HOUR] * 60 / self.time_increment
        )
        self.config[Config.DEMAND_SCENARIO] = SCENARIO_UNBALANCED
        self.config[Config.DEMAND_SAMPLING] = True
        self.config[Config.DEMAND_CLASSED] = True

    @property
    def label(self):

        return (
            f"{self.config[Config.ROWS]:04}_"
            f"{self.config[Config.COLS]:04}_"
            f"{self.config[Config.PICKUP_ZONE_RANGE]:02}_"
            f"{self.config[Config.AGGREGATION_LEVELS]}_"
            f"{self.config[Config.FLEET_SIZE]:04}_"
            f"{self.config[Config.BATTERY_LEVELS]:04}_"
            f"{self.config[Config.INCUMBENT_AGGREGATION_LEVEL]:01}_"
            f"{self.config[Config.TIME_INCREMENT]:02}_"
            f"{self.config[Config.STEP_SECONDS]:04}"
        )


class ConfigNetwork(ConfigStandard):
    def __init__(self, config=None):

        self.current_iteration = 0
        self.current_step = 0

        if not config:
            config = dict()

        super().__init__(config)
        # Colors
        # https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
        self.color_fleet_status = {
            IDLE: "#24aafe",
            ASSIGN: "#53bc53",
            SERVICING: "#53bc53",
            REBALANCE: "firebrick",
            RETURN: "gray",
            RECHARGING: "#e55215",
            CRUISING: "blue",
            "Total": "magenta",
        }

        self.config[Config.CAR_SIZE_TABU] = 0
        self.config[Config.TEST_LABEL] = ""
        self.config[Config.TUNE_LABEL] = None

        # Speed cars (kmh) - 20KMH
        self.config["SPEED"] = 20

        self.config["PROJECTION"] = PROJECTION_MERCATOR
        self.config[Config.LEVEL_DIST_LIST] = []

        # List of time aggregation (min) starting with the disaggregate
        # level, that is, the time increment
        self.config[Config.LEVEL_TIME_LIST] = [
            self.config[Config.TIME_INCREMENT]
        ]

        # Battery ######################################################

        self.config["RECHARGE_RATE"] = 483  # km/hour
        self.config["BATTERY_SIZE_DISTANCE"] = 322

        self.config["BATTERY_SIZE_KWH_DISTANCE"] = (
            self.config["BATTERY_SIZE"] / self.config["BATTERY_SIZE_DISTANCE"]
        )

        self.config["BATTERY_SIZE_DISTANCE_LEVEL"] = (
            self.config["BATTERY_SIZE_DISTANCE"]
            / self.config["BATTERY_LEVELS"]
        )

        # Time #########################################################

        self.config[Config.STEP_SECONDS] = 60

        # Network ######################################################
        self.config[Config.CENTROID_LEVEL] = 0
        self.config[Config.N_CLOSEST_NEIGHBORS] = ((0, 8),)
        self.config[Config.N_CLOSEST_NEIGHBORS_EXPLORE] = ((1, 8),)
        self.config[Config.NEIGHBORHOOD_LEVEL] = 1
        self.config[Config.REBALANCE_LEVEL] = (1,)
        self.config[Config.REBALANCE_REACH] = None
        self.config[Config.REBALANCE_MULTILEVEL] = False
        self.config[Config.PENALIZE_REBALANCE] = True

        # Constraints
        self.config[Config.SQ_GUARANTEE] = False
        self.config[Config.MAX_CARS_LINK] = None

        # How much time does it take (min) to recharge one single level?
        self.config["RECHARGE_TIME_SINGLE_LEVEL"] = int(
            60
            * self.config["BATTERY_SIZE_DISTANCE_LEVEL"]
            / self.config["RECHARGE_RATE"]
        )

        self.config[Config.DEMAND_CENTER_LEVEL] = 3

        # DEMAND DATA ##################################################
        # Data correspond to 1 day NY demand
        self.config[Config.DEMAND_TOTAL_HOURS] = 24
        self.config[Config.DEMAND_EARLIEST_HOUR] = 0
        self.config[Config.DEMAND_RESIZE_FACTOR] = 1
        self.config[Config.DEMAND_MAX_STEPS] = int(
            self.config[Config.DEMAND_TOTAL_HOURS] * 60 / self.time_increment
        )
        self.config[Config.EARLIEST_STEP_MIN] = int(
            self.config[Config.DEMAND_EARLIEST_HOUR] * 60 / self.time_increment
        )
        self.config[Config.ALLOW_USER_BACKLOGGING] = False

        # USERS ###################################################### #
        self.config[Config.TRIP_BASE_FARE] = {
            Config.SQ_CLASS_1: 4.8,
            Config.SQ_CLASS_2: 2.4,
        }
        self.config[Config.TRIP_REJECTION_PENALTY] = {
            Config.SQ_CLASS_1: 4.8,
            Config.SQ_CLASS_2: 2.4,
        }
        self.config[Config.TRIP_DISTANCE_RATE_KM] = {
            Config.SQ_CLASS_1: 1,
            Config.SQ_CLASS_2: 1,
        }
        self.config[Config.TRIP_TOLERANCE_DELAY_MIN] = {
            Config.SQ_CLASS_1: 5,
            Config.SQ_CLASS_2: 5,
        }
        self.config[Config.TRIP_MAX_PICKUP_DELAY] = {
            Config.SQ_CLASS_1: 5,
            Config.SQ_CLASS_2: 10,
        }

        self.config[Config.TRIP_CLASS_PROPORTION] = {
            Config.SQ_CLASS_1: 0.1,
            Config.SQ_CLASS_2: 0.9,
        }

        # HIRING ##################################################### #
        self.config[Config.PROFIT_MARGIN] = 0.3
        self.config[Config.CONTRACT_DURATION_LEVEL] = 5  # Min.
        self.config[Config.CONGESTION_PRICE] = 10
        self.config[Config.MIN_CONTRACT_DURATION] = 0.5  # 30 min
        self.config[Config.MEAN_CONTRACT_DURATION] = 2  # 2 hours
        self.config[Config.MAX_CONTRACT_DURATION] = True

        # LEARNING ################################################### #
        self.config[Config.DISCOUNT_FACTOR] = 1
        self.config[Config.HARMONIC_STEPSIZE] = 1
        self.config[Config.STEPSIZE] = 0.1
        self.config[Config.UPDATE_METHOD] = WEIGHTED_UPDATE
        self.current_iteration = 0
        self.current_step = 0
        # MATCHING ################################################### #
        self.config[Config.MATCH_METHOD] = Config.MATCH_DISTANCE
        self.config[Config.MATCH_LEVEL] = 0
        self.config[Config.MATCH_MAX_NEIGHBORS] = 8
        self.config[Config.MATCHING_LEVELS] = (3, 4)
        self.config[Config.LEVEL_RC] = 2
        self.config[Config.MATCHING_DELAY] = 2  # min
        # Disabled (cars can stay idle indefinetely)
        self.config[Config.MAX_IDLE_STEP_COUNT] = None

        # Model
        self.config[Config.LINEARIZE_INTEGER_MODEL] = True
        self.config[Config.USE_ARTIFICIAL_DUALS] = False
        self.config[Config.FAV_FLEET_SIZE] = 0
        # mean, std, clip_a, clip_b
        self.config[Config.FAV_EARLIEST_FEATURES] = (8, 1, 5, 9)
        self.config[Config.FAV_AVAILABILITY_FEATURES] = (2, 1, 1, 4)
        self.config[Config.SEPARATE_FLEETS] = False
        # self.update(config)

        self.config[Config.REACHABLE_NEIGHBORS] = False
        self.config[Config.MAX_TARGETS] = 1000
        self.config[Config.ACTIVATE_THOMPSON] = False
        self.config[Config.IDLE_ANNEALING] = None

        self.config[Config.MYOPIC] = False
        self.config[Config.SAVE_PROGRESS] = 1
        self.config[Config.POLICY_RANDOM] = False

        # Names
        self.config[Config.USE_SHORT_PATH] = False

        self.config[Config.SAVE_TRIP_DATA] = False

        self.config[Config.SAVE_FLEET_DATA] = False

        self.config[Config.USE_CLASS_PROB] = False

    # ---------------------------------------------------------------- #
    # Network version ################################################ #
    # ---------------------------------------------------------------- #

    @property
    def cars_start_from_last_positions(self):
        """True if cars should start from the last visited positions"""
        return self.config[Config.FLEET_START] == FLEET_START_LAST

    @property
    def cars_start_from_initial_positions(self):
        """True if cars should start from the positions chosen in the 
        beginning of the experiment"""
        return self.config[Config.FLEET_START] == FLEET_START_SAME

    @property
    def cars_start_from_random_positions(self):
        """True if cars should start from random positions"""
        return self.config[Config.FLEET_START] == FLEET_START_RANDOM

    @property
    def battery_size_distance(self):
        """Battery size in number of miles """
        return self.config["BATTERY_SIZE_DISTANCE"]

    def get_step_level(self, level):
        return level * self.config["STEP_SECONDS"]

    @property
    def projection(self):
        """Coordinates can be mercator or gps"""
        return self.config["PROJECTION"]

    @property
    def level_dist_list(self):
        """Coordinates can be mercator or gps"""
        return self.config[Config.LEVEL_DIST_LIST]

    @property
    def level_time_list(self):
        """Coordinates can be mercator or gps"""
        return self.config[Config.LEVEL_TIME_LIST]

    @property
    def level_car_origin_dict(self):
        """Car origin for each aggregated level"""
        return self.config[Config.LEVEL_CAR_ORIGIN]

    @property
    def reachable_neighbors(self):
        """Whether method should use all reachable neighbors
        (within a time limit) instead of level neighbors"""
        return self.config[Config.REACHABLE_NEIGHBORS]

    @property
    def activate_thompson(self):
        """Whether method should use all reachable neighbors
        (within a time limit) instead of level neighbors"""
        return self.config[Config.ACTIVATE_THOMPSON]

    @property
    def level_car_type_dict(self):
        """Car type for each aggregated level"""
        return self.config[Config.LEVEL_CAR_TYPE]

    @property
    def level_contract_duration_dict(self):
        """Contract duration for each car type and aggregated level"""
        return self.config[Config.LEVEL_CONTRACT_DURATION]

    @property
    def neighborhood_level(self):
        """Extent of the reachability of the region centers. E.g.,
        level = 0 - Region centers are nodes
        level = 1 - Region centers can access neighbors within
        step_seconds distance.
        level = 2 - Region cetners can access neighbors within
        2*step_seconds distance.
        """
        return self.config["NEIGHBORHOOD_LEVEL"]

    @property
    def n_neighbors(self):
        """Number of closest region centers each region center can
        access."""
        return self.config["N_CLOSEST_NEIGHBORS"]

    @property
    def centroid_level(self):
        """Centroid level for ODs. If 0, ODs are id nodes.
        If > 0, get id of superior hierarchical level."""
        return self.config[Config.CENTROID_LEVEL]

    @property
    def n_neighbors_explore(self):
        """Region centers to explore when parked for more than
        MAX_IDLE_STEP_COUNT."""
        return self.config[Config.N_CLOSEST_NEIGHBORS_EXPLORE]

    @property
    def linearize_integer_model(self):
        """Transform integer model into linear model (fixed) and
        resolve"""
        return self.config[Config.LINEARIZE_INTEGER_MODEL]

    @property
    def use_artificial_duals(self):
        """Insert vf in missed demand positions"""
        return self.config[Config.USE_ARTIFICIAL_DUALS]

    # ---------------------------------------------------------------- #
    # Matching ####################################################### #
    # ---------------------------------------------------------------- #

    @property
    def match_method(self):
        return self.config[Config.MATCH_METHOD]

    @property
    def match_level(self):
        return self.config[Config.MATCH_LEVEL]

    @property
    def matching_levels(self):
        return self.config[Config.MATCHING_LEVELS]

    def match_neighbors(self):
        return self.config[Config.MATCH_METHOD] == Config.MATCH_NEIGHBORS

    def match_in_center(self):
        return self.config[Config.MATCH_METHOD] == Config.MATCH_CENTER

    @property
    def match_max_neighbors(self):
        return (
            self.config[Config.MATCH_MAX_NEIGHBORS]
            == Config.MATCH_MAX_NEIGHBORS
        )

    @property
    def level_rc(self):
        """Region center level from where cars are hired"""
        return self.config[Config.LEVEL_RC]

    # ---------------------------------------------------------------- #
    # Network data ################################################### #
    # ---------------------------------------------------------------- #

    @property
    def name(self):
        return self.config[Config.NAME]

    @property
    def region(self):
        return self.config[Config.REGION]

    @property
    def node_count(self):
        return self.config[Config.NODE_COUNT]

    @property
    def fav_fleet_size(self):
        return self.config[Config.FAV_FLEET_SIZE]

    @property
    def fav_availability_features(self):
        return self.config[Config.FAV_AVAILABILITY_FEATURES]

    @property
    def fav_earliest_features(self):
        return self.config[Config.FAV_EARLIEST_FEATURES]

    @property
    def separate_fleets(self):
        return self.config[Config.SEPARATE_FLEETS]

    @property
    def fav_depot_level(self):
        return self.config[Config.FAV_DEPOT_LEVEL]

    @property
    def depot_share(self):
        """Percentage of nodes which are depots"""
        return self.config[Config.DEPOT_SHARE]

    @property
    def edge_count(self):
        return self.config[Config.EDGE_COUNT]

    @property
    def center_count_dict(self):
        return self.config[Config.CENTER_COUNT]

    @property
    def rebalance_level(self):
        """Level of centers cars rebalance to"""
        return self.config[Config.REBALANCE_LEVEL]

    @property
    def penalize_rebalance(self):
        # If True, rebalancing is further punished (discount value that
        # could have been gained by staying still)
        return self.config[Config.PENALIZE_REBALANCE]

    @property
    def max_cars_link(self):
        # If True, add service quality constraints
        return self.config[Config.MAX_CARS_LINK]

    @property
    def sq_guarantee(self):
        # If True, add service quality constraints
        return self.config[Config.SQ_GUARANTEE]

    @property
    def test_label(self):
        return self.config[Config.TEST_LABEL]

    @property
    def rebalance_reach(self):
        """Car can reach nodes up to 'rebalance_reach' distance"""
        return self.config[Config.REBALANCE_REACH]

    @property
    def rebalance_multilevel(self):
        """If True, rebalance to all levels below REBALANCE LEVEL set"""
        return self.config[Config.REBALANCE_MULTILEVEL]

    @property
    def profit_margin(self):
        """Profit margin of hired cars"""
        return self.config[Config.PROFIT_MARGIN]

    @property
    def congestion_price(self):
        """How much cars pay to circulate in downtown"""
        return self.config[Config.CONGESTION_PRICE]

    @property
    def mean_contract_duration(self):
        """How long cars are available to be hired in average"""
        return self.config[Config.MEAN_CONTRACT_DURATION]

    @property
    def min_contract_duration(self):
        """Minimum available time necessary to work for the platform"""
        return self.config[Config.MIN_CONTRACT_DURATION]

    @property
    def max_contract_duration(self):
        """Return True, if FAVs stay until the end of the experiment"""
        return self.config[Config.MAX_CONTRACT_DURATION]

    @property
    def contract_duration_level(self):
        """Contract duration is sliced in levels of X minutes"""
        return self.config[Config.CONTRACT_DURATION_LEVEL]

    # LEARNING ################################################### #
    @property
    def discount_factor(self):
        """Post cost is multiplied by weight in [0,1]"""
        return self.config[Config.DISCOUNT_FACTOR]

    @property
    def stepsize_harmonic(self):
        """Value 'a' from harmonic stepsize = a/(a+n)"""
        return self.config[Config.HARMONIC_STEPSIZE]

    @property
    def stepsize_rule(self):
        """Fixed, harmonic, or """
        return self.config[Config.STEPSIZE_RULE]

    @property
    def stepsize_constant(self):
        """Fixed size stepsize, generally 0.1"""
        return self.config[Config.STEPSIZE_CONSTANT]

    @property
    def update_method(self):
        """How value functions are updated"""
        return self.config[Config.UPDATE_METHOD]

    def update_values_averaged(self):
        """How value functions are updated"""
        return self.update_method == AVERAGED_UPDATE

    def update_values_smoothed(self):
        """How value functions are updated"""
        return self.update_method == WEIGHTED_UPDATE

    def get_levels(self):
        levels = ", ".join(
            [
                (
                    f"{self.config[Config.LEVEL_TIME_LIST][temporal]}-"
                    f"{self.config[Config.LEVEL_DIST_LIST][spatial]}"
                )
                for (
                    temporal,
                    spatial,
                    battery,
                    contract,
                    car_type,
                    car_origin,
                ) in self.config[Config.AGGREGATION_LEVELS]
            ]
        )
        return levels

    @property
    def time_max_cars_link(self):
        return self.config[Config.TIME_MAX_CARS_LINK]

    def get_reb_neighbors(self):
        reb_neigh = ", ".join(
            [
                f"{level}-{n_neighbors}"
                for level, n_neighbors in self.config[
                    Config.N_CLOSEST_NEIGHBORS
                ]
            ]
        )
        return reb_neigh

    @property
    def max_targets(self):
        return self.config[Config.MAX_TARGETS]

    # ################################################################ #
    # LABELS ######################################################### #
    # ################################################################ #
    @property
    def label_reb_neigh(self):
        reb_neigh = ", ".join(
            [
                f"{level}-{n_neighbors}"
                for level, n_neighbors in self.config[
                    Config.N_CLOSEST_NEIGHBORS
                ]
            ]
        )
        return reb_neigh

    @property
    def label_reach_neigh(self):
        if self.reachable_neighbors:
            reach_neigh = f"reach_{self.time_increment:01}min"
        return reach_neigh

    @property
    def label_reb_neigh_explore(self):
        reb_neigh_explore = ", ".join(
            [
                f"{level}-{n_neighbors}"
                for level, n_neighbors in self.config[
                    Config.N_CLOSEST_NEIGHBORS_EXPLORE
                ]
            ]
        )
        return reb_neigh_explore

    @property
    def label_idle_annealing(self):
        idle_annealing = "[X]" if self.idle_annealing is not None else ""
        return idle_annealing

    @property
    def label_levels(self):
        levels = ", ".join(
            [
                (
                    f"{temporal}"
                    f"{spatial}"
                    f"{contract}"
                    f"{car_type}"
                    f"{car_origin}"
                )
                for (
                    temporal,
                    spatial,
                    battery,
                    contract,
                    car_type,
                    car_origin,
                ) in self.config[Config.AGGREGATION_LEVELS]
            ]
        )

        return levels

    @property
    def label_sample(self):

        # Is the demand sampled or fixed?
        sample = "S" if self.config[Config.DEMAND_SAMPLING] else "F"

        return sample

    @property
    def label_start(self):
        # Does fleet start from random positions or last?
        # L = Last visited position
        # S = Same position
        start = (
            "L"
            if self.cars_start_from_last_positions
            else ("R" if self.cars_start_from_random_positions else "I")
        )

        return start

    @property
    def label_stations(self):
        # Set the initial stations of FAVs
        stations = ""
        if self.fav_fleet_size > 0:
            if self.depot_share:
                stations = f"[S{self.depot_share:3.2f}]"
            elif self.fav_depot_level:
                stations = f"[S{self.fav_depot_level}]"
        return stations

    @property
    def label_max_contract(self):
        # Set the initial stations of FAVs
        max_contract = ""
        if self.fav_fleet_size > 0:
            max_contract = (
                "[M]" if self.config[Config.MAX_CONTRACT_DURATION] else ""
            )
        return max_contract

    @property
    def label_max_link(self):
        max_link = (
            f"[L({self.max_cars_link:02})]" if self.max_cars_link else ""
        )
        return max_link

    @property
    def label_penalize(self):
        penalize = f"[P]" if self.penalize_rebalance else ""
        return penalize

    @property
    def label_lin(self):
        lin = (
            "LIN_INT_"
            if self.config[Config.LINEARIZE_INTEGER_MODEL]
            else "LIN_"
        )
        return lin

    @property
    def label_artificial(self):
        artificial = "[A]_" if self.config[Config.USE_ARTIFICIAL_DUALS] else ""
        return artificial

    @property
    def label_explore(self):
        explore = (
            f"-[{self.label_reb_neigh_explore}][I({self.config[Config.MAX_IDLE_STEP_COUNT]:02})]"
            if self.config[Config.MAX_IDLE_STEP_COUNT]
            else ""
        )
        return explore

    @property
    def label_thomp(self):
        thomp = (
            f"[thompson={self.max_targets:02}]"
            if self.activate_thompson
            else ""
        )
        return thomp

    @property
    def label(self, name=""):

        return self.concise_label

        if self.config[Config.TUNE_LABEL] is not None:
            return self.config[Config.TUNE_LABEL]

        return (
            f"{self.test_label}_"
            f"{self.label_idle_annealing}"
            f"{self.label_artificial}"
            f"{self.label_lin}"
            # f"{self.config[Config.NAME]}_"
            # f"{self.config[Config.DEMAND_SCENARIO]}_"
            f"cars={self.fleet_size:04}-{self.fav_fleet_size:04}{self.label_stations}{self.label_max_contract}({self.label_start})_"
            f"t={self.time_increment}_"
            # f"{self.config[Config.BATTERY_LEVELS]:04}_"
            f"levels[{len(self.aggregation_levels)}]=({self.label_levels})_"
            f"rebal=([{self.label_reb_neigh}]{self.label_explore}{self.label_thomp}[tabu={self.car_size_tabu:02}]){self.label_max_link}{self.label_penalize}_"
            # f"{self.config[Config.TIME_INCREMENT]:02}_"
            # f#"{self.config[Config.STEP_SECONDS]:04}_"
            # f"{self.config[Config.PICKUP_ZONE_RANGE]:02}_"
            # f"{self.config[Config.NEIGHBORHOOD_LEVEL]:02}_"
            # f"{reb_neigh}_"
            f"[{self.demand_earliest_hour:02}h,"
            f"+{self.offset_repositioning_min}m"
            f"+{self.demand_total_hours:02}h"
            f"+{self.offset_termination_min}m]_"
            f"match={self.matching_delay:02}_"
            f"{self.demand_resize_factor:3.2f}({self.label_sample})_"
            f"{self.discount_factor:3.2f}_"
            f"{self.stepsize_constant:3.2f}_"
            f"{self.sl_config_label}"
            # f"{self.config[Config.HARMONIC_STEPSIZE]:02}_"
            # f"{self.config[Config.CONGESTION_PRICE]:2}"
        )

    @property
    def concise_label(self):
        prob = "_P" if self.use_class_prob else ""
        return (
            f"{self.test_label}_"
            f"{self.label_idle_annealing}"
            f"{self.label_artificial}"
            f"{self.label_lin}"
            f"{(f'C{self.centroid_level}_' if self.centroid_level > 0 else '')}"
            # f"{self.config[Config.NAME]}_"
            # f"{self.config[Config.DEMAND_SCENARIO]}_"
            f"V={self.fleet_size:04}-{self.fav_fleet_size:04}{self.label_stations}{self.label_max_contract}({self.label_start})_"
            f"I={self.time_increment}_"
            # f"{self.config[Config.BATTERY_LEVELS]:04}_"
            f"L[{len(self.aggregation_levels)}]=({self.label_levels})_"
            f"R=([{self.label_reb_neigh}]{self.label_explore}{self.label_thomp}"
            # f"[tabu={self.car_size_tabu:02}])"
            f"{self.label_max_link}"
            # f"{self.label_penalize}_"
            # f"{self.config[Config.TIME_INCREMENT]:02}_"
            # f#"{self.config[Config.STEP_SECONDS]:04}_"
            # f"{self.config[Config.PICKUP_ZONE_RANGE]:02}_"
            # f"{self.config[Config.NEIGHBORHOOD_LEVEL]:02}_"
            # f"{reb_neigh}_"
            f"_T=[{self.demand_earliest_hour:02}h,"
            f"+{self.offset_repositioning_min}m"
            f"+{self.demand_total_hours:02}h"
            f"+{self.offset_termination_min}m]_"
            # f"match={self.matching_delay:02}_"
            f"{self.demand_resize_factor:3.2f}({self.label_sample})_"
            f"{self.discount_factor:3.2f}_"
            f"{self.stepsize_constant:3.2f}_"
            f"{self.sl_config_label}"
            # f"{prob}"
            # f"{self.config[Config.HARMONIC_STEPSIZE]:02}_"
            # f"{self.config[Config.CONGESTION_PRICE]:2}"
        )

    @staticmethod
    def load(file_path):
        with open(file_path, "r") as f:
            config = json.load(f)
            c = ConfigNetwork(config)
            c.update(config)
            for k1 in [
                "CENTER_COUNT",
                "LEVEL_CAR_ORIGIN",
                "LEVEL_CAR_TYPE",
                "LEVEL_CONTRACT_DURATION",
            ]:
                c.config[k1] = {int(k2): v for k2, v in c.config[k1].items()}

        return c

    @property
    def path_depot_list(self):
        # Save list of FAV depot ids
        # E.g.: rotterdam_N10364_E23048_fav_depots_0.01_level_00.npy
        path_depots = (
            f"{FOLDER_FAV_ORIGINS}{self.region.split(',')[0].lower()}"
            f"_C{self.centroid_level}"
            f"_N{self.node_count}_E{self.edge_count}"
            "_fav_depots_"
            f"{(self.depot_share if self.depot_share else 1):04.2f}_"
            "level_"
            f"{(self.fav_depot_level if self.fav_depot_level else 0):02}.npy"
        )

        return path_depots


def save_json(data, file_path=None, folder=None, file_name=None):
    if not file_path:
        file_path = folder + file_name + ".json"
    with open(file_path, "a+") as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4)
