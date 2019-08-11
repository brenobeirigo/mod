import os
import sys
import json
from datetime import datetime, timedelta

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

# Trip data to group in steps
TRIPS_FILE_ALL = "trips_2011-02-01.csv"
TRIPS_FILE_4 = "32874_samples_01_feb_2011_NY.csv"
NY_TRIPS_EXCERPT_DAY = root + f"/data/input/nyc/{TRIPS_FILE_ALL}"

FOLDER_TUNING = root + "/data/input/tuning/"
FOLDER_NYC_TRIPS = root + f"/data/input/nyc/"
TRIP_FILES = [
    f'{FOLDER_NYC_TRIPS}{t}'
    for t in [
        "trips_2011-01-04-enriched.csv"
        # "trips_2011-02-01.csv",
        # "trips_2011-02-08.csv",
        # "trips_2011-02-15.csv",
        # "trips_2011-02-22.csv",
    ]
]

# Car statuses
IDLE = "Idle"
RECHARGING = "Recharging"
ASSIGN = "With passenger"
CRUISING = "Cruising"
REBALANCE = "Rebalancing"

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

    # This configuration refers to which test case?
    TEST_LABEL = "TEST_LABEL"
    # Determined in tuning
    TUNE_LABEL = "TUNE_LABEL"

    SPEED = "SPEED"
    FLEET_SIZE = "FLEET_SIZE"
    FLEET_START = "FLEET_START"

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
    TRIP_COST_DISTANCE = "TRIP_COST_DISTANCE"
    TOTAL_TRIPS = "TOTAL_TRIPS"
    MIN_TRIPS = "MIN_TRIPS"
    MAX_TRIPS = "MAX_TRIPS"
    PICKUP_ZONE_RANGE = "PICKUP_ZONE_RANGE"

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

    ZONE_WIDTH = "ZONE_WIDTH"
    VALID_ZONES = "VALID_ZONES"
    ROWS = "ROWS"
    COLS = "COLS"
    ORIGIN_CENTERS = "ORIGIN_CENTERS"
    ORIGIN_CENTER_ZONE_SIZE = "ORIGIN_CENTER_ZONE_SIZE"
    DESTINATION_CENTERS = "DESTINATION_CENTERS"

    # Recharging
    RECHARGE_THRESHOLD = "RECHARGE_THRESHOLD"
    RECHARGE_BASE_FARE = "RECHARGE_BASE_FARE"
    RECHARGE_COST_DISTANCE = "RECHARGE_COST_DISTANCE"
    RECHARGE_RATE = "RECHARGE_RATE"
    COST_RECHARGE_SINGLE_INCREMENT = "COST_RECHARGE_SINGLE_INCREMENT"
    TIME_INCREMENT = "TIME_INCREMENT"
    TOTAL_TIME = "TOTAL_TIME"
    OFFSET_REPOSIONING = "OFFSET_REPOSIONING"
    OFFSET_TERMINATION = "OFFSET_TERMINATION"
    TIME_PERIODS = "TIME_PERIODS"

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
    NEIGHBORHOOD_LEVEL = "NEIGHBORHOOD_LEVEL"
    REBALANCE_LEVEL = "REBALANCE_LEVEL"
    PENALIZE_REBALANCE = "PENALIZE_REBALANCE"
    REBALANCE_REACH = "REBALANCE_REACH"
    REBALANCE_MULTILEVEL = "REBALANCE_MULTILEVEL"
    MATCHING_LEVELS = "MATCHING_LEVELS"

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

    def __init__(self, config):

        self.config = config

    # ################################################################ #
    # ## Area ######################################################## #
    # ################################################################ #

    @property
    def origin_centers(self):
        return self.config[Config.ORIGIN_CENTERS]

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
    def trip_cost_fare(self):
        """Trip cost per mile in dollars"""
        return self.config["TRIP_COST_DISTANCE"]

    @property
    def pickup_zone_range(self):
        """Duration of the time steps in (min)"""
        return self.config["PICKUP_ZONE_RANGE"]

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

    @property
    def offset_repositioning(self):
        """Number of time steps with no trips before 
        demand (for reposition)"""
        return self.config["OFFSET_REPOSIONING"]

    @property
    def offset_termination(self):
        """Number of time steps with no trips after demand (so
        that all passengers can be delivered)"""
        return self.config["OFFSET_TERMINATION"]

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

    def update(self, dict_update):

        # Guarantee elements are tuples
        if Config.REBALANCE_LEVEL in dict_update:
            dict_update[Config.REBALANCE_LEVEL] = tuple(
                dict_update[Config.REBALANCE_LEVEL]
            )

        if Config.N_CLOSEST_NEIGHBORS in dict_update:
            dict_update[Config.N_CLOSEST_NEIGHBORS] = tuple(
                dict_update[Config.N_CLOSEST_NEIGHBORS]
            )

        if Config.AGGREGATION_LEVELS in dict_update:
            dict_update[Config.AGGREGATION_LEVELS] = tuple(
                dict_update[Config.AGGREGATION_LEVELS]
            )

        self.config.update(dict_update)

        self.config["BATTERY_SIZE_KWH_DISTANCE"] = (
            self.config["BATTERY_SIZE"] / self.config["BATTERY_SIZE_DISTANCE"]
        )

        self.config["BATTERY_SIZE_KWH_DISTANCE"] = (
            self.config["BATTERY_SIZE"] / self.config["BATTERY_SIZE_DISTANCE"]
        )

        # # Total number of time periods
        # self.config["TIME_PERIODS"] = int(
        #     self.config["OFFSET_REPOSIONING"]
        #     + self.config["TOTAL_TIME"] * 60 / self.config["TIME_INCREMENT"]
        #     + self.config["OFFSET_TERMINATION"]
        # )

        #       Total number of time periods
        self.config["TIME_PERIODS"] = int(
            self.config["OFFSET_REPOSIONING"]
            + self.config[Config.DEMAND_TOTAL_HOURS]
            * 60
            / self.config["TIME_INCREMENT"]
            + self.config["OFFSET_TERMINATION"]
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

        # Creating folders to log MIP models
        self.folder_mip = FOLDER_OUTPUT + self.label + "/mip/"
        self.folder_mip_log = self.folder_mip + "log/"
        self.folder_mip_lp = self.folder_mip + "lp/"
        self.folder_adp_log = FOLDER_OUTPUT + self.label + "/logs/"

        self.config[Config.TIME_INCREMENT_TIMEDELTA] = timedelta(
            minutes=self.config[Config.TIME_INCREMENT]
        )

        self.config[Config.DEMAND_EARLIEST_DATETIME] = (
            datetime.strptime("2011-02-01 00:00", "%Y-%m-%d %H:%M")
            + timedelta(hours=self.config[Config.DEMAND_EARLIEST_HOUR])
            - timedelta(minutes=self.config[Config.OFFSET_REPOSIONING])
        )

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
    def demand_resize_factor(self):
        return self.config[Config.DEMAND_RESIZE_FACTOR]

    @property
    def demand_max_steps(self):
        return self.config[Config.DEMAND_MAX_STEPS]

    @property
    def demand_earliest_step_min(self):
        return self.config[Config.EARLIEST_STEP_MIN]

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
        self.config["OFFSET_REPOSIONING"] = 3

        # Offset at the end to guarantee trips terminate
        self.config["OFFSET_TERMINATION"] = 11

        # Total number of time periods
        self.config["TIME_PERIODS"] = int(
            self.config["OFFSET_REPOSIONING"]
            + self.config["TOTAL_TIME"] * 60 / self.config["TIME_INCREMENT"]
            + self.config["OFFSET_TERMINATION"]
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
        self.config["RECHARGE_COST_DISTANCE"] = 0.1  # dollar
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

        if not config:
            config = dict()

        super().__init__(config)

        self.color_fleet_status = {
            IDLE: "#24aafe",
            ASSIGN: "#53bc53",
            REBALANCE: "firebrick",
            RECHARGING: "#e55215",
            CRUISING: "#e55215",
            "Total": "black",
        }

        self.config[Config.TEST_LABEL] = ""
        self.config[Config.TUNE_LABEL] = None

        # Speed cars (kmh) - 20KMH
        self.config["SPEED"] = 20

        self.config["PROJECTION"] = PROJECTION_MERCATOR
        self.config[Config.LEVEL_DIST_LIST] = []

        # List of time aggregation (min) starting with the disaggregate
        # level, that is, the time increment
        self.config[Config.LEVEL_TIME_LIST] = [self.config[Config.TIME_INCREMENT]]

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
        self.config[Config.N_CLOSEST_NEIGHBORS] = (4,)
        self.config[Config.NEIGHBORHOOD_LEVEL] = 1

        self.config[Config.REBALANCE_LEVEL] = (1,)
        self.config[Config.REBALANCE_REACH] = None
        self.config[Config.REBALANCE_MULTILEVEL] = False
        self.config[Config.PENALIZE_REBALANCE] = True

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

        # HIRING ##################################################### #
        self.config[Config.PROFIT_MARGIN] = 0.3
        self.config[Config.CONTRACT_DURATION_LEVEL] = 5  # Min.
        self.config[Config.CONGESTION_PRICE] = 10

        # LEARNING ################################################### #
        self.config[Config.DISCOUNT_FACTOR] = 1
        self.config[Config.HARMONIC_STEPSIZE] = 1
        self.config[Config.STEPSIZE] = 0.1
        self.config[Config.UPDATE_METHOD] = WEIGHTED_UPDATE

        # MATCHING ################################################### #
        self.config[Config.MATCH_METHOD] = Config.MATCH_DISTANCE
        self.config[Config.MATCH_LEVEL] = 0
        self.config[Config.MATCH_MAX_NEIGHBORS] = 8
        self.config[Config.MATCHING_LEVELS] = (3, 4)
        self.config[Config.LEVEL_RC] = 2



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
        return self.config[Config.MATCH_MAX_NEIGHBORS] == Config.MATCH_MAX_NEIGHBORS

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

    @property
    def label(self, name=""):

        if self.config[Config.TUNE_LABEL] is not None:
            return self.config[Config.TUNE_LABEL]

        reb_neigh = ", ".join(
            [
                f"{level}-{n_neighbors}"
                for level, n_neighbors in self.config[Config.N_CLOSEST_NEIGHBORS]
            ]
        )
        levels = ", ".join([
            (
                f"{self.config[Config.LEVEL_TIME_LIST][temporal]}-"
                f"{self.config[Config.LEVEL_DIST_LIST][spatial]}"
            )
            for (temporal, spatial, contract, car_type, car_origin) in self.config[Config.AGGREGATION_LEVELS]])

        # Is the demand sampled or fixed?
        sample = ("S" if self.config[Config.DEMAND_SAMPLING] else "F")

        # Does fleet start from random positions or last?
        # L = Last visited position
        # S = Same position
        start = (
            "L" if self.cars_start_from_last_positions else (
                "R" if self.cars_start_from_random_positions else "I"
            )
        )

        return (
            f"{self.config[Config.TEST_LABEL]}_"
            # f"{self.config[Config.NAME]}_"
            # f"{self.config[Config.DEMAND_SCENARIO]}_"
            f"cars={self.config[Config.FLEET_SIZE]:04}({start})_"
            #f"{self.config[Config.BATTERY_LEVELS]:04}_"
            f"levels[{len(self.config[Config.AGGREGATION_LEVELS])}]=({levels})_"
            f"rebal=({reb_neigh})_"
            # f"{self.config[Config.TIME_INCREMENT]:02}_"
            # f#"{self.config[Config.STEP_SECONDS]:04}_"
            # f"{self.config[Config.PICKUP_ZONE_RANGE]:02}_"
            # f"{self.config[Config.NEIGHBORHOOD_LEVEL]:02}_"
            # f"{reb_neigh}_"
            f"[{self.config[Config.DEMAND_EARLIEST_HOUR]:02}h,"
            f"+{self.config[Config.DEMAND_TOTAL_HOURS]:02}h]_"
            f"{self.config[Config.DEMAND_RESIZE_FACTOR]:3.2f}({sample})_"
            f"{self.config[Config.DISCOUNT_FACTOR]:3.2f}_"
            f"{self.config[Config.STEPSIZE_CONSTANT]:3.2f}"
            # f"{self.config[Config.HARMONIC_STEPSIZE]:02}_"
            # f"{self.config[Config.CONGESTION_PRICE]:2}"
        )


    def log_path(self, iteration=""):
        return self.folder_adp_log + f"{iteration:04}.log"


    def save(self, file_path):
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile, sort_keys=True, indent=4)


def save_json(data, file_path=None, folder=None, file_name=None):
    if not file_path:
        file_path = folder + file_name + ".json"
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4)
