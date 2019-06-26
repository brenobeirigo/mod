import os
import sys
from datetime import datetime, timedelta

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

# Trip data to group in steps
TRIPS_FILE_ALL = "231896_trips_NYC_2011-02-01.csv"
TRIPS_FILE_4 = "32874_samples_01_feb_2011_NY.csv"
NY_TRIPS_EXCERPT_DAY = root + f"/data/input/{TRIPS_FILE_ALL}"

# Output folder
FOLDER_OUTPUT = root + "/data/output/"

# Plot folders
FOLDER_SERVICE_PLOT = root + "/data/output/service_plot/"
FOLDER_FLEET_PLOT = root + "/data/output/fleet_plot/"
FOLDER_EPISODE_TRACK = root + "/data/output/track_episode/"

# Map projections for visualization
PROJECTION_MERCATOR = "MERCATOR"
PROJECTION_GPS = "GPS"

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


class Config:

    SPEED = "SPEED"
    FLEET_SIZE = "FLEET_SIZE"

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

    # Network
    STEP_SECONDS = "STEP_SECONDS"  # In km/h
    N_CLOSEST_NEIGHBORS = "N_CLOSEST_NEIGHBORS"
    NEIGHBORHOOD_LEVEL = "NEIGHBORHOOD_LEVEL"
    LEVEL_DIST_LIST = "LEVEL_LIST"
    REBALANCE_LEVEL = "REBALANCE_LEVEL"
    REBALANCE_MULTILEVEL = "REBALANCE_MULTILEVEL"

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

    # NETWORK INFO
    NAME = "NAME"
    REGION = "REGION"
    NODE_COUNT = "NODE_COUNT"
    EDGE_COUNT = "EDGE_COUNT"
    CENTER_COUNT = "CENTER_COUNT"

    # HIRING
    PROFIT_MARGIN = "PROFIT_MARGIN"

    def __init__(self, config):

        self.config = config

    ####################################################################
    ### Area ###########################################################
    ####################################################################

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
        """Minimum battery charge percentage 
        (float in [0,1] interval) """
        return self.config["RECHARGE_THRESHOLD"]

    def calculate_cost_recharge(self, recharging_time_min):
        recharging_time_h = recharging_time_min / 60.0
        return self.config["RECHARGE_BASE_FARE"] + (
            self.config["RECHARGE_COST_DISTANCE"]
            * self.config["RECHARGE_RATE"]
            * recharging_time_h
        )
    
    def get_travel_cost(self, distance):
        """Return the cost of travelling 'distance' meters"""
        return self.config["RECHARGE_COST_DISTANCE"] * distance/1000.0

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

    def calculate_fare(self, distance_trip):
        return (
            self.config["TRIP_BASE_FARE"]
            + self.config["TRIP_COST_DISTANCE"] * distance_trip
        )

    def update(self, dict_update):
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
            + self.config[Config.DEMAND_TOTAL_HOURS] * 60 / self.config["TIME_INCREMENT"]
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

        if not os.path.exists(self.folder_mip):
            os.makedirs(self.folder_mip_log)
            os.makedirs(self.folder_mip_lp)

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
        t = self.demand_earliest_datetime + steps*self.time_increment_timedelta
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
        self.config["AGGREGATION_LEVELS"] = 5

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

        # Speed cars (kmh) - 20KMH
        self.config["SPEED"] = 20

        self.config["PROJECTION"] = PROJECTION_MERCATOR
        self.config[Config.LEVEL_DIST_LIST] = []

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
        self.config["N_CLOSEST_NEIGHBORS"] = (4,)
        self.config["NEIGHBORHOOD_LEVEL"] = 1

        self.config[Config.REBALANCE_LEVEL] = (1,)
        self.config[Config.REBALANCE_MULTILEVEL] = False

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

        # LEARNING ################################################### #
        self.config[Config.DISCOUNT_FACTOR] = 1 
        self.config[Config.HARMONIC_STEPSIZE] = 1
        self.config[Config.STEPSIZE] = 0.1
    # ---------------------------------------------------------------- #
    # Network version ################################################ #
    # ---------------------------------------------------------------- #

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
    def rebalance_multilevel(self):
        """If True, rebalance to all levels below REBALANCE LEVEL set"""
        return self.config[Config.REBALANCE_MULTILEVEL]

    @property
    def profit_margin(self):
        """Level of centers cars rebalance to"""
        return self.config[Config.PROFIT_MARGIN]

    @property
    def discount_factor(self):
        """Post cost is multiplied by weight in [0,1]"""
        return self.config[Config.DISCOUNT_FACTOR]
    
    @property
    def harmonic_stepsize(self):
        """Value 'a' from harmonic stepsize = a/(a+n)"""
        return self.config[Config.HARMONIC_STEPSIZE]

    @property
    def label(self, name=""):

        reb_neigh = "_".join([
            f'{level}={n_neighbors}'
            for level, n_neighbors in list(zip(
                self.config[Config.REBALANCE_LEVEL],
                self.config[Config.N_CLOSEST_NEIGHBORS]
            ))])

        return (
            f"network_{self.config[Config.NAME]}_"
            f"{self.config[Config.DEMAND_SCENARIO]}_"
            f"{self.config[Config.FLEET_SIZE]:04}_"
            f"{self.config[Config.BATTERY_LEVELS]:04}_"
            f"{self.config[Config.AGGREGATION_LEVELS]}_"
            f"{self.config[Config.TIME_INCREMENT]:02}_"
            f"{self.config[Config.STEP_SECONDS]:04}_"
            f"{self.config[Config.PICKUP_ZONE_RANGE]:02}_"
            f"{self.config[Config.NEIGHBORHOOD_LEVEL]:02}_"
            f"{reb_neigh}_"
            f"{self.config[Config.DEMAND_EARLIEST_HOUR]:02}_"
            f"{self.config[Config.DEMAND_TOTAL_HOURS]:02}_"
            f"{self.config[Config.DEMAND_RESIZE_FACTOR]:02}_"
            f"{self.config[Config.DISCOUNT_FACTOR]:2}_"
            f"{self.config[Config.HARMONIC_STEPSIZE]:02}"
        )
