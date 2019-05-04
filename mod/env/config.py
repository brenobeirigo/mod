import os
import sys

# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

# Trip data to group in steps
NY_TRIPS_EXCERPT_DAY = root + '/data/input/32874_samples_01_feb_2011_NY.csv'

class Config:

    def __init__(self, config):
        
        self.config = config

    ################################################################
    ### Battery info ###############################################
    ################################################################

    @property
    def recharge_base_fare(self):
        """Trip base fare in dollars"""
        return self.config['RECHARGE_BASE_FARE']

    @property
    def recharge_cost_mile(self):
        """Trip base fare in dollars"""
        return self.config['RECHARGE_COST_MILE']
        
    @property
    def recharge_rate(self):
        """Trip base fare in dollars"""
        return self.config['RECHARGE_RATE']
    
    @property
    def recharge_threshold(self):
        """Minimum battery charge percentage 
        (float in [0,1] interval) """
        return self.config['RECHARGE_THRESHOLD']

    def calculate_cost_recharge(self, recharging_time_min):
        recharging_time_h = recharging_time_min/60.0
        return (
            self.config['RECHARGE_BASE_FARE']
            + (
                self.config['RECHARGE_COST_MILE']
                *self.config['RECHARGE_RATE']
                *recharging_time_h
            )
        )

    def get_full_recharging_time(self, distance):
        """Get recharge time in relation to recharge distance
        according to recharge rate in miles/hour
        
        Arguments:
            distance {float} -- miles
        
        Returns:
            int, int --recharge time in minutes and time steps
        """
        
        hours_recharging = distance/self.config['RECHARGE_RATE']

        minutes_recharging = hours_recharging*60

        time_steps_recharging = (
            minutes_recharging/self.config['TIME_INCREMENT']
        )

        # print(
        #     f'RECHARGING(miles:{distance:>3.2f}'
        #     f' - h:{hours_recharging:>3.2f}'
        #     f' - m:{minutes_recharging:>3.2f})'
        #     f' - steps:{time_steps_recharging:>3.2f})'
        # )

        return minutes_recharging, int(round(time_steps_recharging))

    ################################################################
    ### Battery ####################################################
    ################################################################

    @property
    def battery_size_miles(self):
        """Battery size in number of miles """
        return self.config['BATTERY_SIZE_MILE']
    
    @property
    def battery_levels(self):
        """Number of discrete levels"""
        return self.config['BATTERY_LEVELS']
    
    @property
    def battery_size_kwh_mile(self):
        """Maximum battery size in miles"""
        return self.config['BATTERY_SIZE_KWH_MILE']

    ################################################################
    ### Trip #######################################################
    ################################################################

    @property
    def trip_base_fare(self):
        """Trip base fare in dollars"""
        return self.config['TRIP_BASE_FARE']

    @property
    def trip_cost_fare(self):
        """Trip cost per mile in dollars"""
        return self.config['TRIP_COST_MILE']

    @property
    def pickup_zone_range(self):
        """Duration of the time steps in (min)"""
        return self.config['PICKUP_ZONE_RANGE']

    @property
    def time_increment(self):
        """Duration of the time steps in (min)"""
        return self.config['TIME_INCREMENT']
        
    @property
    def speed_mph(self):
        """Speed in mph"""
        return self.config['SPEED_MPH']

    @property
    def zone_widht(self):
        """Zone width in miles"""
        return self.config['ZONE_WIDTH']

    @property
    def time_steps(self):
        """Time steps in minutes"""
        return self.config['TIME_PERIODS']

    @property
    def rows(self):
        """Number of rows in zone"""
        return self.config['ROWS']
    

    @property
    def cols(self):
        """Number of colums in zone"""
        return self.config['COLS']

    @property
    def fleet_size(self):
        """Number of cars"""
        return self.config['FLEET_SIZE']

    @property
    def aggregation_levels(self):
        """Number of aggregation levels"""
        return self.config['AGGREGATION_LEVELS']


    ################################################################
    ### Demand #####################################################
    ################################################################

    @property
    def min_trips(self):
        """Minimum number of trips (15min) """
        return self.config['MIN_TRIPS']
    
    @property
    def max_trips(self):
        """Maximum number of trips (15min)"""
        return self.config['MAX_TRIPS']
    
    @property
    def offset_repositioning(self):
        """Number of time steps with no trips before 
        demand (for reposition)"""
        return self.config['OFFSET_REPOSIONING']
    
    @property
    def offset_termination(self):
        """Number of time steps with no trips after demand (so
        that all passengers can be delivered)"""
        return self.config['OFFSET_TERMINATION']


    def resize_zones(self, factor):
        # Each zone has width = 0.5 miles
        self.config['ZONE_WIDTH'] = int(self.config['ZONE_WIDTH']*factor)
        self.config['ROWS'] = int(self.config['ROWS']*factor)
        self.config['COLS'] = int(self.config['COLS']*factor)
        self.config['VALID_ZONES'] = int(self.config['VALID_ZONES']*(factor*factor))

    def __str__(self):
        return self.config.__str__()

    def calculate_fare(self, distance_trip):
        return self.config['TRIP_BASE_FARE'] + self.config['TRIP_COST_MILE']*distance_trip

    def update(self,dict_update):
        self.config.update(dict_update)

class ConfigStandard(Config):

    def __init__(self, config = None):

        if not config:
            config = dict()
        super().__init__(config)
        
        self.config = dict()

        ############################################################
        # Car ######################################################
        ############################################################

        # Speed cars (mph) - 20MPH
        self.config['SPEED_MPH'] = 17
        
        # Total fleet
        self.config['FLEET_SIZE'] = 1500

        ############################################################
        # Battery ##################################################
        ############################################################

        self.config['BATTERY_SIZE_MILE'] = 200 # miles
        self.config['BATTERY_SIZE'] = 66 # kWh
        self.config['BATTERY_LEVELS'] = 20 # levels
        # How many KWh per mile?
        self.config['BATTERY_SIZE_KWH_MILE'] = (
            self.config['BATTERY_SIZE']/self.config['BATTERY_SIZE_MILE']
        )
        
        ############################################################
        # Demand characteristics ###################################
        ############################################################

        self.config['MEAN_TRIP_DISTANCE']=24.8 # miles
        self.config['SD_TRIP_DISTANCE']=7 # TODO it was guessed
        self.config['MINIMUM_TRIP_DISTANCE']=2 # 5th percentile is 6
        self.config['MAXIMUM_TRIP_DISTANCE']=65 # 95th percentile is 57.5
        # Simulation parameters
        self.config['TRIP_BASE_FARE'] = 2.4 # dollar
        # TODO can vary according to: 
        # - Where trip originates
        # - time of the day
        # - surge pricing
        self.config['TRIP_COST_MILE'] = 1 # dollar

        # Total number of trips (min, max) = (40, 640) in period
        self.config['TOTAL_TRIPS'] = 32874
        self.config['MIN_TRIPS'] = 40
        self.config['MAX_TRIPS'] = 640

        ############################################################
        # Map settings #############################################
        ############################################################

        # How many surrounding zones cars check for new costumers
        self.config['PICKUP_ZONE_RANGE'] = 2

        # How many aggregation levels (level 0, i.e., no aggregation)
        # included
        self.config['AGGREGATION_LEVELS'] = 5

        # Each zone has width = 0.5 miles
        self.config['ZONE_WIDTH'] = 0.5

        # The New Jersey is divided into 201 bt 304 rectangular
        # zones of width 0.5 miles with 21634 valid zones
        self.config['VALID_ZONES'] = 21634
        self.config['ROWS'] = 201
        self.config['COLS'] = 304

    
        self.config['RECHARGE_THRESHOLD'] = 0.1 # 10%
        self.config['RECHARGE_BASE_FARE'] = 1 # dollar
        self.config['RECHARGE_COST_MILE'] = 0.1 # dollar
        self.config['RECHARGE_RATE'] = 300 # miles/hour

        ############################################################
        # Time  ####################################################
        ############################################################

        # Lenght of time incremnts (min) - default is 15min
        self.config['TIME_INCREMENT'] = 15

        # Total horizon (h)
        self.config['TOTAL_TIME'] = 24

        # Offset at the beginning to reposition vehicles
        self.config['OFFSET_REPOSIONING'] = 3

        # Offset at the end to guarantee trips terminate
        self.config['OFFSET_TERMINATION'] = 11

        # Total number of time periods
        self.config['TIME_PERIODS'] = int(
            self.config['OFFSET_REPOSIONING']
            + self.config['TOTAL_TIME']*60/self.config['TIME_INCREMENT']
            + self.config['OFFSET_TERMINATION']
        )