import functools
from collections import defaultdict

import numpy as np

# State indexes
TIME = 0
LOCATION = 1
BATTERY = 2
CONTRACT = 3
CARTYPE = 4
ORIGIN = 5

attributes = [TIME, LOCATION, BATTERY, CONTRACT, CARTYPE, ORIGIN]
STEPSIZE_HARMONIC = "HARM"
STEPSIZE_CONSTANT = "CONST"
STEPSIZE_MCCLAIN = "MCCL"

STEPSIZE_RULES = [STEPSIZE_HARMONIC, STEPSIZE_CONSTANT, STEPSIZE_MCCLAIN]

TIME_INCREMENT = 5

CONTRACT_DISAGGREGATE = 1
CONTRACT_L1 = 5
CONTRACT_L2 = 15
CONTRACT_L3 = 60

DISCARD = "-"
DISAGGREGATE = 0

adp_label_dict = {DISCARD: "-", DISAGGREGATE: "*"}


class Adp:
    def __init__(
            self,
            points,
            agregation_levels,
            temporal_levels,
            car_type_levels_dict,
            contract_levels_dict,
            car_origin_levels_dict,
            stepsize,
            stepsize_rule=STEPSIZE_CONSTANT,
            stepsize_constant=0.1,
            stepsize_harmonic=1,
    ):
        self.aggregation_levels = agregation_levels
        self.temporal_levels = temporal_levels
        self.car_type_levels_dict = car_type_levels_dict
        self.contract_levels_dict = contract_levels_dict
        self.car_origin_levels_dict = car_origin_levels_dict
        self.stepsize = stepsize
        self.points = points
        self.stepsize_rule = stepsize_rule
        self.stepsize_harmonic = stepsize_harmonic
        self.stepsize_constant = stepsize_constant

        # Adp track
        self.n = 0
        self.reward = list()
        self.service_rate = list()
        self.pk_delay = list()
        self.car_time = list()
        self.weights = defaultdict(list)

        # ML parameters
        self.init_learning()
        self.init_weighting_settings()

    ####################################################################
    # Learning #########################################################
    ####################################################################

    def init_learning(self):

        # -------------------------------------------------------------#
        # Learning #####################################################
        # -------------------------------------------------------------#

        # What is the value of a car attribute assuming aggregation
        # level and time steps

        self.values = [
            defaultdict(float) for g in range(len(self.aggregation_levels))
        ]

        # How many times a cell was actually accessed by a vehicle in
        # a certain region, aggregation level, and time
        self.count = [
            defaultdict(int) for g in range(len(self.aggregation_levels))
        ]

        self.current_weights = np.array([])

    def init_weighting_settings(self):
        # -------------------------------------------------------------#
        # Weighing #####################################################
        # -------------------------------------------------------------#

        # Transient bias
        self.transient_bias = [
            defaultdict(float) for g in range(len(self.aggregation_levels))
        ]

        self.variance_g = [
            defaultdict(float) for g in range(len(self.aggregation_levels))
        ]

        self.step_size_func = [
            defaultdict(lambda: 1.0)
            for g in range(len(self.aggregation_levels))
        ]

        self.lambda_stepsize = [
            defaultdict(lambda: 1.0)
            for g in range(len(self.aggregation_levels))
        ]

        # Aggregation bias
        self.aggregation_bias = [
            defaultdict(float) for g in range(len(self.aggregation_levels))
        ]

        # Estimate of the variance of observations made of state
        # s, using data from aggregation level g, after n
        # observations.
        self.variance_error = [
            defaultdict(float) for g in range(len(self.aggregation_levels))
        ]

        # Variance of our estimate of the mean v[-,s,g,n]
        self.variance = [
            defaultdict(float) for g in range(len(self.aggregation_levels))
        ]

        # Total variation (variance plus the square of the bias)
        self.total_variation = [
            defaultdict(float) for g in range(len(self.aggregation_levels))
        ]

        # Set up weight track to initial conditions
        self.reset_weight_track()

    def reset_weight_track(self):
        self.counts = 0
        self.weight_track = defaultdict(
            lambda: np.zeros(len(self.aggregation_levels))
        )

    def update_weight_track(self, weight_vector, key="V"):
        # Update weight vector
        self.weight_track[key] = (
                                         self.counts * self.weight_track[key] + weight_vector
                                 ) / (self.counts + 1)

        self.counts += 1

    ####################################################################
    # Smoothed #########################################################
    ####################################################################

    def get_weight(self, g, a, vf_0):

        # WEIGHTING ############################################

        # Bias due to aggregation error = v[-,a, g] - v[-, a, 0]
        aggregation_bias = self.values[g][a] - vf_0

        # Bias due to smoothing of transient data series (value
        # function change every iteration)
        transient_bias = self.transient_bias[g][a]

        variance_g = self.variance_g[g][a]

        # Lambda stepsize from iteration n-1
        lambda_step_size = self.lambda_stepsize[g][a]

        # Estimate of the variance of observations made of state
        # s, using data from aggregation level g, after n
        # observations.
        variance_error = self.get_total_variance(
            variance_g, transient_bias, lambda_step_size
        )

        # Variance of our estimate of the mean v[-,s,g,n]
        variance = lambda_step_size * variance_error

        # Total variation (variance plus the square of the bias)
        total_variation = variance + (aggregation_bias ** 2)

        if total_variation == 0:
            return 0
        else:
            return 1 / total_variation

    def get_weighted_value(self, t, pos, battery):

        # Post decision attribute at time post_t
        a = (pos, battery)

        # Get point object associated to position
        point = self.points[pos]

        value_estimation = 0
        # Check if value function for disaggregate level exists
        if t in self.values and 0 in self.values[t] and a in self.values[t][0]:
            # Return previosly defined disaggregated value function
            return self.values[t][0][a]

        # Calculate value estimation based on hierarchical aggregation
        else:

            # Get va
            weight_vector = np.zeros(self.aggregation_levels - 1)
            value_vector = np.zeros(self.aggregation_levels - 1)

            for g in range(1, self.aggregation_levels):
                pos_g = point.id_level(g)
                # Find attribute at level g
                a_g = (pos_g, battery)

                # Current value function of attribute at level g
                value_vector[g - 1] = (
                    self.values[t][g][a_g]
                    if t in self.values
                       and 0 in self.values[t]
                       and a_g in self.values[t][g]
                    else 0
                )

                weight_vector[g - 1] = self.get_weight(t, g, a_g)

            # Normalize (weights have to sum up to one)
            weight_sum = sum(weight_vector)

            if weight_sum > 0:
                weight_vector = weight_vector / weight_sum

                # Get weighted value function
                value_estimation = sum(
                    np.prod([weight_vector, value_vector], axis=0)
                )

                # Update weight vector
                self.agg_weight_vectors[(t, pos, battery)] = weight_vector

        return value_estimation

    def get_total_variance(
            self, total_variation, transient_bias, lambda_stepsize
    ):
        r = (total_variation - (transient_bias ** 2)) / (1 + lambda_stepsize)

        return r

    def get_variance_g(self, dif_vfs, fixed_stepsize, variance_tag):

        # We now need to compute s^2[a,g] which is the estimate of the
        # variance of observations (v) for states (a) for which
        # G(a) = a_g (the observations of states that aggregate up
        # to a).

        r = (1 - fixed_stepsize) * variance_tag + fixed_stepsize * (
                dif_vfs ** 2
        )

        return r

    def get_lambda_stepsize(self, current_stepsize, lambda_stepsize):

        r = (((1 - current_stepsize) ** 2) * lambda_stepsize) + (
                current_stepsize ** 2
        )

        return r

    def get_transient_bias(self, current_bias, dif_vfs, stepsize):

        # The transient bias (due to smoothing): When we smooth on past
        # observations, we obtain an estimate v[-,s,g,n-1] that tends to
        #  underestimate (or overestimate if v(^,n) tends to decrease)
        # the true mean of v[^,n].
        r = (1 - stepsize) * current_bias + stepsize * dif_vfs

        return r

    def get_weights(self, steps):

        try:
            weight_vector_sum = sum(self.agg_weight_vectors.values())
            avg_agg_levels = weight_vector_sum / sum(weight_vector_sum)

        except:
            return np.zeros(len(self.aggregation_levels))

        return avg_agg_levels

    def get_stepsize(self, previous_stepsize):
        """Return a stepsize to update value functions according to
        the stepsize rule"""

        if self.stepsize_rule == STEPSIZE_HARMONIC:
            # Generalized harmonic stepsize
            # Notice that a_stepsize is 1 when count is zero
            stepsize = self.stepsize_harmonic / (
                    self.stepsize_harmonic + max(1, self.n) - 1
            )

        # Fixed value passed as parameter
        elif self.stepsize_rule == STEPSIZE_CONSTANT:

            stepsize = self.stepsize_constant

        # McClain’s stepsize rule (stop at stepsize_constant)
        elif self.stepsize_rule == STEPSIZE_MCCLAIN:
            stepsize = previous_stepsize / (
                    1 + previous_stepsize - self.stepsize_constant
            )
        else:
            stepsize = 1 / self.n

        return stepsize

    # # @functools.lru_cache(maxsize=None)
    def get_state(self, g, disaggregate):

        level = self.aggregation_levels[g]
        # Get point object associated to position

        # Time in level g (g_time, g_time(t))
        t_g = self.time_step_level(disaggregate[TIME], level=level[TIME])

        # Position in level g
        point = self.points[disaggregate[LOCATION]]
        pos_g = point.id_level(level[LOCATION])

        contract_duration_g = self.contract_level(
            disaggregate[CARTYPE],
            disaggregate[CONTRACT],
            level=level[CONTRACT],
        )

        # Get car type at current level
        car_type_g = self.car_type_level(
            disaggregate[CARTYPE], level=level[CARTYPE]
        )

        # Get car origin at current level
        car_origin_g = self.car_origin_level(
            disaggregate[CARTYPE], disaggregate[ORIGIN], level=level[ORIGIN]
        )

        # Find attribute at level g
        state_g = (
            t_g,
            pos_g,
            disaggregate[BATTERY],
            contract_duration_g,
            car_type_g,
            car_origin_g,
        )

        return state_g

    def update_values_smoothed(self, t, duals):

        # List of duals associated to tuples (level g, attribute[g])
        # The new value of an aggregate level correspond to the average
        # of these duals
        level_update_list = defaultdict(list)

        for a, v_ta in duals.items():

            pos, battery = a

            # Updating value function at disaggregate level
            self.values[t][0][a] = (
                                           1 - self.step_size_func[t][0][a]
                                   ) * v_ta + self.step_size_func[t][0][a] * v_ta

            # Update the number of times disaggregate state was accessed
            self.count[t][0][a] += 1

            # Get point object associated to position
            point = self.points[pos]

            # Append duals to all superior hierachical states
            for g in range(1, self.aggregation_levels):
                # Find attribute at level g
                a_g = (point.id_level(g), battery)

                # Value is later used to update a_g
                level_update_list[(g, a_g)].append(v_ta)

        for state_g, value_list_g in level_update_list.items():
            g, a_g = state_g

            # Number of times state g was accessed
            count_ta_g = len(value_list_g)

            # Average value function considering all elements sharing
            # the same state at level g
            v_ta_g = sum(value_list_g) / count_ta_g

            self.update_weights(t, g, a_g, v_ta_g, count_ta_g)

            # Updating value function at gth level with smoothing
            self.values[t][g][a_g] = (
                    (1 - self.step_size_func[t][g][a_g]) * self.values[t][g][a_g]
                    + self.step_size_func[t][g][a_g] * v_ta_g
            )

    ####################################################################
    # True averaging ###################################################
    ####################################################################

    def get_value(self, t, pos, battery, level=0):

        # Point associated to position at disaggregate level
        point = self.points[pos]

        # Attribute considering aggregation level
        attribute = (point.id_level(level), battery)

        # Value function
        value = self.values[t][level][attribute]

        return value

    def averaged_update(self, t, duals):
        """Update values without smoothing

        Arguments:
            step {int} -- Current time step
            duals {dict} -- Dictionary of attribute tuples and duals
        """

        for (pos, battery), new_vf_0 in duals.items():

            # Get point object associated to position
            point = self.points[pos]

            for g in range(self.aggregation_levels):
                # Find attribute at level g
                a_g = (point.id_level(g), battery)

                # Current value function of attribute at level g
                current_vf = self.values[t][g][a_g]

                # Increment every time attribute ta_g is accessed
                self.count[t][g][a_g] += 1

                # Incremental averaging
                count_ta_g = self.count[t][g][a_g]
                increment = (new_vf_0 - current_vf) / count_ta_g

                # Update attribute mean value
                self.values[t][g][a_g] += increment

        # Update weights using new value function estimate
        # self.update_weights(t, g, a_g, new_vf_0, 1)

    ####################################################################
    # Save/Load ########################################################
    ####################################################################

    def load_progress(self, progress):
        self.values = progress["values"]
        self.count = progress["counts"]
        self.transient_bias = progress["transient_bias"]
        self.variance_g = progress["variance_g"]
        self.step_size_func = progress["stepsize"]
        self.lambda_stepsize = progress["lambda_stepsize"]
        self.aggregation_bias = progress["aggregation_bias"]

    def read_progress(self, path):
        """Load episodes learned so farD

        Returns:
            values, counts -- Value functions and count per aggregation
                level.
        """

        progress = np.load(path, allow_pickle=True).item()

        self.n = progress.get("episodes", list())
        self.reward = progress.get("reward", list())
        self.pk_delay = progress.get("pk_delay", list())
        self.car_time = progress.get("car_time", list())
        self.service_rate = progress.get("service_rate", list())
        self.weights = progress.get("weights", list())

        print(
            f"\n### Loading {self.n} episodes from '{path}'."
            f"\n -       Last reward: {self.reward[self.n - 1]:15.2f} "
            f"(max={max(self.reward):15,.2f})"
            f"\n - Last service rate: {self.service_rate[self.n - 1]:15.2%} "
            f"(max={max(self.service_rate):15.2%})\n"
            f"\n - Last pickup delay: {self.pk_delay[self.n - 1]}"
            # TODO
            # f"\n -    Last car times: {self.car_time[self.n-1]}"
        )

        for g in range(len(self.aggregation_levels)):
            for a, saved in progress["progress"][g].items():
                v, c, t_bias, variance, step, lam, agg_bias = saved
                self.values[g][a] = v
                self.count[g][a] = c
                self.transient_bias[g][a] = t_bias
                self.variance_g[g][a] = variance
                self.step_size_func[g][a] = step
                self.lambda_stepsize[g][a] = lam
                self.aggregation_bias[g][a] = agg_bias

        return self.n, self.reward, self.service_rate, self.weights

    @property
    def current_data(self):

        adp_data = [
            {
                a: (
                    value,
                    self.count[g][a],
                    self.transient_bias[g][a],
                    self.variance_g[g][a],
                    self.step_size_func[g][a],
                    self.lambda_stepsize[g][a],
                    self.aggregation_bias[g][a],
                )
                for a, value in a_value.items()
            }
            for g, a_value in enumerate(self.values)
        ]

        return adp_data

    @property
    def current_data_np(self):

        adp_data = {
            tuple(t)
            + (g,)
            + tuple(a): np.array(
                [
                    value,
                    self.count[t][g][a],
                    self.transient_bias[t][g][a],
                    self.variance_g[t][g][a],
                    self.step_size_func[t][g][a],
                    self.lambda_stepsize[t][g][a],
                    self.aggregation_bias[t][g][a],
                ]
            )
            for t, g_a in self.values.items()
            for g, a_value in g_a.items()
            for a, value in a_value.items()
            if self.count[t][g][a] > 0
        }

        return adp_data

    @property
    def current_data_np2(self):

        adp_data = np.array(
            [
                [
                    *(t + (g,) + a),
                    value,
                    self.count[t][g][a],
                    self.transient_bias[t][g][a],
                    self.variance_g[t][g][a],
                    self.step_size_func[t][g][a],
                    self.lambda_stepsize[t][g][a],
                    self.aggregation_bias[t][g][a],
                ]
                for t, g_a in self.values.items()
                for g, a_value in g_a.items()
                for a, value in a_value.items()
                if self.count[t][g][a] > 0
            ]
        )

        return adp_data

    def load_episode(self, path, label):

        """Load .npy dictionary containing value functions of last
        episode.

        Arguments:
            path {str} -- File with saved value functions
        """
        values_old = np.load(path + label + ".npy", allow_pickle=True).item()
        # print(values_old)
        for t, g_a in values_old.items():
            for g, a_value in g_a.items():

                # Time in level g
                t_g = self.time_step_level(t, level=g)

                for a, value in a_value.items():
                    self.values[t_g][g][a] = value

    @functools.lru_cache(maxsize=None)
    def time_step_level(self, t, level=DISAGGREGATE):
        """Time steps in minutes"""
        # Since t start from 1, t-1 guarantee first slice is of size 2
        g_t = (t - 1) // self.temporal_levels[level]
        return g_t

    @functools.lru_cache(maxsize=None)
    def car_origin_level(self, car_type, car_origin, level=DISAGGREGATE):

        try:
            spatial_level = self.car_origin_levels_dict[car_type][level]
            point = self.points[car_origin]

            # Find attribute at level g
            origin_g = point.id_level(spatial_level)

            return origin_g

        except:
            return DISCARD

    @functools.lru_cache(maxsize=None)
    def car_type_level(self, car_type, level=DISAGGREGATE):

        if level == DISAGGREGATE:
            return car_type

        # TODO Calculate for different levels

        return self.car_type_levels_dict[car_type][DISCARD]

    @functools.lru_cache(maxsize=None)
    def contract_level(self, car_type, contract_duration, level=DISAGGREGATE):

        try:
            time_slot = self.contract_levels_dict[car_type][level]
            return contract_duration // time_slot

        except:
            return self.contract_levels_dict[car_type][DISCARD]
