import numpy as np
from collections import defaultdict
import mod.env.adp.adp as adp
import mod.util.log_util as la
from pprint import pprint
from mod.env.adp import AdpHired

np.set_printoptions(precision=4)

VF = 0
COUNT = 1
TRANSIENT_BIAS = 2
VARIANCE_G = 3
STEPSIZE_FUNC = 4
LAMBDA_STEPSIZE = 5

TIME = 0
LOCATION = 1
BATTERY = 2
CONTRACT = 3
CARTYPE = 4
ORIGIN = 5

attributes = [TIME, LOCATION, BATTERY, CONTRACT, CARTYPE, ORIGIN]


class AdpHiredSimple(adp.Adp):
    def __init__(self, points, config):

        self.config = config

        aggregation_levels = self.config.aggregation_levels
        temporal_levels = self.config.level_time_list
        car_type_levels = self.config.level_car_type_dict
        contract_levels = self.config.level_contract_duration_dict
        car_origin_levels = self.config.level_car_origin_dict
        stepsize = self.config.stepsize
        stepsize_rule = self.config.stepsize_rule
        stepsize_constant = self.config.stepsize_constant
        stepsize_harmonic = self.config.stepsize_harmonic

        super().__init__(
            points,
            aggregation_levels,
            temporal_levels,
            car_type_levels,
            contract_levels,
            car_origin_levels,
            stepsize,
            stepsize_rule=stepsize_rule,
            stepsize_constant=stepsize_constant,
            stepsize_harmonic=stepsize_harmonic,
        )

        self.data = defaultdict(lambda: np.array([0, 0, 0, 0, 1, 1]))

    def get_state(self, g, disaggregate):

        level = self.aggregation_levels[g]
        # Get point object associated to position

        # Time in level g (g_time, g_time(t))
        t_g = self.time_step_level(
            disaggregate[adp.TIME], level=level[adp.TIME]
        )

        # Position in level g
        point = self.points[disaggregate[adp.LOCATION]]
        pos_g = point.id_level(level[adp.LOCATION])

        contract_duration_g = self.contract_level(
            disaggregate[adp.CARTYPE],
            disaggregate[adp.CONTRACT],
            level=level[adp.CONTRACT],
        )

        # Get car type at current level
        car_type_g = self.car_type_level(
            disaggregate[adp.CARTYPE], level=level[adp.CARTYPE]
        )

        # Get car origin at current level
        car_origin_g = self.car_origin_level(
            disaggregate[adp.CARTYPE],
            disaggregate[adp.ORIGIN],
            level=level[adp.ORIGIN],
        )

        # Find attribute at level g
        state_g = (
            g,
            t_g,
            pos_g,
            disaggregate[adp.BATTERY],
            contract_duration_g,
            car_type_g,
            car_origin_g,
        )

        return state_g

    def get_weighted_value(self, disaggregate):

        value_estimation = 0

        # Calculate value estimation based on hierarchical aggregation
        weight_vector = np.zeros(len(self.aggregation_levels))
        value_vector = np.zeros(len(self.aggregation_levels))

        state_0 = self.get_state(0, disaggregate)
        value_vector[0] = self.data[state_0][VF]
        weight_vector[0] = self.get_weight(state_0, state_0)

        for g in range(len(self.aggregation_levels) - 1, 0, -1):

            # Get state at level g from disaggregate data
            state_g = self.get_state(g, disaggregate)

            # if superior level is zero, inferior levels are also zero
            # if state_g not in self.data:
            #     break

            value_vector[g] = self.data[state_g][VF]

            weight_vector[g] = self.get_weight(state_g, state_0)

        # Normalize (weights have to sum up to one)
        weight_sum = sum(weight_vector)

        if weight_sum > 0:

            weight_vector = weight_vector / weight_sum

            # Get weighted value function
            value_estimation = sum(
                np.prod([weight_vector, value_vector], axis=0)
            )

            self.update_weight_track(
                weight_vector, key=disaggregate[adp.CARTYPE]
            )

            # la.log_weights(
            #     self.config.log_path(self.n),
            #     (t, pos, battery, contract_duration, car_type, car_origin),
            #     weight_vector,
            #     value_vector,
            #     value_estimation,
            # )

        return value_estimation

    def update_vf(self, state_g, v_ta_g):
        # Updating lambda stepsize using previous stepsizes
        self.data[state_g][LAMBDA_STEPSIZE] = self.get_lambda_stepsize(
            self.data[state_g][STEPSIZE_FUNC],
            self.data[state_g][LAMBDA_STEPSIZE],
        )

        # Updating value function at gth level with smoothing
        old_v_ta_g = self.data[state_g][VF]
        stepsize = self.data[state_g][STEPSIZE_FUNC]
        new_v_ta_g = (1 - stepsize) * old_v_ta_g + stepsize * v_ta_g
        self.data[state_g][VF] = new_v_ta_g

        # Updates ta_g stepsize
        self.data[state_g][STEPSIZE_FUNC] = self.get_stepsize(
            self.data[state_g][STEPSIZE_FUNC]
        )

    def update_values_smoothed_single(self, t, duals):

        # List of duals associated to tuples (level g, attribute[g])
        # The new value of an aggregate level correspond to the average
        # of these duals
        # level_update_list = defaultdict(list)

        for car_flow_attribute, v_ta_sampled in duals.items():

            disaggregate = (t,) + car_flow_attribute

            # Append duals to all superior hierachical states
            for g in range(len(self.aggregation_levels)):

                # Value is later used to update a_g
                state_g = self.get_state(g, disaggregate)

                # level_update_list[state_g].append(v_ta_sampled)

                # Update the number of times state was accessed
                self.data[state_g][COUNT] += 1

                # Bias due to smoothing of transient data series
                # (value function change every iteration)
                self.data[state_g][TRANSIENT_BIAS] = self.get_transient_bias(
                    self.data[state_g][TRANSIENT_BIAS],
                    v_ta_sampled,
                    self.data[state_g][VF],
                    self.stepsize,
                )

                # Estimate of total squared variation,
                self.data[state_g][VARIANCE_G] = self.get_variance_g(
                    v_ta_sampled,
                    self.data[state_g][VF],
                    self.stepsize,
                    self.data[state_g][VARIANCE_G],
                )

                self.update_vf(state_g, v_ta_sampled)

    def update_values_smoothed(self, t, duals):

        # List of duals associated to tuples (level g, attribute[g])
        # The new value of an aggregate level correspond to the average
        # of these duals
        level_update_list = defaultdict(list)

        for car_flow_attribute, v_ta_sampled in duals.items():

            disaggregate = (t,) + car_flow_attribute

            # Append duals to all superior hierachical states
            for g in range(len(self.aggregation_levels)):

                # Value is later used to update a_g
                state_g = self.get_state(g, disaggregate)

                level_update_list[state_g].append(v_ta_sampled)

                # Update the number of times state was accessed
                self.data[state_g][COUNT] += 1

                # Bias due to smoothing of transient data series
                # (value function change every iteration)
                self.data[state_g][TRANSIENT_BIAS] = self.get_transient_bias(
                    self.data[state_g][TRANSIENT_BIAS],
                    v_ta_sampled,
                    self.data[state_g][VF],
                    self.stepsize,
                )

                # Estimate of total squared variation,
                self.data[state_g][VARIANCE_G] = self.get_variance_g(
                    v_ta_sampled,
                    self.data[state_g][VF],
                    self.stepsize,
                    self.data[state_g][VARIANCE_G],
                )

        # Loop states (including disaggregate), average all values that
        # aggregate up to ta_g, and smooth average to previous value
        for state_g, value_list_g in level_update_list.items():

            # Updating lambda stepsize using previous stepsizes
            self.data[state_g][LAMBDA_STEPSIZE] = self.get_lambda_stepsize(
                self.data[state_g][STEPSIZE_FUNC],
                self.data[state_g][LAMBDA_STEPSIZE],
            )

            # Average value function considering all elements sharing
            # the same state at level g
            v_ta_g = sum(value_list_g) / len(value_list_g)

            # Updating value function at gth level with smoothing
            old_v_ta_g = self.data[state_g][VF]
            stepsize = self.data[state_g][STEPSIZE_FUNC]
            new_v_ta_g = (1 - stepsize) * old_v_ta_g + stepsize * v_ta_g
            self.data[state_g][VF] = new_v_ta_g

            # Updates ta_g stepsize
            self.data[state_g][STEPSIZE_FUNC] = self.get_stepsize(
                self.data[state_g][STEPSIZE_FUNC]
            )

        # Log how duals are updated
        la.log_update_values_smoothed(
            self.config.log_path(self.n), t, level_update_list, self.values
        )

    def get_weight(self, state_g, state_0):

        # WEIGHTING ############################################

        # Bias due to aggregation error = v[-,a, g] - v[-, a, 0]
        aggregation_bias = self.data[state_g][VF] - self.data[state_0][VF]

        # Bias due to smoothing of transient data series (value
        # function change every iteration)
        transient_bias = self.data[state_g][TRANSIENT_BIAS]

        variance_g = self.data[state_g][VARIANCE_G]

        # Lambda stepsize from iteration n-1
        lambda_step_size = self.data[state_g][LAMBDA_STEPSIZE]

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
