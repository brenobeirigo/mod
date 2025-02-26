import numpy as np
from collections import defaultdict
import mod.env.adp.adp as adp
import mod.util.log_util as la
from pprint import pprint
import functools

np.set_printoptions(precision=4)


class AdpHired(adp.Adp):
    def __init__(self, points, config):

        self.config = config

        agregation_levels = self.config.aggregation_levels
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
            agregation_levels,
            temporal_levels,
            car_type_levels,
            contract_levels,
            car_origin_levels,
            stepsize,
            stepsize_rule=stepsize_rule,
            stepsize_constant=stepsize_constant,
            stepsize_harmonic=stepsize_harmonic,
        )

        self.weighted_values = dict()

    ####################################################################
    # Smoothed #########################################################
    ####################################################################

    # # @functools.lru_cache(maxsize=None)
    def get_weighted_value(self, disaggregate):

        value_estimation, weight_vector = self.weighted_values.get(
            disaggregate, (0, np.zeros(len(self.aggregation_levels)))
        )
        if value_estimation > 0:
            self.update_weight_track(
                weight_vector, key=disaggregate[adp.CARTYPE]
            )
            return value_estimation

        # Calculate value estimation based on hierarchical aggregation
        # weight_vector = np.zeros(len(self.aggregation_levels))
        value_vector = np.zeros(len(self.aggregation_levels))

        a_0 = self.get_state(0, disaggregate)

        vf_0 = self.values[0].get(a_0, 0)

        for g in reversed(range(len(self.aggregation_levels))):

            a_g = self.get_state(g, disaggregate)

            if a_g not in self.values[g]:
                break

            value_vector[g] = self.values[g][a_g]

            # if value_vector[g] == 0:
            #     break

            weight_vector[g] = self.get_weight(g, a_g, vf_0)

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
        self.weighted_values[disaggregate] = (value_estimation, weight_vector)
        return value_estimation

    def update_values_smoothed(self, step, duals):

        # List of duals associated to tuples (level g, attribute[g])
        # The new value of an aggregate level correspond to the average
        # of these duals
        # level_update_list = defaultdict(list)
        level_update = defaultdict(lambda: np.zeros(2))

        for a, v_ta_sampled in duals.items():

            disaggregate = (step,) + a

            # Append duals to all superior hierachical states
            for g in range(len(self.aggregation_levels)):

                a_g = self.get_state(g, disaggregate)

                # Value is later used to update a_g
                # level_update_list[(g, a_g)].append(v_ta_sampled)
                g_a = (g, a_g)
                level_update[g_a][0] += v_ta_sampled
                level_update[g_a][1] += 1

                # Update the number of times state was accessed
                self.count[g][a_g] += 1

                # Bias due to smoothing of transient data series
                # (value function change every iteration)
                self.transient_bias[g][a_g] = self.get_transient_bias(
                    self.transient_bias[g][a_g],
                    v_ta_sampled,
                    self.values[g].get(a_g, 0),
                    self.stepsize,
                )

                # Estimate of total squared variation,
                self.variance_g[g][a_g] = self.get_variance_g(
                    v_ta_sampled,
                    self.values[g].get(a_g, 0),
                    self.stepsize,
                    self.variance_g[g][a_g],
                )

        # Loop states (including disaggregate), average all values that
        # aggregate up to ta_g, and smooth average to previous value
        for state_g, value_count_g in level_update.items():

            g, a_g = state_g

            # Updating lambda stepsize using previous stepsizes
            self.lambda_stepsize[g][a_g] = self.get_lambda_stepsize(
                self.step_size_func[g][a_g], self.lambda_stepsize[g][a_g]
            )

            # Average value function considering all elements sharing
            # the same state at level g
            v_ta_g = value_count_g[0] / value_count_g[1]

            # Updating value function at gth level with smoothing
            old_v_ta_g = self.values[g].get(a_g, 0)
            stepsize = self.step_size_func[g][a_g]
            new_v_ta_g = (1 - stepsize) * old_v_ta_g + stepsize * v_ta_g
            self.values[g][a_g] = new_v_ta_g

            # Updates ta_g stepsize
            self.step_size_func[g][a_g] = self.get_stepsize(
                self.step_size_func[g][a_g]
            )

        # Log how duals are updated
        la.log_update_values_smoothed(
            self.config.log_path(self.n), step, [], self.values
        )

    ####################################################################
    # True averaging ###################################################
    ####################################################################

    def get_value(self, t, pos, battery, contract_duration, car_type, level=0):

        # Point associated to position at disaggregate level
        point = self.points[pos]

        # Time in level g
        t_g = self.time_step_level(t, level=level)

        # Attribute considering aggregation level
        attribute = (
            point.id_level(level),
            battery,
            contract_duration,
            car_type,
        )

        # Value function
        value = self.values[t_g][level][attribute]

        return value

    def averaged_update(self, t, duals):
        """Update values without smoothing

        Arguments:
            step {int} -- Current time step
            duals {dict} -- Dictionary of attribute tuples and duals
        """

        # pprint(duals)

        for a, v_ta in duals.items():

            pos, battery, contract_duration, car_type = a

            # Get point object associated to position
            point = self.points[pos]

            for g_time, g in self.aggregation_levels:

                # Time in level g
                t_g = self.time_step_level(t, level=g_time)

                # Find attribute at level g
                a_g = (point.id_level(g), battery, contract_duration, car_type)

                # Current value function of attribute at level g
                current_vf = self.values[t_g][g][a_g]

                # Increment every time attribute ta_g is accessed
                self.count[t_g][g][a_g] += 1

                # Incremental averaging
                count_ta_g = self.count[t_g][g][a_g]
                increment = (v_ta - current_vf) / count_ta_g

                # Update attribute mean value
                self.values[t_g][g][a_g] += increment

        # Update weights using new value function estimate
        # self.update_weights(t, g, a_g, new_vf_0, 1)

    @property
    def current_data(self):

        adp_data = {
            g: {
                a: (
                    self.values[g][a],
                    self.count[g][a],
                    self.transient_bias[g][a],
                    self.variance_g[g][a],
                    self.step_size_func[g][a],
                    self.lambda_stepsize[g][a],
                    self.aggregation_bias[g][a],
                )
                for a in self.values[g]
            }
            for g in range(len(self.aggregation_levels))
        }

        return adp_data
