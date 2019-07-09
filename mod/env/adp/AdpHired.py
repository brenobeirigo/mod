import numpy as np
from collections import defaultdict
import mod.env.adp.adp as adp


class AdpHired(adp.Adp):
    def __init__(
        self,
        points,
        agregation_levels,
        stepsize,
        stepsize_rule=adp.STEPSIZE_CONSTANT,
        stepsize_constant=0.1,
        stepsize_harmonic=1,
    ):

        super().__init__(
            points,
            agregation_levels,
            stepsize,
            stepsize_rule=stepsize_rule,
            stepsize_constant=stepsize_constant,
            stepsize_harmonic=stepsize_harmonic,
        )

    ####################################################################
    # Smoothed #########################################################
    ####################################################################

    def get_weights_and_agg_value(self, t, pos, battery, contract_duration=32):
        # Get point object associated to position
        point = self.points[pos]

        # Value function of level 0 in previous iteration
        v_ta_0 = (
            self.values[t][0][(pos, battery)]
            if t in self.values
            and 0 in self.values[t]
            and (pos, battery, contract_duration) in self.values[t][0]
            else 0
        )

        weight_vector = np.zeros(self.aggregation_levels)
        value_vector = np.zeros(self.aggregation_levels)

        for g in range(self.aggregation_levels):

            # Find attribute at level g
            ta_g = (point.id_level(g), battery, contract_duration)

            # Current value function of attribute at level g
            value_vector[g] = (
                self.values[t][g][ta_g]
                if t in self.values
                and 0 in self.values[t]
                and (pos, battery, contract_duration) in self.values[t][0]
                else 0
            )

            # WEIGHTING ############################################

            # Bias due to aggregation error = v[-,a, g] - v[-, a, 0]
            aggregation_bias = self.aggregation_bias[t][g][ta_g]

            # Bias due to smoothing of transient data series (value
            # function change every iteration)
            transient_bias = self.transient_bias[t][g][ta_g]

            variance_g = self.variance_g[t][g][ta_g]

            # Lambda stepsize from iteration n-1
            lambda_step_size = self.lambda_stepsize[t][g][ta_g]

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
                weight_vector[g] = 0
            else:
                weight_vector[g] = 1 / total_variation

        if len(np.unique(value_vector)) <= 1:
            value_estimation = 0
        else:
            weight_vector = weight_vector / sum(weight_vector)
            value_estimation = sum(
                np.prod([weight_vector, value_vector], axis=0)
            )
            # Upate weight vector
            # self.agg_weight_vectors[(t, pos, battery)] = weight_vector

        return weight_vector, value_estimation

    def get_weighted_value(self, t, pos, battery, contract_duration, car_type):

        # Get point object associated to position
        point = self.points[pos]

        value_estimation = 0

        # Calculate value estimation based on hierarchical aggregation
        weight_vector = np.zeros(self.aggregation_levels)
        value_vector = np.zeros(self.aggregation_levels)

        for g in range(0, self.aggregation_levels):

            pos_g = point.id_level(g)
            # Find attribute at level g
            a_g = (pos_g, battery, contract_duration, car_type)

            # Current value function of attribute at level g
            value_vector[g] = (
                self.values[t][g][a_g]
                if t in self.values
                and 0 in self.values[t]
                and a_g in self.values[t][g]
                else 0
            )

            weight_vector[g] = self.get_weight(t, g, a_g)

        # Normalize (weights have to sum up to one)
        weight_sum = sum(weight_vector)

        if weight_sum > 0:

            weight_vector = weight_vector / weight_sum

            # Get weighted value function
            value_estimation = sum(
                np.prod([weight_vector, value_vector], axis=0)
            )

            # Update weight vector
            self.agg_weight_vectors[
                (t, pos, battery, contract_duration, car_type)
            ] = weight_vector

        return value_estimation

    def update_values_smoothed(self, t, duals):

        # List of duals associated to tuples (level g, attribute[g])
        # The new value of an aggregate level correspond to the average
        # of these duals
        level_update_list = defaultdict(list)

        for a, v_ta in duals.items():

            pos, battery, contract_duration, car_type = a

            # Get point object associated to position
            point = self.points[pos]

            # Append duals to all superior hierachical states
            for g in range(0, self.aggregation_levels):

                # Find attribute at level g
                a_g = (point.id_level(g), battery, contract_duration, car_type)

                # Value is later used to update a_g
                level_update_list[(g, a_g)].append(v_ta)

        for state_g, value_list_g in level_update_list.items():

            g, a_g = state_g

            # Number of times state g was accessed
            count_ta_g = len(value_list_g)

            # Average value function considering all elements sharing
            # the same state at level g
            # TODO Test if looping to all values and updating makes more sense (instead of average)
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

    def get_value(self, t, pos, battery, contract_duration, car_type, level=0):

        # Point associated to position at disaggregate level
        point = self.points[pos]

        # Attribute considering aggregation level
        attribute = (
            point.id_level(level),
            battery,
            contract_duration,
            car_type,
        )

        # Value function
        value = self.values[t][level][attribute]

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

            for g in range(self.aggregation_levels):

                # Find attribute at level g
                a_g = (point.id_level(g), battery, contract_duration, car_type)

                # Current value function of attribute at level g
                current_vf = self.values[t][g][a_g]

                # Increment every time attribute ta_g is accessed
                self.count[t][g][a_g] += 1

                # Incremental averaging
                count_ta_g = self.count[t][g][a_g]
                increment = (v_ta - current_vf) / count_ta_g

                # Update attribute mean value
                self.values[t][g][a_g] += increment

        # Update weights using new value function estimate
        # self.update_weights(t, g, a_g, new_vf_0, 1)

    # ################################################################ #
    # Tracking ####################################################### #
    # ################################################################ #

    def get_weights(self):

        fleet_weights_dict = defaultdict(list)
        fleet_weights_avg_dict = defaultdict(
            lambda: np.zeros(self.aggregation_levels)
        )

        try:
            for attribute, weight_vectors in self.agg_weight_vectors.items():
                _, _, _, _, car_type = attribute
                fleet_weights_dict[car_type].append(weight_vectors)

            for fleet_type, weight_vectors_list in fleet_weights_dict.items():
                fleet_weights_avg_dict[fleet_type] = sum(
                    weight_vectors_list
                ) / len(weight_vectors_list)

        except:
            pass

        return fleet_weights_avg_dict
