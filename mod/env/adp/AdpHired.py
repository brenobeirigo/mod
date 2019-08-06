import numpy as np
from collections import defaultdict
import mod.env.adp.adp as adp
import mod.util.log_aux as la
from pprint import pprint

np.set_printoptions(precision=4)


class AdpHired(adp.Adp):
    def __init__(
        self,
        points,
        agregation_levels,
        temporal_levels,
        car_type_levels,
        contract_levels,
        car_origin_levels,
        stepsize,
        stepsize_rule=adp.STEPSIZE_CONSTANT,
        stepsize_constant=0.1,
        stepsize_harmonic=1,
        logger_name=None,
    ):

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

        self.logger_name = logger_name

    ####################################################################
    # Smoothed #########################################################
    ####################################################################

    def get_weighted_value(
        self, t, pos, battery, contract_duration, car_type, car_origin
    ):

        state = (t, pos, battery, contract_duration, car_type, car_origin)

        # Get point object associated to position
        point = self.points[pos]

        value_estimation = 0

        # Calculate value estimation based on hierarchical aggregation
        weight_vector = np.zeros(len(self.aggregation_levels))
        value_vector = np.zeros(len(self.aggregation_levels))

        for i, (g_time, g, g_contract, g_cartype, g_carorigin) in enumerate(
            self.aggregation_levels
        ):

            # Time in level g (g_time, g_time(t))
            t_g = self.time_step_level(t, level=g_time)
            contract_duration_g = self.contract_level(
                car_type, contract_duration, level=g_contract
            )

            # Get car type at current level
            car_type_g = self.car_type_level(car_type, level=g_cartype)

            # Get car origin at current level
            car_origin_g = self.car_origin_level(
                car_type, car_origin, level=g_carorigin
            )

            # Position in level g
            pos_g = point.id_level(g)

            # Find attribute at level g
            a_g = (
                pos_g,
                battery,
                contract_duration_g,
                car_type_g,
                car_origin_g,
            )

            # Current value function of attribute at level g
            value_vector[i] = (
                self.values[t_g][g][a_g]
                if t_g in self.values
                and g in self.values[t_g]
                and a_g in self.values[t_g][g]
                else 0
            )

            weight_vector[i] = self.get_weight(t_g, g, a_g)

        # Normalize (weights have to sum up to one)
        weight_sum = sum(weight_vector)

        if weight_sum > 0:

            weight_vector = weight_vector / weight_sum

            # Get weighted value function
            value_estimation = sum(
                np.prod([weight_vector, value_vector], axis=0)
            )

            # Update weight vector
            self.agg_weight_vectors[state] = weight_vector

            la.log_weights(
                self.logger_name,
                state,
                weight_vector,
                value_vector,
                value_estimation,
            )

        return value_estimation

    def update_values_smoothed(self, t, duals):

        # List of duals associated to tuples (level g, attribute[g])
        # The new value of an aggregate level correspond to the average
        # of these duals
        level_update_list = defaultdict(list)

        for a, v_ta_sampled in duals.items():

            pos, battery, contract_duration, car_type, car_origin = a

            # Get point object associated to position
            point = self.points[pos]

            # Append duals to all superior hierachical states
            for (
                g_time,
                g,
                g_contract,
                g_cartype,
                g_carorigin,
            ) in self.aggregation_levels:

                # Tuple t_g = (g_time, g_time(t))
                t_g = self.time_step_level(t, level=g_time)

                contract_duration_g = self.contract_level(
                    car_type, contract_duration, level=g_contract
                )
                car_type_g = self.car_type_level(car_type, level=g_cartype)

                car_origin_g = self.car_origin_level(
                    car_type, car_origin, level=g_carorigin
                )

                # Find attribute at level g
                a_g = (
                    point.id_level(g),
                    battery,
                    contract_duration_g,
                    car_type_g,
                    car_origin_g,
                )

                # Value is later used to update a_g
                level_update_list[(t_g, g, a_g)].append(v_ta_sampled)

                # Update the number of times state was accessed
                self.count[t_g][g][a_g] += 1

                # Bias due to smoothing of transient data series
                # (value function change every iteration)
                self.transient_bias[t_g][g][a_g] = self.get_transient_bias(
                    self.transient_bias[t_g][g][a_g],
                    v_ta_sampled,
                    self.values[t_g][g][a_g],
                    self.stepsize,
                )

                # Estimate of total squared variation,
                self.variance_g[t_g][g][a_g] = self.get_variance_g(
                    v_ta_sampled,
                    self.values[t_g][g][a_g],
                    self.stepsize,
                    self.variance_g[t_g][g][a_g],
                )

        # Loop states (including disaggregate), average all values that
        # aggregate up to ta_g, and smooth average to previous value
        for state_g, value_list_g in level_update_list.items():

            t_g, g, a_g = state_g

            # Updating lambda stepsize using previous stepsizes
            self.lambda_stepsize[t_g][g][a_g] = self.get_lambda_stepsize(
                self.step_size_func[t_g][g][a_g],
                self.lambda_stepsize[t_g][g][a_g],
            )

            # Average value function considering all elements sharing
            # the same state at level g
            v_ta_g = sum(value_list_g) / len(value_list_g)

            # Updating value function at gth level with smoothing
            old_v_ta_g = self.values[t_g][g][a_g]
            stepsize = self.step_size_func[t_g][g][a_g]
            new_v_ta_g = (1 - stepsize) * old_v_ta_g + stepsize * v_ta_g
            self.values[t_g][g][a_g] = new_v_ta_g

            # Updates ta_g stepsize
            self.step_size_func[t_g][g][a_g] = self.get_stepsize(
                self.step_size_func[t_g][g][a_g]
            )

        # Log how duals are updated
        la.log_update_values_smoothed(
            self.logger_name, t, level_update_list, self.values
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

    # ################################################################ #
    # Tracking ####################################################### #
    # ################################################################ #

    def get_weights(self):

        fleet_weights_dict = defaultdict(list)
        fleet_weights_avg_dict = defaultdict(
            lambda: np.zeros(len(self.aggregation_levels))
        )

        try:
            for attribute, weight_vectors in self.agg_weight_vectors.items():
                _, _, _, _, car_type, _ = attribute
                fleet_weights_dict[car_type].append(weight_vectors)

            for fleet_type, weight_vectors_list in fleet_weights_dict.items():

                weight_vector_sum = sum(weight_vectors_list)
                fleet_weights_avg_dict[fleet_type] = weight_vector_sum / sum(
                    weight_vector_sum
                )

        except:
            pass

        return fleet_weights_avg_dict
