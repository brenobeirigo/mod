import numpy as np
from collections import defaultdict


class Adp:
    def __init__(self, points, agregation_levels, stepsize, harmonic_stepsize):
        self.aggregation_levels = agregation_levels
        self.harmonic_stepsize = harmonic_stepsize
        self.stepsize = stepsize
        self.points = points

        # Adp track
        self.n = 0
        self.reward = list()
        self.service_rate = list()
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
        self.values = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # How many times a cell was actually accessed by a vehicle in
        # a certain region, aggregation level, and time
        self.count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Averaging weights each round
        self.counts = np.zeros(self.aggregation_levels)
        self.weight_track = np.zeros(self.aggregation_levels)

        self.current_weights = np.array([])

    def init_weighting_settings(self):
        # -------------------------------------------------------------#
        # Weighing #####################################################
        # -------------------------------------------------------------#

        # Transient bias
        self.transient_bias = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        self.variance_g = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        self.step_size_func = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        self.lambda_stepsize = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # Aggregation bias
        self.aggregation_bias = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        self.agg_weight_vectors = dict()
         # Estimate of the variance of observations made of state
        # s, using data from aggregation level g, after n
        # observations.
        self.variance_error = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # Variance of our estimate of the mean v[-,s,g,n]
        self.variance = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

        # Total variation (variance plus the square of the bias)
        self.total_variation = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

    ####################################################################
    # Smoothed #########################################################
    ####################################################################

    def get_weights_and_agg_value(self, t, pos, battery):
        # Get point object associated to position
        point = self.points[pos]

        # Value function of level 0 in previous iteration
        v_ta_0 = (
            self.values[t][0][(pos, battery)]
            if t in self.values
            and 0 in self.values[t]
            and (pos, battery) in self.values[t][0]
            else 0
        )

        weight_vector = np.zeros(self.aggregation_levels)
        value_vector = np.zeros(self.aggregation_levels)

        for g in range(self.aggregation_levels):

            # Find attribute at level g
            ta_g = (point.id_level(g), battery)

            # Current value function of attribute at level g
            value_vector[g] = (
                self.values[t][g][ta_g]
                if t in self.values
                and 0 in self.values[t]
                and (pos, battery) in self.values[t][0]
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

    def get_weight(self, t, g, a):

        # WEIGHTING ############################################

        # Bias due to aggregation error = v[-,a, g] - v[-, a, 0]
        aggregation_bias = self.aggregation_bias[t][g][a]

        # Bias due to smoothing of transient data series (value
        # function change every iteration)
        transient_bias = self.transient_bias[t][g][a]

        variance_g = self.variance_g[t][g][a]

        # Lambda stepsize from iteration n-1
        lambda_step_size = self.lambda_stepsize[t][g][a]

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
        return (total_variation - (transient_bias ** 2)) / (
            1 + lambda_stepsize
        )

    def get_variance_g(self, v, v_g, fixed_stepsize, variance_tag):

        # We now need to compute s^2[a,g] which is the estimate of the
        # variance of observations (v) for states (a) for which
        # G(a) = a_g (the observations of states that aggregate up
        # to a).

        return (1 - fixed_stepsize) * variance_tag + fixed_stepsize * (
            (v - v_g) ** 2
        )

    def get_lambda_stepsize(self, current_stepsize, lambda_stepsize):

        return (((1 - current_stepsize) ** 2) * lambda_stepsize) + (
            current_stepsize ** 2
        )

    def get_transient_bias(self, current_bias, v, v_g, stepsize):

        # The transient bias (due to smoothing): When we smooth on past
        # observations, we obtain an estimate v[-,s,g,n-1] that tends to
        #  underestimate (or overestimate if v(^,n) tends to decrease)
        # the true mean of v[^,n].
        transient_bias = (1 - stepsize) * current_bias + stepsize * (v - v_g)

        return transient_bias

    def get_weights(self, steps):

        # print("Calculating average weights")
        # avg_vec = np.zeros(self.aggregation_levels)
        # for t in range(1, steps+1):
        #     for point in self.points:
        #         p = point.id
        #         for battery in range(0,self.battery_levels+1):
        #             vector, value = self.get_weights_and_agg_value(t,p,battery)

        #             avg_vec += vector

        # return avg_vec/(steps*len(self.points)*self.battery_levels)

        try:
            avg_agg_levels = sum(self.agg_weight_vectors.values()) / len(
                self.agg_weight_vectors
            )
        except:
            return np.zeros(self.aggregation_levels)

        self.agg_weight_vectors = dict()

        return avg_agg_levels

    def update_weights(self, t, g, a_g, sampled_v, count_g):

        # WEIGHTING ################################################## #

        # Current value function of attribute at level g
        old_v_ta_g = self.values[t][g][a_g]

        # Updating

        # Bias due to smoothing of transient data series (value
        # function change every iteration)
        current_transient_bias = self.transient_bias[t][g][a_g]
        self.transient_bias[t][g][a_g] = self.get_transient_bias(
            current_transient_bias, sampled_v, old_v_ta_g, self.stepsize
        )

        # Bias due to aggregation error = v[-,a, g] - v[-, a, 0]
        self.aggregation_bias[t][g][a_g] = old_v_ta_g - sampled_v

        self.variance_g[t][g][a_g] = self.get_variance_g(
            sampled_v, old_v_ta_g, self.stepsize, self.variance_g[t][g][a_g]
        )

        # Updating lambda stepsize using previous stepsizes
        self.lambda_stepsize[t][g][a_g] = self.get_lambda_stepsize(
            self.step_size_func[t][g][a_g], self.lambda_stepsize[t][g][a_g]
        )

        # Update the number of times state was accessed
        self.count[t][g][a_g] += count_g

        # Generalized harmonic stepsize
        # Notice that a_stepsize is 1 when count is zero
        a_stepsize = self.harmonic_stepsize
        # stepsize = a_stepsize / (a_stepsize + self.count[t][g][a_g] - 1)
        stepsize = a_stepsize / (a_stepsize + max(1, self.n) - 1)
        self.step_size_func[t][g][a_g] = stepsize


        # Estimate of the variance of observations made of state
        # s, using data from aggregation level g, after n
        # observations.
        self.variance_error[t][g][a_g] = self.get_total_variance(
            self.variance_g[t][g][a_g],
            self.transient_bias[t][g][a_g],
            self.lambda_stepsize[t][g][a_g]
        )

        # Variance of our estimate of the mean v[-,s,g,n]
        self.variance[t][g][a_g] =  (
            self.lambda_stepsize[t][g][a_g] * self.variance_error[t][g][a_g]
        )

        # Total variation (variance plus the square of the bias)
        self.total_variation[t][g][a_g] = (
            self.variance[t][g][a_g] + (self.aggregation_bias[t][g][a_g] ** 2)
        )

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

        # pprint(duals)

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

        progress = np.load(path).item()

        self.n = progress.get("episodes", list())
        self.reward = progress.get("reward", list())
        self.service_rate = progress.get("service_rate", list())
        self.weights = progress.get("weights", list())

        print(
            f"\n### Loading {self.n} episodes from '{path}'."
            f"\n -       Last reward: {self.reward[self.n-1]:15,.2f} "
            f"(max={max(self.reward):15,.2f})"
            f"\n - Last service rate: {self.service_rate[self.n-1]:15.2%} "
            f"(max={max(self.service_rate):15.2%})\n"
        )

        for t, g_a in progress["progress"].items():
            for g, a_saved in g_a.items():
                for a, saved in a_saved.items():
                    v, c, t_bias, variance, step, lam, agg_bias = saved
                    self.values[t][g][a] = v
                    self.count[t][g][a] = c
                    self.transient_bias[t][g][a] = t_bias
                    self.variance_g[t][g][a] = variance
                    self.step_size_func[t][g][a] = step
                    self.lambda_stepsize[t][g][a] = lam
                    self.aggregation_bias[t][g][a] = agg_bias

        return self.n, self.reward, self.service_rate, self.weights

    @property
    def current_data(self):

        adp_data = {
            t: {
                g: {
                    a: (
                        value,
                        self.count[t][g][a],
                        self.transient_bias[t][g][a],
                        self.variance_g[t][g][a],
                        self.step_size_func[t][g][a],
                        self.lambda_stepsize[t][g][a],
                        self.aggregation_bias[t][g][a],
                    )
                    for a, value in a_value.items()
                }
                for g, a_value in g_a.items()
            }
            for t, g_a in self.values.items()
        }

        return adp_data

    def load_episode(self, path, label):

        """Load .npy dictionary containing value functions of last
        episode.

        Arguments:
            path {str} -- File with saved value functions
        """
        values_old = np.load(path + label + ".npy").item()
        # print(values_old)
        for t, g_a in values_old.items():
            for g, a_value in g_a.items():
                for a, value in a_value.items():
                    self.values[t][g][a] = value

