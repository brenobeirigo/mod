import itertools

from .decisions import rebalance_decisions, rebalance_decisions_thompson, stay_decision


class DecisionSetReactive:
    def __init__(self, env, trips):
        self.env = env
        self.trips = trips

        # Rebalancing targets are pickup ids of rejected trips
        self.targets = [target.o.id for target in self.trips]

        # Only idle cars can rebalance to targets
        # Get REBALANCE and STAY decisions
        self.all_decisions = set()
        # How many cars can rebalance? Hired cars can rebalance only if
        # contract limit is not surpassed.
        self.n_cars_can_rebalance = 0
        self.attribute_trips_dict = dict()
        self.get_rebalancing_decisions()

    def get_rebalancing_decisions(self):
        """Stay and rebalancing decisions for the reactive rebalancing
        policy.

        Parameters
        ----------
        env : AMoD
            AMoD environment
        targets : list
            Rebalancing targets

        Returns
        -------
        set, int
            Set of all decisions (rebalancing + stay)
            Number of cars that can rebalance
        """

        # ##################################################################
        # SORT CARS ########################################################
        # ##################################################################

        for car in itertools.chain(self.env.available, self.env.available_hired):

            # Stay ####################################################### #
            d_stay = stay_decision(car)
            self.all_decisions.add(d_stay)

            if self.env.config.activate_thompson:
                d_rebalance = rebalance_decisions_thompson(car, self.targets, self.env)
            else:
                d_rebalance = rebalance_decisions(car, self.targets, self.env)

            if not d_rebalance:
                # Remove from tabu if not empty.
                # Avoid cars are corned indefinitely
                if car.tabu:
                    car.tabu.popleft()
            else:
                # Rebalance decision was created
                self.n_cars_can_rebalance += 1

            # TODO this is here because of a lack or rebalancing options
            # thompson selected is small 0.2
            if len(d_rebalance) == 1:
                d_stay = stay_decision(car)
                self.all_decisions.add(d_stay)

            # Vehicles can stay idle for a maximum number of steps.
            # If they surpass this number, they can rebalance to farther
            # areas.
            if self.env.config.max_idle_step_count:

                # Car can rebalance to farther locations besides the
                # closest after staying still for idle_step_count steps
                if car.idle_step_count >= self.env.config.max_idle_step_count:
                    farther = self.env.get_zone_neighbors(car.point.id, explore=True)

                    # print(f"farther: {farther} - d_rebalance: {d_rebalance}")
                    d_rebalance.update(rebalance_decisions(car, farther, self.env))
                    # d_rebalance = d_rebalance | farther

            # print(f"farther: {d_rebalance}")

            self.all_decisions.update(d_rebalance)
