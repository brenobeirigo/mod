import mod.env.trip as tp


class Scenario:

    def __init__(self, amod):
        self.amod = amod


class ScenarioUmbalanced(Scenario):

    def __init__(self, amod, path_trips):
        """Generate origin and destination clusters according to origin/destination center count"""
        super().__init__(amod)

        origins, destinations = amod.get_od_lists(amod)

        # Get demand pattern from NY city
        self.step_trip_count = tp.get_trip_count_step(
            path_trips,
            step=self.amod.config.time_increment,
            multiply_for=self.amod.demand_resize_factor,
            earliest_step=self.amod.demand_earliest_step_min,
            max_steps=self.amod.demand_max_steps,
        )
        # Sample ods for iteration n
        self.step_trip_list = tp.get_trips_random_ods(
            self.amod.points,
            self.step_trip_count,
            offset_start=self.amod.config.offset_repositioning_steps,
            offset_end=self.amod.config.offset_termination_steps,
            origins=origins,
            destinations=destinations,
            classed=self.amod.demand_is_classed,
        )


class ScenarioNYC(Scenario):

    def __init__(self, amod, path_trips, n):
        super().__init__(amod)

        self.step_trip_list, self.step_trip_count = tp.get_ny_demand(
            self.amod.config,
            path_trips,
            self.amod.points,
            seed=n,
            prob_dict=self.amod.config.prob_dict,
            centroid_level=self.amod.config.centroid_level,
            unreachable_ods=self.amod.unreachable_ods,
        )
