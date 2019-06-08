from functools import partial
from bokeh.models import ColumnDataSource, Toggle, Slider
from threading import Thread
from bokeh.themes import built_in_themes
from bokeh.plotting import curdoc, figure
from bokeh.layouts import column, row
from tornado import gen
from bokeh.document import without_document_lock
from collections import defaultdict
from bokeh.tile_providers import get_provider, Vendors

from mod.env.car import Car
import mod.env.visual as vi
import mod.env.network as nw
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class PlotTrack:

    # Plot steps
    ENABLE_PLOT = True
    # Delay after each assignment (in seconds)

    STEP_DELAY = 0

    # Max alpha of value function spots
    MAX_ALPHA_VALUE_FUNCTION = 0.3
    CAR_FILL_ALPHA = 0.5
    # Size of car glyph
    CAR_SIZE = 8

    PLOT_STEP = 0
    OPT_STEP = 1
    PLOT_EPISODE = 2
    AMOD = 3

    # Number of coordinates composing the car paths within a step
    SHOW_SP_LINES = False
    SHOW_LINES = True

    N_POINTS = 20
    STEP_DURATION = 60

    FRAME_UPDATE_DELAY = 1

    def __init__(self, episode, step, opt_step):

        # -------------------------------------------------------------------- #
        # Slide steps ahead ################################################## #
        # -------------------------------------------------------------------- #

        self.steps_ahead = 0

        self.slide_alpha = Slider(
            title="Opacity lines",
            start=0,
            end=1,
            value=0.2,
            step=0.05,
            width=150,
        )

        self.all_points = dict(x=[], y=[])
        # This is important! Save curdoc() to make sure all threads
        # see the same document.
        self.doc = curdoc()
        self.doc.theme = "caliber"
        self.doc.title = "Simulation"
        self.center_lines = []

        # create a plot and style its properties
        self.p = figure(
            title="Simulation",
            x_axis_type="mercator",
            y_axis_type="mercator",
            plot_height=800,
            border_fill_color="white",
            background_fill_color="white",
        )

        self.p.title.text_font_size = "25px"
        self.p.add_tile(get_provider(Vendors.CARTODBPOSITRON_RETINA))
        self.p.title.align = "center"

        self.plot_step = step
        self.plot_episode = episode
        self.opt_step = opt_step
        self.env = None
        self.trips_dict = dict()
        self.step_car_path_dict = defaultdict(lambda: defaultdict(dict))

        source_point_value = ColumnDataSource(
            data=dict(x=[], y=[], fill_alpha=[])
        )

        # -------------------------------------------------------------------- #
        # Value function point style ######################################### #
        # -------------------------------------------------------------------- #

        self.value_function = self.p.circle(
            x="x",
            y="y",
            size=30,
            color="purple",
            fill_alpha="fill_alpha",
            line_width=0,
            muted_alpha=0.0,
            legend="Value function",
            source=source_point_value,
        )

        self.source = {
            Car.REBALANCE: self.p.triangle(
                x=[],
                y=[],
                size=PlotTrack.CAR_SIZE,
                color="firebrick",
                fill_alpha=PlotTrack.CAR_FILL_ALPHA,
                line_width=0,
                muted_alpha=0.1,
                legend=Car.REBALANCE,
            ),
            Car.ASSIGN: self.p.triangle(
                x=[],
                y=[],
                size=PlotTrack.CAR_SIZE,
                color="green",
                fill_alpha=PlotTrack.CAR_FILL_ALPHA,
                line_width=0,
                muted_alpha=0.1,
                legend=Car.ASSIGN,
            ),
            Car.IDLE: self.p.triangle(
                x=[],
                y=[],
                size=PlotTrack.CAR_SIZE,
                # color="navy",
                fill_alpha=0.0,
                line_width=0.5,
                line_color="navy",
                muted_alpha=0.1,
                legend=Car.IDLE,
            ),
            Car.RECHARGING: self.p.triangle(
                x=[],
                y=[],
                size=PlotTrack.CAR_SIZE,
                color="purple",
                line_width=0,
                fill_alpha=PlotTrack.CAR_FILL_ALPHA,
                muted_alpha=0.1,
                legend=Car.RECHARGING,
            ),
            "o": self.p.circle(
                x=[],
                y=[],
                size=15,
                color="green",
                fill_alpha=0.3,
                line_width=0,
                # line_color="green",
                muted_alpha=0.1,
                legend="Origins",
            ),
            "d": self.p.circle(
                x=[],
                y=[],
                size=15,
                color="firebrick",
                fill_alpha=0.3,
                line_width=0,
                # line_color="firebrick",
                muted_alpha=0.1,
                legend="Destinations",
            ),
        }

        self.slide_alpha.on_change("value", self.update_line_alpha_centers)
        self.slide_time_ahead = Slider(
            title="Steps ahead", start=0, end=30, value=0, step=1, width=150
        )
        self.slide_time_ahead.on_change("value", self.update_time_ahead)

    @gen.coroutine
    @without_document_lock
    def update_line_alpha_centers(self, attrname, old, new):
        # global run_plot
        for c_lines in self.center_lines:
            c_lines.glyph.line_alpha = self.slide_alpha.value

    def config_figure(self):
        self.p.legend.click_policy = "mute"
        self.p.legend.location = "bottom_right"
        # p.outline_line_color = None

    @gen.coroutine
    def update_plot_frame(self):

        if self.plot_step in self.trips_dict:

            # Plot all trips
            for od, trips in self.trips_dict[self.plot_step].items():
                self.source[od].data_source.data = trips

        if (
            self.plot_step < self.opt_step
            and self.plot_step in self.step_car_path_dict
        ):
            status_movements, next_step = vi.get_next_frame(
                self.step_car_path_dict, self.plot_step
            )

            # Update the car paths in all car statuses (i.e., rebalancing,
            # parked, picking up user and recharging)
            for status in Car.status_list:
                car_paths_xy = status_movements.get(status, dict(x=[], y=[]))
                self.source[status].data_source.data = car_paths_xy

            if next_step > self.plot_step:

                # Trips are created in step n, and vehicles are scheduled in
                # step n + 1
                # try:
                #     del trips_dict[current_plot_step-1]
                # except:
                #     pass
                # Update plot title
                self.plot_step = next_step

                # Updating title
                self.p.title.text = (
                    f"Episode: {self.plot_episode:>5} - "
                    f"Time step: {self.plot_step:>5}"
                )

                # Update attribute value functions
                self.update_value_function(self.plot_step, 20)

    @gen.coroutine
    def update_attribute(self, attribute, value, param):
        if param:
            attribute.data_source.data[param] = value
        else:
            attribute.data_source.data = value

    def update_screen(self, attribute=None, value=None, param=None):

        # Updating alpha values
        self.doc.add_next_tick_callback(
            partial(
                self.update_attribute,
                attribute=attribute,
                value=value,
                param=param,
            )
        )

    @gen.coroutine
    @without_document_lock
    def update_value_function(self, steps_ahead, battery_level):
        """Update the alpha of all value function spots considering a number
        of steps ahead the current time step.

        Parameters
        ----------
        source_point_value: point data
            Source of value function data (alphas are "updated")
        amod : Environment
            Environment where value functions and points are saved.
        steps_ahead : int
            Number of steps ahead value functions should be shown.
        battery_level : int
            Show value functions corresponding to a battery level.
        """
        print("Calculating value functions...")

        # Value function of all points
        values = np.zeros(len(self.env.points))

        # Values correspond to aggregation level
        agg_level = 0

        # Number of steps ahead value functions are visualized
        future_step = self.plot_step + steps_ahead

        # Get all valid value function throughout the map at a certain level
        for point in self.env.points:

            # Value function corresponds to position and battery level,
            # i.e., how valuable is to have a vehicle at position p with
            # a certain battery level
            attribute = (point.id, battery_level)

            # Checking whether value function was defined
            if (
                future_step in self.env.values
                and agg_level in self.env.values[future_step]
                and attribute in self.env.values[future_step][agg_level]
            ):
                values[point.id] = self.env.values[future_step][agg_level][
                    attribute
                ]

        # Total value function throughout all points
        total = np.sum(values)

        if total > 0:
            # Values are between 0 and 1
            values = values / np.sum(values)

            # Values are normalized
            values = (values - np.min(values)) / (
                np.max(values) - np.min(values)
            )
            # Resize alpha factor
            values = PlotTrack.MAX_ALPHA_VALUE_FUNCTION * values

        print("Finished calculating...")
        self.update_screen(
            attribute=self.value_function, value=values, param="fill_alpha"
        )

    def start_animation(self, sim):

        thread = Thread(target=partial(sim, plot_track=self))

        thread.start()
        self.doc.add_periodic_callback(
            self.update_plot_frame, PlotTrack.FRAME_UPDATE_DELAY
        )

    @gen.coroutine
    def update_first(self, lines, level_centers, level_demand, level_fleet):

        print("Drawing centers...")

        point_regular = self.p.circle(
            x=[],
            y=[],
            size=2,
            color="firebrick",
            fill_alpha=0.5,
            line_width=0,
            legend="Intersection",
            muted_alpha=0,
        )

        column_elements = []

        source_point_value = self.value_function.data_source

        toggles = defaultdict(list)
        i = -1
        for level, centers in level_centers.items():
            i += 1
            # Ignore level 0
            if level == 0:
                regular = centers
                self.all_points = regular
                point_regular.data_source.data = regular
                source_point_value.data = {
                    **{"fill_alpha": np.zeros(len(centers["x"]))},
                    **regular,
                }
                continue

            for line_type, lines_xy in lines.items():

                active = False
                region_fleet = ""
                region_demand = ""
                if level == level_demand:
                    region_demand = " [D] "
                    active = True
                if level == level_fleet:
                    region_fleet = "[F] "
                    active = True

                lines_level_glyph = self.p.multi_line(
                    [],
                    [],
                    line_color="firebrick",
                    line_alpha=0.05,
                    line_width=3,
                    muted_alpha=0.00,
                    visible=active,
                )
                self.center_lines.append(lines_level_glyph)

                point_centers = self.p.circle(
                    x=[],
                    y=[],
                    size=6,
                    color="white",
                    line_width=1,
                    line_color="firebrick",
                    visible=active,
                )

                point_centers.data_source.data = centers
                lines_level_glyph.data_source.data = lines_xy[level]

                toggle = Toggle(
                    label=f"Level {i:>2} ({level:>3}){region_demand}{region_fleet}",
                    active=active,
                    width=150,
                )

                toggle.js_link("active", lines_level_glyph, "visible")
                toggle.js_link("active", point_centers, "visible")

                toggles[line_type].append(toggle)

        for line_type in lines:
            column_elements.extend(toggles[line_type])

        column_elements.append(self.slide_alpha)
        column_elements.append(self.slide_time_ahead)

        self.doc.add_root(row(column(*column_elements), self.p))
        self.config_figure()

        print("Centers, toggles, and slides created.")

    def plot_centers(
        self,
        points,
        levels,
        level_demand,
        level_fleet,
        show_sp_lines=True,
        show_lines=True,
    ):

        centers_xy, lines_xy = vi.get_center_elements(
            points, levels, sp_lines=show_sp_lines, direct_lines=show_lines
        )

        self.doc.add_next_tick_callback(
            partial(
                self.update_first,
                lines=lines_xy,
                level_centers=centers_xy,
                level_fleet=level_fleet,
                level_demand=level_demand,
            )
        )

        print("---- Finished plotting centers.")

    @gen.coroutine
    def update_time_ahead(self, attrname, old, new):
        steps_ahead = self.slide_time_ahead.value
        print("Changing global steps ahead", steps_ahead)
        self.update_value_function(steps_ahead, 20)

    def multithreading(self, func, args, workers):
        with ThreadPoolExecutor(workers) as ex:
            res = ex.map(func, args)
        return list(res)

    def compute_movements(self, step):

        # Get car paths
        for car in self.env.cars:

            # Car path was stored in previous step since its route covers
            # more than one time step
            if car.id not in self.step_car_path_dict[step]:

                # segmented_sp = nw.query_segmented_sp(
                #     c.previous,
                #     c.point,
                #     n_points,
                #     step_duration,
                #     projection="MERCATOR",
                #     waypoint=c.waypoint,
                # )

                dif = car.arrival_time - car.previous_arrival

                # If vehice is parked
                if dif == 0:
                    segmented_sp = [[[car.point.x, car.point.y]]]
                # Vehicle is moving
                else:
                    segmented_sp = nw.query_sp_sliced(
                        car.previous,
                        car.point,
                        PlotTrack.N_POINTS * dif,
                        dif,
                        projection="MERCATOR",
                        waypoint=car.waypoint,
                    )

                # if segmented_sp[0]:
                #     print(f'{c.id} - INSERT TW: ({c.previous_arrival},{c.arrival_time}) Segmented SP: {len(segmented_sp)}')
                #     print("S:", c.previous, c.point)
                # Update car movement in step
                for i, s in enumerate(segmented_sp):
                    self.step_car_path_dict[step + i][car.id] = (car.status, s)

            # else:
            #     print(
            #         # f"Segmented: {[len(s) for status, s in movement_step_fleet_dict[step]]}."
            #         f"\n################ {c} ##############################"
            #         f"\n-           Status: {c.status} "
            #         f"\n- Previous arrival: {c.previous_arrival} "
            #         f"\n-          Arrival: {c.arrival_time} "
            #         f"\n-             Step: {c.step}/{step} "
            #     )

    def get_next_frame(self, step):

        if step in self.step_car_path_dict and self.step_car_path_dict[step]:

            xy_status = defaultdict(lambda: dict(x=[], y=[]))
            count_finished = 0

            for status, path_car in self.step_car_path_dict[step].values():

                if len(path_car) > 1:
                    x, y = path_car.pop(0)
                    xy_status[status]["x"].append(x)
                    xy_status[status]["y"].append(y)
                else:
                    x, y = path_car[0]
                    count_finished += 1
                    xy_status[status]["x"].append(x)
                    xy_status[status]["y"].append(y)

            if count_finished == len(self.step_car_path_dict[step].keys()):
                return xy_status, step + 1
            else:
                return xy_status, step
