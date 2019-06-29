from functools import partial
from bokeh.models import ColumnDataSource, Toggle, Slider, Div, Label
from threading import Thread
from bokeh.themes import built_in_themes
from bokeh.plotting import curdoc, figure
from bokeh.layouts import column, row
from tornado import gen
from bokeh.document import without_document_lock
from collections import defaultdict
from bokeh.tile_providers import get_provider, Vendors

from mod.env.car import Car, HiredCar
import mod.env.visual as vi
import mod.env.network as nw
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from mod.env.config import FOLDER_OUTPUT
import os


class PlotTrack:

    # Plot steps
    ENABLE_PLOT = False
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

    REGION_CENTER_LINE_WIDHT = 1
    REGION_CENTER_LINE_ALPHA = 0.3

    # Number of coordinates composing the car paths within a step
    SHOW_SP_LINES = False
    SHOW_LINES = True

    N_POINTS = 30
    STEP_DURATION = 60

    FRAME_UPDATE_DELAY = 1

    def update_label_text(self, main_fleet, secondary_fleet, trips):
        pass

    def __init__(self, config):
        self.config = config
        self.output_path = FOLDER_OUTPUT + config.label
        self.output_folder_simulation = self.output_path + "/simulation/"

        # Creating folder to save partial info
        if not os.path.exists(self.output_folder_simulation):
            os.makedirs(self.output_folder_simulation)

        self.path_region_center_data = (
            self.output_folder_simulation + "region_center_data.npy"
        )

        # ------------------------------------------------------------ #
        # Slide steps ahead ########################################## #
        # ------------------------------------------------------------ #

        self.steps_ahead = 0
        self.slide_alpha = Slider(
            title="Opacity lines",
            start=0,
            end=1,
            value=0.2,
            step=0.05,
            width=150,
        )

        self.fleet_stats = dict()
        self.decisions = dict()

        self.stats = Div(text="", align="center")

        self.all_points = dict(x=[], y=[])
        # This is important! Save curdoc() to make sure all threads
        # see the same document.
        self.doc = curdoc()
        self.doc.theme = "caliber"
        self.doc.title = "Simulation"

        # All lines (alpha control)
        self.center_lines = []

        # create a plot and style its properties
        self.p = figure(
            title="Simulation",
            x_axis_type="mercator",
            y_axis_type="mercator",
            plot_height=800,
            plot_width=1000,
            border_fill_color="white",
            background_fill_color="white",
        )

        self.p.title.text_font_size = "25px"
        self.p.title.align = "center"
        self.p.add_tile(get_provider(Vendors.CARTODBPOSITRON_RETINA))

        self.plot_step = 0
        self.plot_episode = 0
        self.opt_step = 0
        self.opt_episode = 0
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
                muted_alpha=0.0,
                legend=Car.REBALANCE,
            ),
            Car.ASSIGN: self.p.triangle(
                x=[],
                y=[],
                size=PlotTrack.CAR_SIZE,
                color="green",
                fill_alpha=PlotTrack.CAR_FILL_ALPHA,
                line_width=0,
                muted_alpha=0.0,
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
                muted_alpha=0.0,
                legend=Car.IDLE,
            ),
            Car.RECHARGING: self.p.triangle(
                x=[],
                y=[],
                size=PlotTrack.CAR_SIZE,
                color="purple",
                line_width=0,
                fill_alpha=PlotTrack.CAR_FILL_ALPHA,
                muted_alpha=0.0,
                legend=Car.RECHARGING,
            ),
            "o": self.p.circle(
                x=[],
                y=[],
                size=15,
                color="green",
                fill_alpha=0.3,
                line_width=0,
                muted_alpha=0.0,
                legend="Origins",
            ),
            "d": self.p.circle(
                x=[],
                y=[],
                size=15,
                color="firebrick",
                fill_alpha=0.3,
                line_width=0,
                muted_alpha=0.0,
                legend="Destinations",
            ),
        }

        self.slide_alpha.on_change("value", self.update_line_alpha_centers)
        self.slide_time_ahead = Slider(
            title="Time step", start=1, end=15, value=1, step=1, width=150
        )
        self.slide_time_ahead.on_change("value", self.update_time_ahead)

        self.slide_battery_level = Slider(
            title="Battery level", start=0, end=1, value=0, step=1, width=150
        )
        self.slide_battery_level.on_change("value", self.update_time_ahead)

        self.slide_agg_level = Slider(
            title="Aggregation level",
            start=0,
            end=10,
            value=0,
            step=1,
            width=150,
        )
        self.slide_agg_level.on_change("value", self.update_time_ahead)

    def set_env(self, env):
        self.env = env
        self.slide_agg_level.end = env.config.aggregation_levels
        self.slide_time_ahead.end = env.config.time_steps
        self.slide_battery_level.end = env.config.battery_levels

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

        # print("###############", self.plot_step, "<", self.opt_step)
        # pprint(self.step_car_path_dict)
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
                current_time = self.config.get_time(
                    self.plot_step,
                    format='%I:%M %p'
                )
                self.p.title.text = (
                    f"Episode: {self.plot_episode:>5} - "
                    f"Time: {current_time} - "
                    f"Step: {self.plot_step:>5}/{self.config.time_steps:>5}"
                )

                # Stats
                self.stats.text = self.get_fleet_stats(self.plot_step)

                # Update attribute value functions
                # self.update_value_function(
                #     self.plot_step,
                #     self.slide_battery_level.value,
                #     self.slide_agg_level.value
                # )

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
    def update_value_function(self, steps_ahead, battery_level, agg_level):
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
        agg_level : int
            Values correspond to aggregation level

        """
        print("Calculating value functions...")

        # Value function of all points
        values = np.zeros(len(self.env.points))

        # Number of steps ahead value functions are visualized
        future_step = steps_ahead

        # Get all valid value function throughout the map at a certain level
        for point in self.env.points:

            # Value function corresponds to position and battery level,
            # i.e., how valuable is to have a vehicle at position p with
            # a certain battery level
            attribute = (point.id, battery_level)

            # Checking whether value function was defined
            # if (
            #     future_step in self.env.values
            #     and agg_level in self.env.values[future_step]
            #     and attribute in self.env.values[future_step][agg_level]
            # ):
            # id_g = point.id_level(agg_level)
            estimate = self.env.get_weighted_value(
                future_step, point.id, battery_level
            )
            values[point.id] = estimate
            # self.env.values[future_step][agg_level][attribute]

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

    def create_value_function_points(self, points):
        self.value_function.data_source.data = {
            **{"fill_alpha": np.zeros(len(points["x"]))},
            **points,
        }

    def create_regular_points(self, points):

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

        point_regular.data_source.data = points
        self.all_points = points

    def get_region_center_toggle(
        self, lines_xy, i_level, level, level_demand, level_fleet, centers
    ):

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
            line_alpha=PlotTrack.REGION_CENTER_LINE_ALPHA,
            line_width=PlotTrack.REGION_CENTER_LINE_WIDHT,
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
            label=(
                f"Level {i_level:>2} ({level:>3}s)"
                f"{region_demand + region_fleet:>7}"
            ),
            active=active,
            width=150,
        )

        toggle.js_link("active", lines_level_glyph, "visible")
        toggle.js_link("active", point_centers, "visible")

        return toggle

    @staticmethod
    def default_to_regular(d):
        if isinstance(d, defaultdict):
            d = {k: PlotTrack.default_to_regular(v) for k, v in d.items()}
        return d

    def get_fleet_stats(self, step):

        # text = "<h4>### FLEET STATS </h4>"
        text = "<table>"
        for fleet_type, status_count in self.fleet_stats[step].items():

            text += f"<tr><td style='font-size:16px;text-align:right'><b>{fleet_type}</b></td>"

            for status, count in status_count.items():
                text += (
                    f"<td style='text-align:right'>"
                    f"<b>{status}:</b>"
                    "</td><td style='width:15px'>"
                    f"{count}"
                    "<td>"
                )
            text += "</tr>"

        text += "</table>"

        # text = "<h4>### FLEET STATS </h4>"
        text += "<table><tr>"
        for decision, count in self.decisions[step].items():
            text += (
                f"<td style='text-align:right'><b>{decision}:</b>"
                "</td><td style='width:15px'>"
                f"{count}"
                "<td>"
            )

        text += "</tr> </table>"
        return text

    @gen.coroutine
    @without_document_lock
    def update_first(self, lines, level_centers, level_demand, level_fleet):

        print("Drawing centers...")

        column_elements = []

        toggles = defaultdict(list)
        i = -1
        for level, centers in level_centers.items():
            i += 1
            # Level 0 corresponds to regular intersections
            if level == 0:
                self.create_regular_points(centers)
                self.create_value_function_points(centers)
                continue

            # Line types = Direct lines or Shortest Path lines, from
            #           centers to elements.
            # Lines xy =  Dicitionary of line coordinates, for example,
            #           {x=[[x1, x2], [x1, x3]], y=[[y1, y2], [y1, y3]]}
            for line_type, lines_xy in lines.items():

                toggle = self.get_region_center_toggle(
                    lines_xy, i, level, level_demand, level_fleet, centers
                )

                toggles[line_type].append(toggle)

        # Add all toggles to column
        for line_type in lines:
            # Title before region center toggles
            line_type_title = Div(
                text=f"<h3>{line_type} lines</h3>", width=150
            )
            column_elements.append(line_type_title)

            # Toggles
            column_elements.extend(toggles[line_type])

        column_elements.append(self.slide_alpha)
        column_elements.append(self.slide_time_ahead)
        column_elements.append(self.slide_battery_level)
        column_elements.append(self.slide_agg_level)

        title = Div(
            text=(f"<h1>{self.env.config.region}</h1>"), align="center"
        )

        network_info = Div(
            text=(
                f"<h2>{self.env.config.node_count} nodes & "
                f"{self.env.config.edge_count} edges</h2>"
            ),
            align="center",
        )

        center_count = Div(
            text=(
                " - ".join(
                    [
                        f"<b>{dist}</b>({count})"
                        for dist, count in self.env.config.center_count_dict.items()
                    ]
                )
            )
        )

        self.doc.add_root(
            column(
                title,
                network_info,
                self.stats,
                row(column(*column_elements), self.p),
                center_count,
            )
        )

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
        path_center_data=None,
    ):

        try:
            print("\nReading center data...")
            center_lines_dict = np.load(self.path_region_center_data).item()
            centers_xy = center_lines_dict["centers_xy"]
            lines_xy = center_lines_dict["lines_xy"]
            print("Center data loaded successfully.")

        # Centers were not previously saved
        except Exception as e:

            print(
                f"\nFailed reading center data. Exception: {e} "
                "\nPulling center data from server..."
            )
            centers_xy, lines_xy = vi.get_center_elements(
                points, levels, sp_lines=show_sp_lines, direct_lines=show_lines
            )
            print("Saving center data...")
            np.save(
                self.path_region_center_data,
                {
                    "centers_xy": PlotTrack.default_to_regular(centers_xy),
                    "lines_xy": PlotTrack.default_to_regular(lines_xy),
                },
            )
            print(f"Center data saved at '{self.path_region_center_data}'")

        self.doc.add_next_tick_callback(
            partial(
                self.update_first,
                lines=lines_xy,
                level_centers=centers_xy,
                level_fleet=level_fleet,
                level_demand=level_demand,
            )
        )

        print("Finished plotting centers.")

    @gen.coroutine
    def update_time_ahead(self, attrname, old, new):
        steps_ahead = self.slide_time_ahead.value
        battery_level = self.slide_battery_level.value
        agg_level = self.slide_agg_level.value

        print("Changing value function", steps_ahead, battery_level, agg_level)
        self.update_value_function(steps_ahead, battery_level, agg_level)

    def multithreading(self, func, args, workers):
        with ThreadPoolExecutor(workers) as ex:
            res = ex.map(func, args)
        return list(res)

    def compute_movements(self, step):

        self.fleet_stats[step] = self.env.get_fleet_stats()
        self.decisions[step] = self.env.decision_dict

        fleet = self.env.cars

        # If working with hired vehicles, only compute movements from those
        # which started working, i.e., hired.
        try:
            active_hired = [
                car for car in self.env.hired_cars if not car.started_contract
            ]
            fleet += active_hired

        except:
            pass
        # Get car paths
        for car in fleet:

            # if car.status == Car.IDLE:
            #     continue

            # Car path was stored in previous step since its route
            # covers more than one time step
            if car.id not in self.step_car_path_dict[step]:

                # segmented_sp = nw.query_segmented_sp(
                #     c.previous,
                #     c.point,
                #     n_points,
                #     step_duration,
                #     projection="MERCATOR",
                #     waypoint=c.waypoint,
                # )

                if car.previous == car.point:
                    segmented_sp = [[[car.point.x, car.point.y]]]

                # Vehicle is moving
                else:
                    # TODO should be current time?
                    dif = car.arrival_time - car.previous_arrival

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
                    if not s:
                        print(
                            f"NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO TYPE: {car.type}  -  STATUS: {car.status} - dif:{dif} - arrival:{car.arrival_time}/previous:{car.previous_arrival}- Segmented: {segmented_sp}"
                        )

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

    # def get_next_frame(self, step):

    #     if step in self.step_car_path_dict and self.step_car_path_dict[step]:

    #         xy_status = defaultdict(lambda: dict(x=[], y=[]))
    #         count_finished = 0

    #         for status, path_car in self.step_car_path_dict[step].values():

    #             if len(path_car) > 1:
    #                 x, y = path_car.pop(0)
    #                 xy_status[status]["x"].append(x)
    #                 xy_status[status]["y"].append(y)

    #             # Does not erase last position visited by car
    #             # When number of coordinates vary, it is desirible that
    #             # cars that have already travelled their paths wait in
    #             # the last position they visited.

    #             elif len(path_car) == 1:
    #                 x, y = path_car[0]
    #                 count_finished += 1
    #                 xy_status[status]["x"].append(x)
    #                 xy_status[status]["y"].append(y)

    #             else:
    #                 print(step)

    #                 # pprint(self.step_car_path_dict)
    #                 # pprint(self.step_car_path_dict[step])

    #                 # pass
    #                 # TODO Sometimes path_car[0] does no exist. This
    #                 # cannot happen since coordinates are popped when
    #                 # there are more than 2 elements.
    #                 # Multithreading? path_car was not populated
    #                 # correctly in the first place?

    #         if count_finished == len(self.step_car_path_dict[step].keys()):
    #             return xy_status, step + 1
    #         else:
    #             return xy_status, step

    # ################################################################ #
    # START ########################################################## #
    # ################################################################ #

    def start_animation(self, opt_method):
        """Start animation using opt_method as a base.

        In opt_method, movements are computed and passed to simulator.

        Parameters
        ----------
        opt_method : def
            Function where optimization takes place.
        """

        thread = Thread(
            target=partial(opt_method, plot_track=self, config=self.config)
        )

        thread.start()
        self.doc.add_periodic_callback(
            self.update_plot_frame, PlotTrack.FRAME_UPDATE_DELAY
        )
