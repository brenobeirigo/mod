import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


# Adding project folder to import modules
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

from mod.env.amod import Amod, AmodNetwork
from mod.env.visual import StepLog, EpisodeLog
from mod.env.config import ConfigNetwork, NY_TRIPS_EXCERPT_DAY
from mod.env.match import adp_network, myopic, fcfs
from mod.env.car import Car
from mod.env.trip import (
    get_random_trips,
    get_trip_count_step,
    get_trips_random_ods,
)
import mod.env.network as nw
from pprint import pprint

from functools import partial
from random import random
from threading import Thread
import time

from bokeh.models import (
    ColumnDataSource,
    Toggle,
    BoxAnnotation,
    CheckboxButtonGroup,
    GridBox,
    Label,
    Slider,
)
from bokeh.themes import built_in_themes
from bokeh.plotting import curdoc, figure
from bokeh.layouts import layout, column, row
from tornado import gen
from collections import defaultdict
from bokeh.tile_providers import get_provider, Vendors

tile_provider = get_provider(Vendors.CARTODBPOSITRON)

# Plot steps
plot = False

# This is important! Save curdoc() to make sure all threads
# see the same document.
doc = curdoc()
doc.theme = "caliber"
doc.title = "Simulation"

# create a plot and style its properties
p = figure(
    title="Simulation",
    x_axis_type="mercator",
    y_axis_type="mercator",
    plot_height=800,
)

p.border_fill_color = "white"
p.background_fill_color = "white"
p.outline_line_color = None
p.grid.grid_line_color = None
p.title.text_font_size = "25px"
p.title.align = "center"
p.add_tile(tile_provider)

car_fill_alpha = 0.3

# Bokeh plot
point_r = p.triangle(
    x=[],
    y=[],
    size=8,
    color="firebrick",
    fill_alpha=car_fill_alpha,
    line_width=0,
    muted_alpha=0.1,
    legend=Car.REBALANCE,
)

point_g = p.triangle(
    x=[],
    y=[],
    size=8,
    color="green",
    fill_alpha=car_fill_alpha,
    line_width=0,
    muted_alpha=0.1,
    legend=Car.ASSIGN,
)

# https://bokeh.pydata.org/en/latest/docs/reference/models/annotations.html#bokeh.models.annotations.Label
# label = Label(
#     x_units="screen",
#     y_units="screen",
#     x_offset=200,
#     y_offset=200,
#     text="Some Stuf dfd fdf df d fd fdf df df df dfdf df df dfdf d fdf ddf df df df dff",
#     render_mode="css",
#     background_fill_color="white",
#     background_fill_alpha=1.0,
#     text_font_size="14px",
# )

# p.add_layout(label)

point_s = p.triangle(
    x=[],
    y=[],
    size=8,
    # color="navy",
    fill_alpha=0.0,
    line_width=0.5,
    line_color="navy",
    muted_alpha=0.1,
    legend=Car.IDLE,
)
point_b = p.triangle(
    x=[],
    y=[],
    size=8,
    color="purple",
    line_width=0,
    fill_alpha=car_fill_alpha,
    muted_alpha=0.1,
    legend=Car.RECHARGING,
)

point_o = p.circle(
    x=[],
    y=[],
    size=15,
    color="green",
    fill_alpha=0.3,
    line_width=0,
    # line_color="green",
    muted_alpha=0.1,
    legend="Origins",
)
point_d = p.circle(
    x=[],
    y=[],
    size=15,
    color="firebrick",
    fill_alpha=0.3,
    line_width=0,
    # line_color="firebrick",
    muted_alpha=0.1,
    legend="Destinations",
)

source_points = {
    Car.REBALANCE: point_r,
    Car.ASSIGN: point_g,
    Car.IDLE: point_s,
    Car.RECHARGING: point_b,
    "o": point_o,
    "d": point_d,
}

p.legend.click_policy = "mute"

center_lines = []

slide_alpha = Slider(
    title="Opacity lines", start=0, end=1, value=0.1, step=0.05, width=150
)


@gen.coroutine
def update_line_alpha_centers(attrname, old, new):

    for c_lines in center_lines:
        c_lines.glyph.line_alpha = slide_alpha.value


slide_alpha.on_change("value", update_line_alpha_centers)


@gen.coroutine
def update_first(lines, regular, centers, levels, level_demand, level_fleet):

    point_regular = p.circle(
        x=[], y=[], size=1, color="black", fill_alpha=0.5, line_width=0.1
    )

    list_toggle = []

    point_regular.data_source.data = regular

    for i, level in enumerate(levels):
        if i == 0:
            continue
        active = False
        region_fleet = ""
        region_demand = ""
        if i == level_demand:
            region_demand = " [D] "
            active = True
        if i == level_fleet:
            region_fleet = "[F] "
            active = True

        lines_level_glyph = p.multi_line(
            [],
            [],
            line_color="firebrick",
            line_alpha=0.05,
            muted_alpha=0.00,
            visible=active,
        )

        center_lines.append(lines_level_glyph)

        point_centers = p.circle(
            x=[],
            y=[],
            size=6,
            color="white",
            line_width=1,
            line_color="firebrick",
            visible=active,
        )

        toggle = Toggle(
            label=f"Level {i:>2} ({level:>3}){region_demand}{region_fleet}",
            button_type="success",
            active=active,
            width=150,
        )

        toggle.js_link("active", lines_level_glyph, "visible")
        toggle.js_link("active", point_centers, "visible")
        # slide_alpha.js_link("value", lines_level_glyph, "opacity")

        list_toggle.append(toggle)

        point_centers.data_source.data = centers[level]
        lines_level_glyph.data_source.data = lines[level]

    list_toggle.append(slide_alpha)

    doc.add_root(row(column(*list_toggle), p))

    return center_lines


@gen.coroutine
def update_timestep(status_fleet, trips, episode, timestep):

    p.title.text = f"Episode: {episode:>5} - " f"Time step: {timestep:>5}"

    for status in status_fleet:
        source_points[status].data_source.data = status_fleet[status]

    for t in trips:
        source_points[t].data_source.data = trips[t]


def plot_centers(points, levels, level_demand, level_fleet):

    center_lines = defaultdict(lambda: defaultdict(list))
    regular = defaultdict(list)
    center_points = defaultdict(lambda: defaultdict(list))
    center_ids = set()

    for p in points:
        regular["x"] += [p.x]
        regular["y"] += [p.y]

        for i, level in enumerate(levels):
            if i == 0:
                continue
            cid = p.id_level(i)

            c_point = points[cid]

            center_lines[level]["xs"].append([c_point.x, p.x])
            center_lines[level]["ys"].append([c_point.y, p.y])

            if c_point.id not in center_ids:
                center_points[level]["x"].append(c_point.x)
                center_points[level]["y"].append(c_point.y)
                center_ids.add(c_point.id)

    doc.add_next_tick_callback(
        partial(
            update_first,
            lines=center_lines,
            regular=regular,
            centers=center_points,
            levels=levels,
            level_fleet=level_fleet,
            level_demand=level_demand,
        )
    )


def plot_fleet(cars, trips, episode, timestep):

    # xy_status = defaultdict(lambda: defaultdict(list))
    # print(
    #     f"### {timestep} ##########################################################"
    # )
    car_sp = dict()
    for c in cars:
        # xy_status[c.status]["x"].append(c.point.x)
        # xy_status[c.status]["y"].append(c.point.y)
        sp = nw.query_sp(c.previous, c.point, "MERCATOR")
        car_sp[c.id] = sp
        # print(
        #     f"{c.id:>04} - {c.status:>12} = {c.previous.id:>04}->{c.point.id:>04} - SP:{sp}"
        # )

    xy_trips = defaultdict(lambda: defaultdict(list))
    xy_trips["o"]["x"] = []
    xy_trips["o"]["y"] = []

    # Origin, destination coordinates
    for t in trips:
        xy_trips["o"]["x"].append(t.o.x)
        xy_trips["o"]["y"].append(t.o.y)

        xy_trips["d"]["x"].append(t.d.x)
        xy_trips["d"]["y"].append(t.d.y)

    while True:
        xy_status = defaultdict(lambda: defaultdict(list))
        count_finished = 0
        for c in cars:
            if len(car_sp[c.id]) > 1:
                x, y = car_sp[c.id].pop(0)
            else:
                x, y = car_sp[c.id][0]
                count_finished += 1

            xy_status[c.status]["x"].append(x)
            xy_status[c.status]["y"].append(y)

        doc.add_next_tick_callback(
            partial(
                update_timestep,
                status_fleet=xy_status,
                trips=xy_trips,
                episode=episode,
                timestep=timestep,
            )
        )
        time.sleep(0.0001)
        if count_finished == len(cars):
            break


def sim(plot):

    # -----------------------------------------------------------------#
    # Amod environment #################################################
    # -----------------------------------------------------------------#

    config = ConfigNetwork()
    config.update(
        {
            # Fleet
            ConfigNetwork.FLEET_SIZE: 1500,
            ConfigNetwork.BATTERY_LEVELS: 20,
            # Time - Increment (min)
            ConfigNetwork.TIME_INCREMENT: 1,
            ConfigNetwork.OFFSET_REPOSIONING: 15,
            ConfigNetwork.OFFSET_TERMINATION: 30,
            # NETWORK ##################################################
            # Region centers are created in steps of how much time?
            ConfigNetwork.STEP_SECONDS: 30,
            # Demand spawn from how many centers?
            ConfigNetwork.ORIGIN_CENTERS: 2,
            # Demand arrives in how many centers?
            ConfigNetwork.DESTINATION_CENTERS: 2,
            # OD level extension
            ConfigNetwork.DEMAND_CENTER_LEVEL: 3,
            # Cars rebalance to up to #region centers
            ConfigNetwork.N_CLOSEST_NEIGHBORS: 8,
            # Cars can access locations within region centers
            # established in which neighborhood level?
            ConfigNetwork.NEIGHBORHOOD_LEVEL: 2,
            ConfigNetwork.LEVEL_DIST_LIST: [0, 30, 60, 120, 180, 300, 600],
            # How many levels separated by step seconds? If None, ad-hoc
            # LEVEL_DIST_LIST must be filled
            ConfigNetwork.AGGREGATION_LEVELS: 7,
            ConfigNetwork.SPEED: 30,
        }
    )

    # "centers":{"30":684,"60":235,"90":119,"120":73,"150":50,"180":37,"210":32,"240":24,"270":19,"300":16,"330":13,"360":11,"390":9,"420":9,"450":8,"480":7,"510":6,"540":5,"570":5,"600":4}

    # ################################################################ #
    # Slice demand ################################################### #
    # ################################################################ #

    # Data correspond to 1 day NY demand
    total_hours = 12
    earliest_hour = 14
    resize_factor = 1
    max_steps = int(total_hours * 60 / config.time_increment)
    earliest_step_min = int(earliest_hour * 60 / config.time_increment)

    # ---------------------------------------------------------------- #
    # Episodes ####################################################### #
    # ---------------------------------------------------------------- #
    episodes = 290
    episodeLog = EpisodeLog(config=config)
    amod = AmodNetwork(config)

    # ---------------------------------------------------------------- #
    # Plot centers and guidelines #################################### #
    # ---------------------------------------------------------------- #
    if plot:
        plot_centers(
            amod.points,
            nw.Point.levels,
            config.demand_center_level,
            config.neighborhood_level,
        )
    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    try:
        # Load last episode
        progress = episodeLog.load_progress()
        amod.load_progress(progress)

    except Exception as e:
        print(f"No previous episodes were saved {e}.")

    # ---------------------------------------------------------------- #
    # Trips ########################################################## #
    # ---------------------------------------------------------------- #

    try:
        o_ids, d_ids = episodeLog.load_ods()
        origins = [amod.points[o] for o in o_ids]
        destinations = [amod.points[d] for d in d_ids]
        print(
            f"Loading {len(origins)} origins and "
            f"{len(destinations)} destinations."
        )

    except Exception as e:

        print(f"Error!{e}")

        # Create random centers from where trips come from
        # TODO choose level to query origins
        origins = nw.query_centers(
            amod.points,
            amod.config.origin_centers,
            amod.config.demand_center_level,
        )

        destinations = nw.query_centers(
            amod.points,
            amod.config.destination_centers,
            amod.config.demand_center_level,
        )

        print(
            f"\nSaving {len(origins)} origins and "
            f"{len(destinations)} destinations."
        )
        episodeLog.save_ods(
            [o.id for o in origins], [d.id for d in destinations]
        )

    # Get demand pattern from NY city
    step_trip_count = get_trip_count_step(
        NY_TRIPS_EXCERPT_DAY,
        step=config.time_increment,
        multiply_for=resize_factor,
        earliest_step=earliest_step_min,
        max_steps=max_steps,
    )

    print(
        f"### DEMAND ###"
        f" - min: {min(step_trip_count)}"
        f" - max: {max(step_trip_count)}"
    )

    # ---------------------------------------------------------------- #
    # Experiment ##################################################### #
    # ---------------------------------------------------------------- #

    # Loop all episodes, pick up trips, and learn where they are
    for n in range(episodeLog.n, episodes):

        # Create all episode trips
        step_trip_list = get_trips_random_ods(
            amod.points,
            step_trip_count,
            offset_start=amod.config.offset_repositioning,
            offset_end=amod.config.offset_termination,
            origins=origins,
            destinations=destinations,
        )

        # Start saving data of each step in the adp_network
        step_log = StepLog(amod)

        # Resetting environment
        amod.reset()

        # ------------------------------------------------------------ #
        # Plot fleet current status ################################## #
        # ------------------------------------------------------------ #
        if plot:
            plot_fleet(amod.cars, [], n, 0)

        # Iterate through all steps and match requests to cars
        for step, trips in enumerate(step_trip_list):

            revenue, serviced, rejected = adp_network(
                amod,
                trips,
                step + 1,
                neighborhood_level=config.neighborhood_level,
                n_neighbors=config.n_neighbors,
                # agg_level=amod.config.incumbent_aggregation_level,
            )

            # ---------------------------------------------------------#
            # Update log with iteration ################################
            # ---------------------------------------------------------#
            step_log.add_record(revenue, serviced, rejected)

            # ---------------------------------------------------------#
            # Plotting fleet activity ################################ #
            # ---------------------------------------------------------#
            if plot:
                plot_fleet(amod.cars, trips, n, step + 1)
                # time.sleep(0.5)

            # # Show time step statistics
            # step_log.show_info()

            # # What each vehicle is doing?
            # amod.print_fleet_stats()

        # amod.print_car_traces()

        # -------------------------------------------------------------#
        # Compute episode info #########################################
        # -------------------------------------------------------------#
        print(
            f"####### "
            f"[Episode {n:>5}] "
            f"- {episodeLog.last_episode_stats()} "
            f"#######"
        )
        episodeLog.compute_episode(
            step_log, weights=amod.get_weights(), progress=True
        )

    episodeLog.compute_learning()


# sim(False)
# if __name__ == "__main__":
thread = Thread(target=partial(sim, plot))
thread.start()
