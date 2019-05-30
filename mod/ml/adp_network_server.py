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

from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure

from tornado import gen

from collections import defaultdict

from bokeh.tile_providers import get_provider, Vendors

tile_provider = get_provider(Vendors.CARTODBPOSITRON)

# this must only be modified from a Bokeh session callback
source = ColumnDataSource(data=dict(x=[0], y=[0]))
source_points_g = ColumnDataSource(data=dict(x=[0], y=[0]))
source_points_r = ColumnDataSource(data=dict(x=[0], y=[0]))


# This is important! Save curdoc() to make sure all threads
# see the same document.
doc = curdoc()


@gen.coroutine
def update(x, y):
    source_points_g.data = dict(x=[x], y=[y])
    source_points_r.data = dict(x=[y], y=[x])

    # source_points_g.data = up[Car.REBALANCE]


# p = figure(x_range=[0, 1], y_range=[0, 1])

# create a plot and style its properties
p = figure(
    title="My first interactive plot!",
    x_axis_type="mercator",
    y_axis_type="mercator",
    # sizing_mode='scale_width',
    plot_height=800,
    plot_width=800,
)
p.border_fill_color = "black"
p.background_fill_color = "white"
p.outline_line_color = None
p.grid.grid_line_color = None

p.add_tile(tile_provider)

# Bokeh plot
point_r = p.circle(
    x=[],
    y=[],
    size=8,
    color="red",
    fill_alpha=0.2,
    line_width=0,
    legend=Car.REBALANCE,
)

point_centers = p.circle(
    x=[],
    y=[],
    size=6,
    color="white",
    line_width=1,
    line_color="red",
    legend="Center",
)

point_g = p.circle(
    x=[],
    y=[],
    size=8,
    color="green",
    fill_alpha=0.2,
    line_width=0,
    legend=Car.ASSIGN,
)
point_s = p.circle(
    x=[],
    y=[],
    size=8,
    color="blue",
    fill_alpha=0.2,
    line_width=0,
    legend=Car.IDLE,
)
point_b = p.circle(
    x=[],
    y=[],
    size=15,
    color="black",
    line_width=0,
    fill_alpha=0.5,
    legend=Car.RECHARGING,
)

point_o = p.circle(
    x=[],
    y=[],
    size=15,
    color="white",
    fill_alpha=0.5,
    line_width=1,
    line_color="green",
    legend="Origins",
)
point_d = p.circle(
    x=[],
    y=[],
    size=15,
    color="white",
    fill_alpha=0.5,
    line_width=1,
    line_color="red",
    legend="Destinations",
)

point_regular = p.circle(
    x=[],
    y=[],
    size=1,
    color="black",
    fill_alpha=0.5,
    line_width=0.1,
    legend="regular",
)

lines = p.multi_line([], [], line_color="red", line_alpha=0.05)

source_points = {
    Car.REBALANCE: point_r,
    Car.ASSIGN: point_g,
    Car.IDLE: point_s,
    Car.RECHARGING: point_b,
    "o": point_o,
    "d": point_d,
    "center": point_centers,
    "lines": lines,
    "regular": point_regular,
}

doc.add_root(p)


@gen.coroutine
def update_first(lines, regular, centers):

    source_points["regular"].data_source.data["x"] = regular["x"]
    source_points["regular"].data_source.data["y"] = regular["y"]
    source_points["center"].data_source.data = centers

    source_points["lines"].data_source.data["xs"] = lines["xs"]
    source_points["lines"].data_source.data["ys"] = lines["ys"]


@gen.coroutine
def update_timestep(status_fleet, trips):

    for status in status_fleet:
        source_points[status].data_source.data = status_fleet[status]

    for p in trips:
        source_points[p].data_source.data = trips[p]


def wgs84_to_web_mercator(lon, lat):
    k = 6378137
    x = lon * (k * np.pi / 180.0)
    y = np.log(np.tan((90 + lat) * np.pi / 360.0)) * k
    return x, y


def plot_centers(points, level):

    center_lines = defaultdict(list)
    regular = defaultdict(list)
    center_points = defaultdict(list)
    center_ids = set()

    for p in points:
        px, py = wgs84_to_web_mercator(p.x, p.y)
        regular["x"] += [px]
        regular["y"] += [py]

        cid = p.id_level(level)
        print(p.id, cid)
        c_point = points[cid]

        cx, cy = wgs84_to_web_mercator(c_point.x, c_point.y)
        center_lines["xs"].append([cx, px])
        center_lines["ys"].append([cy, py])

        if c_point.id not in center_ids:
            cx, cy = wgs84_to_web_mercator(c_point.x, c_point.y)
            center_points["x"].append(cx)
            center_points["y"].append(cy)
            center_ids.add(c_point.id)

    doc.add_next_tick_callback(
        partial(
            update_first,
            lines=center_lines,
            regular=regular,
            centers=center_points,
        )
    )


def plot_fleet(cars, trips):

    xy_status = defaultdict(lambda: defaultdict(list))

    for c in cars:
        x, y = wgs84_to_web_mercator(c.point.x, c.point.y)
        xy_status[c.status]["x"].append(x)
        xy_status[c.status]["y"].append(y)

    xy_trips = defaultdict(lambda: defaultdict(list))

    # Origin, destination coordinates
    for t in trips:
        ox, oy = wgs84_to_web_mercator(t.o.x, t.o.y)
        xy_trips["o"]["x"].append(ox)
        xy_trips["o"]["y"].append(oy)

        dx, dy = wgs84_to_web_mercator(t.d.x, t.d.y)
        xy_trips["d"]["x"].append(dx)
        xy_trips["d"]["y"].append(dy)

    doc.add_next_tick_callback(
        partial(update_timestep, status_fleet=xy_status, trips=xy_trips)
    )


def sim(plot=False):

    # -----------------------------------------------------------------#
    # Amod environment #################################################
    # -----------------------------------------------------------------#

    config = ConfigNetwork()
    config.update(
        {
            # Fleet
            ConfigNetwork.FLEET_SIZE: 500,
            ConfigNetwork.BATTERY_LEVELS: 20,
            # Time - Increment (min)
            ConfigNetwork.TIME_INCREMENT: 1,
            ConfigNetwork.OFFSET_REPOSIONING: 15,
            ConfigNetwork.OFFSET_TERMINATION: 30,
            # NETWORK ##################################################
            # Region centers are created in steps of how much time?
            ConfigNetwork.STEP_SECONDS: 30,
            # Demand spawn from how many centers?
            ConfigNetwork.ORIGIN_CENTERS: 3,
            # Cars rebalance to up to #region centers
            ConfigNetwork.N_CLOSEST_NEIGHBORS: 4,
            # Cars can access locations within region centers
            # established in which neighborhood level?
            ConfigNetwork.NEIGHBORHOOD_LEVEL: 3,
            ConfigNetwork.AGGREGATION_LEVELS: 10,
            ConfigNetwork.SPEED: 30,
        }
    )

    # ################################################################ #
    # Slice demand ################################################### #
    # ################################################################ #

    # What is the level covered by origin area?
    # E.g., levels 1, 2, 3 = 60, 120, 180
    # if level_origins = 3
    level_origins = 4

    # Data correspond to 1 day NY demand
    total_hours = 24
    earliest_hour = 0
    resize_factor = 1
    max_steps = int(total_hours * 60 / config.time_increment)
    earliest_step_min = int(earliest_hour * 60 / config.time_increment)

    # ---------------------------------------------------------------- #
    # Episodes ####################################################### #
    # ---------------------------------------------------------------- #
    episodes = 270
    episodeLog = EpisodeLog(config=config)
    amod = AmodNetwork(config)

    # ---------------------------------------------------------------- #
    # Plot centers and guidelines #################################### #
    # ---------------------------------------------------------------- #
    if plot:
        plot_centers(amod.points, config.neighborhood_level)
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
        origin_ids = episodeLog.load_origins()
        origins = [amod.points[p] for p in origin_ids]
        print(f"\n{len(origins)} origins loaded.")

    except:

        # Create random centers from where trips come from
        # TODO choose level to query origins
        origins = nw.query_demand_origin_centers(
            amod.points,
            amod.config.origin_centers,
            amod.config.get_step_level(level_origins),
        )

        print(f"\nSaving {len(origins)} origins.")
        episodeLog.save_origins([o.id for o in origins])

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

    destinations = nw.query_demand_origin_centers(
        amod.points,
        amod.config.origin_centers,
        amod.config.get_step_level(level_origins),
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
                plot_fleet(amod.cars, trips)

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


thread = Thread(target=sim)
thread.start()
