""" Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
"""
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Label, Title, Legend
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from tornado import gen
from bokeh.document import without_document_lock
from functools import partial

import scipy.ndimage.filters as fi
import os
import sys

import pandas as pd
from collections import defaultdict

plot_width = 900
plot_height = 500
start_slider = 0
end_slider = 25
update_rate = 0.1 * 1000 * 60  # 5 min
axes_font_size = "20px"
label_font_size = "20px"
title_font_size = "20px"
smooth_sigma_demand = 0
smooth_sigma_fleet = 0


doc = curdoc()

# exp_name = "DF_LIN_V=0500-0000(R)_I=5_L[4]=(10-0-, 11-0-, 12-0-, 13-0-)_R=([1-6, 2-6, 3-3][L(05)]_T=[06h,+60m+06h+30m]_1.00(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_0.00_B_2.40_15.00_0.00_0.00_1.00"
# exp_name = "DF2_LIN_V=0500-0000(R)_I=5_L[4]=(10-0-, 11-0-, 12-0-, 13-0-)_R=([1-6, 2-6, 3-3][L(05)]_T=[06h,+60m+06h+30m]_1.00(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_0.00_B_2.40_15.00_0.00_0.00_1.00"
# exp_name = "DF_LIN_V=0400-0000(R)_I=5_L[4]=(10-0-, 11-0-, 12-0-, 13-0-)_R=([1-6, 2-6, 3-3][L(05)]_T=[06h,+60m+06h+30m]_1.00(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_0.00_B_7.20_10.00_0.00_0.00_1.00"
exp_name = "B15_LIN_V=0400-0000(R)_I=5_L[4]=(10-0-, 11-0-, 12-0-, 13-0-)_R=([1-6][L(05)]_T=[06h,+60m+06h+30m]_1.00(S)_1.00_0.10_A_4.80_5.00_5.00_4.80_0.00_B_7.20_15.00_0.00_0.00_1.00"
# method = "reactive"
method = "adp/train"
path_fleet = f"C:/Users/LocalAdmin/OneDrive/leap_forward/phd_project/reb/code/mod/data/output/{exp_name}/{method}/fleet/data/"
path_demand = f"C:/Users/LocalAdmin/OneDrive/leap_forward/phd_project/reb/code/mod/data/output/{exp_name}/{method}/service/data/"
path_overall_stats = f"C:/Users/LocalAdmin/OneDrive/leap_forward/phd_project/reb/code/mod/data/output/{exp_name}/{method}/overall_stats.csv"


# Color per vehicle status
color_fleet_status = {
    "Parked": "#24aafe",
    "With passenger": "#53bc53",
    "Repositioning": "firebrick",
    "Recharging": "#e55215",
    "Returning": "#e55215",
    "Driving to pick up": "#e55215",
    "Total": "magenta",
}

# Color per vehicle status
color_dict_demand = {
    "Total demand": "#24aafe",
    "Serviced demand": "#53bc53",
    "Battery level": "#e55215",
}

# Ignored statuses in read data
drop_status_list = ["Recharging", "Returning"]

# Store vehicle count per status at each step of an episode
episode_fleet_dict = defaultdict(dict)

# Store (un)met trips at each step of an episode
episode_demand_dict = defaultdict(lambda: defaultdict(dict))


plot_fleet = figure(plot_width=plot_width, plot_height=plot_height)
plot_demand = figure(plot_width=plot_width, plot_height=plot_height)
plot_iterations = figure(plot_width=plot_width, plot_height=plot_height)

# Setup slider
episode_slider = Slider(
    title="Iteration",
    width=800,
    align="center",
    value=start_slider,
    start=start_slider,
    end=end_slider,
    step=1,
)

# Save column data source per status
source_fleet = dict()
source_demand = dict()
source_overall_stats = dict()


def configure_plot_demand(p):

    # Save glyph per status
    items = dict()
    for status in ["Total demand", "Serviced demand"]:
        data = ColumnDataSource(data=dict(x=[], y=[]))
        source_demand[status] = data
        line = p.line(
            "x",
            "y",
            color=color_dict_demand[status],
            line_width=2,
            line_alpha=1,
            muted_alpha=0.1,
            source=data,
        )

        items[status] = [line]

    # Setup legend
    # https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#legends

    legend = Legend(
        items=list(items.items()),
        location="center_right",
        background_fill_alpha=0.8,
        click_policy="mute",
        border_line_width=0,
        label_text_font_size=label_font_size,
        title_text_font_size=title_font_size,
        title_text_font_style="bold",
        # title="Demand",
    )

    # Setup title
    title = Title(
        align="center",
        text=f"Iteration {episode_slider.value:>4}",
        text_font_size="16pt",
        text_color="#929292",
        text_font_style="normal",
    )

    # Add legend outside plot
    p.add_layout(legend, "right")

    # Add title
    p.title = title

    # Set x axis settings
    p.xaxis.axis_label = "Step"
    p.xaxis.major_label_text_font_size = axes_font_size
    p.xaxis.axis_label_text_font_size = axes_font_size

    # Set y axis settings
    p.yaxis.axis_label = "#Trips"
    p.yaxis.major_label_text_font_size = axes_font_size
    p.yaxis.axis_label_text_font_size = axes_font_size


def configure_plot_fleet(p, status_list):

    # Save glyph per status
    items = dict()
    for status in status_list:
        data = ColumnDataSource(data=dict(x=[], y=[]))
        source_fleet[status] = data
        line = p.line(
            "x",
            "y",
            color=color_fleet_status[status],
            line_width=2,
            line_alpha=1,
            muted_alpha=0.1,
            source=data,
        )

        items[status] = [line]

    # Setup legend
    # https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#legends

    legend = Legend(
        items=list(items.items()),
        location="center_right",
        background_fill_alpha=0.8,
        click_policy="mute",
        border_line_width=0,
        label_text_font_size=label_font_size,
        title_text_font_size=title_font_size,
        title_text_font_style="bold",
        title="Vehicle status",
    )

    # Setup title
    title = Title(
        align="center",
        text=f"Iteration {episode_slider.value:>4}",
        text_font_size="16pt",
        text_color="#929292",
        text_font_style="normal",
    )

    # Add legend outside plot
    p.add_layout(legend, "right")

    # Add title
    p.title = title

    # Set x axis settings
    p.xaxis.axis_label = "Step"
    p.xaxis.major_label_text_font_size = axes_font_size
    p.xaxis.axis_label_text_font_size = axes_font_size

    # Set y axis settings
    p.yaxis.axis_label = "#Vehicle/Status"
    p.yaxis.major_label_text_font_size = axes_font_size
    p.yaxis.axis_label_text_font_size = axes_font_size


def configure_overall_stats(p, status_list):

    # Save glyph per status
    items = dict()
    for status in status_list:
        data = ColumnDataSource(data=dict(x=[], y=[]))
        source_overall_stats[status] = data
        line = p.line(
            "x",
            "y",
            color="red",
            line_width=2,
            line_alpha=1,
            muted_alpha=0.1,
            source=data,
        )

        items[status] = [line]
    print(items)
    # Setup legend
    # https://bokeh.pydata.org/en/latest/docs/user_guide/styling.html#legends

    legend = Legend(
        items=list(items.items()),
        location="center_right",
        background_fill_alpha=0.8,
        click_policy="mute",
        border_line_width=0,
        label_text_font_size=label_font_size,
        title_text_font_size=title_font_size,
        title_text_font_style="bold",
        # title="Vehicle status",
    )

    # Setup title
    title = Title(
        align="center",
        text="Cost",
        text_font_size="16pt",
        text_color="#929292",
        text_font_style="normal",
    )

    # Add legend outside plot
    p.add_layout(legend, "right")

    # Add title
    p.title = title

    # Set x axis settings
    p.xaxis.axis_label = "Iteration"
    p.xaxis.major_label_text_font_size = axes_font_size
    p.xaxis.axis_label_text_font_size = axes_font_size

    # Set y axis settings
    p.yaxis.axis_label = "#Cost"
    p.yaxis.major_label_text_font_size = axes_font_size
    p.yaxis.axis_label_text_font_size = axes_font_size


def load_episode(e, smooth_sigma=0):
    print(f"Loading episode {e:04}...")

    # Create episode file path
    file_path = path_fleet + f"e_fleet_status_{e:04}.csv"

    # Read dataframe corresponding to episode
    d = pd.read_csv(file_path, index_col=[0])

    # Drop existing labels in drop_status_list
    d = d.drop(drop_status_list, axis=1, errors="ignore")
    # x axis of step values
    steps = np.array(d.index.values)

    # Create total column
    d["Total"] = d.apply(
        lambda row: sum([row[status] for status in d.columns.values]), axis=1
    )

    # Loading episode data
    for status in d.columns.values:
        count = list(d[status])

        # Smooth values
        if smooth_sigma > 0:
            count = fi.gaussian_filter1d(count, sigma=smooth_sigma)

        e_data_dict = dict(x=steps, y=count)
        source_fleet[status].data = e_data_dict
        episode_fleet_dict[e][status] = count


def load_overall_stats(smooth_sigma=0):

    # Create episode file path
    file_path = path_overall_stats

    # Read dataframe corresponding to episode
    d = pd.read_csv(file_path, index_col=[0])

    # x axis of step values
    iterations = np.array(d.index.values)

    # Loading episode data
    # for status in d.columns.values:
    for status in ["Total reward"]:
        count = list(d[status])
        # Smooth values
        if smooth_sigma > 0:
            count = fi.gaussian_filter1d(count, sigma=smooth_sigma)

        e_data_dict = dict(x=iterations, y=count)
        source_overall_stats[status].data = e_data_dict


def load_episode_demand(e, smooth_sigma=0):
    print(f"Loading episode {e:04}...")

    # Create episode file path
    file_path = path_demand + f"e_demand_status_{e:04}.csv"

    # Read dataframe corresponding to episode
    d = pd.read_csv(file_path, index_col=[0])

    # Get service rate
    service_rate = d["Serviced demand"].sum() / d["Total demand"].sum()

    # x axis of step values
    steps = np.array(d.index.values)

    # Loading episode data
    # for status in d.columns.values:
    for status in ["Total demand", "Serviced demand"]:
        count = list(d[status])
        # Smooth values
        if smooth_sigma > 0:
            count = fi.gaussian_filter1d(count, sigma=smooth_sigma)

        e_data_dict = dict(x=steps, y=count)
        source_demand[status].data = e_data_dict
        episode_demand_dict[e]["status_count"][status] = count

    episode_demand_dict[e]["service_rate"] = service_rate


@gen.coroutine
def show_fleet_status(episode):
    """Update line chart with the car count per status and time step for
    episode e.

    Parameters
    ----------
    e : int
        pisode number
    """

    # Check if episode was previously loaded
    if episode not in episode_fleet_dict:
        load_episode(episode, smooth_sigma=smooth_sigma_fleet)

    else:
        print(f"Showing episode {episode:04}...")
        for status, count in episode_fleet_dict[episode].items():
            source_fleet[status].data["y"] = count

    plot_fleet.title.text = f"Iteration {episode:>4}"


@gen.coroutine
def show_overall_stats():

    load_overall_stats(smooth_sigma=smooth_sigma_fleet)
    # source_overall_stats[status].data
    # for status, count in source_overall_stats.items():
    #     source_overall_stats[status].data["y"] = count


@gen.coroutine
def show_demand_status(episode):
    """Update line chart with the car count per status and time step for
    episode e.

    Parameters
    ----------
    e : int
        pisode number
    """

    # Check if episode was previously loaded
    if episode not in episode_demand_dict:
        load_episode_demand(episode, smooth_sigma=smooth_sigma_demand)

    else:
        print(f"Showing episode {episode:04}...")
        for status, count in episode_demand_dict[episode][
            "status_count"
        ].items():
            source_demand[status].data["y"] = count

    s_rate = episode_demand_dict[episode]["service_rate"]
    plot_demand.title.text = f"Iteration {episode:>4} ({s_rate:>7.2%})"


def update_data(attrname, old, new):
    # Get the current slider values
    episode = episode_slider.value

    doc.add_next_tick_callback(partial(show_fleet_status, episode))
    doc.add_next_tick_callback(partial(show_demand_status, episode))


episode_slider.on_change("value", update_data)


# @gen.coroutine
def update_episode_list():
    print(f"Updating episode track ({update_rate//(60*1000)} min)...")
    global end_slider
    global episode_slider
    files_demand = os.listdir(path_demand)
    # files_fleet = os.listdir(path_fleet)
    end_slider = len(files_demand)
    episode_slider.end = end_slider
    doc.add_next_tick_callback(partial(show_overall_stats))

    # print(files_demand)
    # print(files_fleet)


# Setup plots (must be the column headers of .csv files)
status_list = [
    "Repositioning",
    "Parked",
    "With passenger",
    "Driving to pick up",
    "Recharging",
    "Returning",
    "Total",
]
list_stats = ["Total reward"]

configure_plot_fleet(plot_fleet, status_list)
configure_plot_demand(plot_demand)
configure_overall_stats(plot_iterations, list_stats)

# Start plot from first episode
update_data("value", start_slider, start_slider)
update_episode_list()

doc.add_root(
    column(row(plot_fleet, plot_demand), plot_iterations, episode_slider)
)
doc.title = "Iteration history"
doc.add_periodic_callback(update_episode_list, update_rate)
