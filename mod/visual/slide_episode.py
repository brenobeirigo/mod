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
start_slider = 1
end_slider = 303
update_rate = 5 * 1000 * 60  # 5 min

smooth_sigma_demand = 1
smooth_sigma_fleet = 5


doc = curdoc()

# exp_name = "network_manhattan-island-new-york-city-new-york-usa_1500_0020_7_01_0030_02_02"
exp_name = "day_manhattan-island-new-york-city-new-york-usa_NYC_0001_0001_5_01_0030_02_04_1=8_05_05_01_0.1_01"
path_fleet = f"C:/Users/LocalAdmin/OneDrive/leap_forward/phd_project/reb/code/mod/data/output/{exp_name}/fleet/data/"
path_demand = f"C:/Users/LocalAdmin/OneDrive/leap_forward/phd_project/reb/code/mod/data/output/{exp_name}/service/data/"

# Color per vehicle status
color_dict_fleet = {
    "Idle": "#24aafe",
    "With passenger": "#53bc53",
    "Rebalancing": "firebrick",
    "Recharging": "#e55215",
    "Total": "black",
}

# Color per vehicle status
color_dict_demand = {
    "Total demand": "#24aafe",
    "Serviced demand": "#53bc53",
    "Battery level": "#e55215",
}

# Store vehicle count per status at each step of an episode
episode_fleet_dict = defaultdict(dict)

# Store (un)met trips at each step of an episode
episode_demand_dict = defaultdict(lambda: defaultdict(dict))


plot_fleet = figure(plot_width=plot_width, plot_height=plot_height)
plot_demand = figure(plot_width=plot_width, plot_height=plot_height)

# Setup slider
episode_slider = Slider(
    title="Episode",
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
        title_text_font_size="12px",
        title_text_font_style="bold",
        # title="Demand",
    )

    # Setup title
    title = Title(
        align="center",
        text=f"Episode {episode_slider.value:>4}",
        text_font_size="16pt",
        text_color="#929292",
        text_font_style="normal",
    )

    # Add legend outside plot
    p.add_layout(legend, "right")

    # Add title and axes
    p.title = title
    p.xaxis.axis_label = "Step"
    p.yaxis.axis_label = "#Trips"


def configure_plot_fleet(p):

    # Save glyph per status
    items = dict()
    for status in [
        "Rebalancing",
        "Idle",
        "With passenger",
        "Recharging",
        "Total",
    ]:
        data = ColumnDataSource(data=dict(x=[], y=[]))
        source_fleet[status] = data
        line = p.line(
            "x",
            "y",
            color=color_dict_fleet[status],
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
        title_text_font_size="12px",
        title_text_font_style="bold",
        title="Vehicle status",
    )

    # Setup title
    title = Title(
        align="center",
        text=f"Episode {episode_slider.value:>4}",
        text_font_size="16pt",
        text_color="#929292",
        text_font_style="normal",
    )

    # Add legend outside plot
    p.add_layout(legend, "right")

    # Add title and axes
    p.title = title
    p.xaxis.axis_label = "Step"
    p.yaxis.axis_label = "#Vehicle/Status"


def load_episode(e, smooth_sigma=0):
    print(f"Loading episode {e:04}...")

    # Create episode file path
    file_path = path_fleet + f"e_fleet_status_{e:04}.csv"

    # Read dataframe corresponding to episode
    d = pd.read_csv(file_path, index_col=[0])

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

    plot_fleet.title.text = f"Episode {episode:>4}"


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
    plot_demand.title.text = f"Episode {episode:>4} ({s_rate:>7.2%})"


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
    # print(files_demand)
    # print(files_fleet)


# Setup plots
configure_plot_fleet(plot_fleet)
configure_plot_demand(plot_demand)

# Start plot from first episode
update_data("value", start_slider, start_slider)
update_episode_list()

doc.add_root(column(row(plot_fleet, plot_demand), episode_slider))
doc.title = "Episode history"
doc.add_periodic_callback(update_episode_list, update_rate)
