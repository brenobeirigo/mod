# Mobility-on-demand fleet management application

## Installation

Locate the transportation environment module at file `tenv\network.py`:

    tenv_mod = "C:\\Users\\LocalAdmin\\OneDrive\\leap_forward\\street_network_server\\tenv"

## ADP config

state = TIME, LOCATION, BATTERY, CONTRACT, CARTYPE, CARORIGIN
decision = ACTION, POSITION, BATTERY, CONTRACT_DURATION, CAR_TYPE, CAR_ORIGIN, ORIGIN, DESTINATION, SQ_CLASS, N_DECISIONS

## Loading intances from files

Indicate the instance file by setting variable:

    instance_name = f"INSTANCE_PATH/exp_settings.json"


## Running the simulator

The command below will start the simulation on the port 5002.

    bokeh serve --show --port 5002 tests\ml\live_simulation.py

The time between each assignment can be set by changing the variable STEP_DELAY (seconds) in the PlotTrack class.

## Running the service rate and fleet status graphs

    bokeh serve --show --port 5003 mod\visual\slide_episode.py

## Creating the environoment

Execute the following commands:

    conda config --prepend channels conda-forge
    conda create -n env_slevels python=3.7
    conda activate env_slevels
    conda install --strict-channel-priority osmnx
    conda install gurobi
    conda install gurobipy
    conda install -c conda-forge --strict-channel-priority h3-py
    conda install bokeh
    pip install seaborn


To activate the environment use:

    C:\Users\LocalAdmin\Anaconda353\Scripts\activate env_slevels
    C:\Users\breno\Anaconda3\Scripts\activate env_slevels

## Execution

### Logging

| Keyword | Function |
|--------:|----------|
|level LOG_LEVEL| Choose logging level LOG_LEVEL = [INFO, DEBUG].|
|log_mip | Save mip model (`.lp`) and mip solution log (`.log`) for each time step and iteration.|
|log_summary | Show fleet stats for every step.|
|log_all | Log all steps (MIP decisions, dual extraction, and VFA update).|
|log_fleet | For each iteration `n`, save `cars_n.csv`  and `cars_result_n.csv` log files in `fleet/fleet_data/` comprising the information of each car in the beginning and in the end of iteration `n`, respectively. The following fields are considered for each vehicle: <br> `id, type, point, waypoint, previous, origin, middle_point, elapsed_distance, time_o_m, distance_o_m, elapsed, remaining_distance, step_m, idle_step_count, interrupted_rebalance_count, tabu, battery_level, trip, point_list, arrival_time, previous_arrival, previous_step, step, revenue, n_trips, distance_traveled, status, curret_trip, time_status, contract_duration`.|
|log_trips | For each iteration `n`, save (1) `trips_n.csv`  and (2) `trips_result_n.csv` log files in `trip_samples_data/` comprising the information of each trip request in the beginning and in the end of iteration `n`, respectively. <br>The following fields are considered in (1): <br> `placement_datetime, pk_id, dp_id, sq_class, max_delay, tolerance, passenger_count, pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude`.<br>The following fields are considered in (2): <br> `placement_datetime,pk_id,dp_id,sq_class,max_delay,tolerance,passenger_count,pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,pickup_step,pickup_delay,pickup_datetime,dropoff_time,dropoff_datetime,picked_by`.|
|log_times | Log the times (per iteration) to create decisions, extract duals, realize decisions, update duals, setup costs, setup constraints, and optimize MIP. Also include the total time.|
|save_plots| Save iteration's demand and fleet statuses for each step.|
|save_df  | Save .csv dataframes with fleet (status, vehicle count) and demand (met, unmet) data|

### Scenario configuration

| Keyword | Function |
|--------:|----------|
|hire| Consider FAV hiring. Related settings: DEPOT_SHARE, FAV_FLEET_SIZE, MAX_CONTRACT_DURATION.|
|n "number of iterations"| Set the number of iterations.|
|FLEET_SIZE "number of vehicles" | Set the fleet size.|
|backlog "extra_waiting"| Activate user backlogging. Rejected users are re-inserted into the demand for `times = extra_waiting/time_increment`. E.g., if the period is `1 min` and a unmatched user can wait up to `10 min`, this user is re-inserted in the demand pool throughout the next 9 iterations.|

### Optimization methods

The folowing keywords can be used to define the optimization method to assign and rebalance vehicles. For a single instance, each keyword creates a separate folder where you can find the results of each method.

| Keyword | Method |
|--------:|--------|
|train "n"| Run ADP algorithm on the training dataset to create VFAs. Update the file progress.npy with the updated results for each visited state every "n" iterations (by default, "n" is 1). |
|test| Run ADP algorithm on the testing dataset and uses the VFAs (from training) to measure the impact of future decisions.|
|myopic| Barebone assignment algorith with no rebalancing.|
|policy_random| Assignment algorithm with random rebalancing.|
|policy_reactive | Assignment algorithm and Alonso-Mora rebalancing (send vehicles to unmet requests but interrupt rebalancing at each period to check whether new requests can be picked up).

## Demand

Enable `USE_CLASS_PROB` to load probabilities of picking up users of different classes throughout time and space.
If disabled, user membership to classes is defined at random according to the proportions defined in `TRIP_CLASS_PROPORTION`. For example, to consider 80% of the demand is comprised of "B" users and the remaining is comprised of "A" users, we can do as follows:

    TRIP_CLASS_PROPORTION: (("A", 0), ("B", 1))

## Execution

Configure the json and run it passing folder and file. For example:

    main.py 'c:/tud/mod/config/adp_tune/' 'TS_2K-B-1-1.json'

Batch file (`.bat`):

    call C:\Users\LocalAdmin\Anaconda3\Scripts\activate env_slevels
    call cd C:\Users\LocalAdmin\OneDrive\leap_forward\phd_project\reb\code\mod\
    call python mod\ml\adp_network_server.py R -FLEET_SIZE 300 -n 500 -train -save_plots

## Memory profiling

Decorate the function you would like to profile with `@profile` and execute passing the option `-m memory_profiler` (e..g, `python -m memory_profiler tenv\util.py`)

## Printing functools

print(amod.cost_func.cache_info())

## Installing cython

https://wiki.python.org/moin/WindowsCompilers

Execution:
`python setup.py build_ext --inplace`

## MIP
### Gurobi error codes

Once an optimize call has returned, the Gurobi optimizer sets the Status attribute of the model to one of several [possible values](https://www.gurobi.com/documentation/6.0/refman/optimization_status_codes.html). E.g.: 


| Status code     | Value | Description |
|-----------------|-------|-------------|
| LOADED          | 1     | Model is loaded but no solution information is available.|
| OPTIMAL         | 2     | Model was solved to optimality (subject to tolerances)  and an optimal solution is available.|
| INFEASIBLE      | 3     | Model was proven to be infeasible.|
| INF_OR_UNBD     | 4     | Model was proven to be either infeasible or unbounded.|

### Gurobi method
Algorithm used to solve continuous models or the root node of a MIP model ([info](https://www.gurobi.com/documentation/8.1/refman/method.html#parameter:Method)). Options are: 
 * -1=automatic
 * 0=primal simplex
 * 1=dual simplex
 * 2=barrier
 * 3=concurrent
 * 4=deterministic concurrent
 * 5=deterministic concurrent simplex

Example:
    
    m.setParam("Method", 1)

### Gurobi MIP focus

If you believe the solver is having no trouble finding good quality solutions, and wish to focus more attention on proving optimality, select MIPFocus=2.
If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound ([info](https://www.gurobi.com/documentation/8.1/refman/mipfocus.html)).

If you are more interested in finding feasible solutions quickly, you can select MIPFocus=1.

Example:

    m.setParam("MIPFocus", 1)

### AMoD


## Online and loaded data

If you define `amod = AmodNetworkHired(config, online=True)` the following is activated:

    # Is data calculated on-the-fly?
    # Calculate everything on the fly
    if online:
        self.revenue = self.online_revenue 
        self.cost = self.online_costs
        self.penalty = self.online_penalty

    # Access data from dictionary
    else:
        self.load_od_data()
        self.revenue = self.loaded_revenue
        self.cost = self.loaded_costs
        self.penalty = self.loaded_penalty

If `online=False`, revenue, cost, and penalty data are loaded into dictionaries.

# Logged information


| Keyword | Description |
|---------|--------|
|step=NUMBER| Indicate the start of step.|
|Reachable| List of reachable trips.|
|Unreachable| List of trips that cannot be reached by any vehicle.|
|TRIP COUNT ORIGIN| Number of trips per origin location.|
|TRIP COUNT DESTINATON| Number of trips per destination location.|
|LOG COSTS [DECISION, SOLUTIONS]| Cost calculation of decision set(DECISION -> COST + DISCOUNT * POST = TOTAL).|
|SOLUTIONS| DECISION = TIMES APPLIED|
|Car attributes| List of cars and attributes (id, position)|
|NODE - NEIGHBORHOOD| Table: node_id, levels_ids, unreachable, neighborhood|
|ATTRIBUTE CAR COUNT| Table: g, step, inbound, cars, unrestricted|

## Methods
### Reactive rebalancing

    1 - Match idle/rebalancing cars to trips (ignore rebalancing decisions)
    2 - Update fleet, such that non-matched cars that were previosly rebalancing, continue to do so.
    3 - Match rejected & outstanding trip origins (the rebalancing targets) to idle vehicles.

## Tuning


| Keyword | Method |
|--------:|--------|
|TEST_LABEL| Label of all tuning instances (default "TUNE") .|
|N_PROCESSES| Number of tuning istances executed in parallel (by default, 2).|
|FOCUS| Key to select pre-configured tuning instances.|
|METHOD| Which optimization method is used in the tuning (-train, -test, -reactive, -myopic).|

Example:

    python tests\ml\tuning.py <TEST_LABEL> <N_PROCESSES>  <METHOD> <FOCUS>
    python tests\ml\tuning.py SH 9 -train hiring

## Notebooks

To compare the weight put in each level in the hierarchical aggregation process, use the file notebooks/

# Data

## Map

The street network consists of a strongly connected directed graph G=(N,E) comprised of N=6,430 nodes and E=11,581 edges.

### Distance matrix

The distance matrix in meters (float):

    dist_matrix_m.csv

The distance matrix in seconds considering a 20km/h speed (integer):

    dist_matrix_duration_s.csv

### Node list

The node list file `nodeset_gps.json` contains the following information:

```json
{
    "nodes": [
        {
            "id": 0,
            "x": -73.9600434,
            "y": 40.7980486
        },
        ...
        {
            "id": 6429,
            "x": -73.93256054530173,
            "y": 40.860556269812896
        }
    ]
}
```
### Node list and regional centers

The node set file `nodeset_info.json` contains the following information:

```json
{
    "nodes": [
        {
            "id": 0,
            "step_center": {
                "60": 1326,
                "300": 2679,
                "600": 6392
            },
            "x": -73.9600434,
            "y": 40.7980486
        },
    ...
        {
            "id": 6429,
            "step_center": {
                "60": 1258,
                "300": 4506,
                "600": 6429
            },
            "x": -73.93256054530173,
            "y": 40.860556269812896
        }
}
```
The `step_center` attribute stores the regional center of each node, reachable within 60, 300, and 600 seconds.

## First class distrubution

In order to assign service quality classes to the requests, we assume that the first-class user locations and request times coincide with the 20\% most generous tippers (among tipping users) of the Manhattan demand occurring between 5AM and 9AM.
To make our sample more representative, we first aggregate all demand data from 2011 Tuesdays.
Then, we assign first-class labels to all requests whose tip/fare ratio ranks over the 80<sup>th</sup> percentile (which is around  0.26).

We aggregate probabilities in 5 min bins, such that 24h = 288 bins. Example:

{
0   [0:00 - 0:05),
1   [0:05 - 0:10),
...,
59  [5:00 - 5:05),
60  [5:05 - 5:10),
...,
107 [8:55 - 9:00),
...,
287 [23:55 - 00:00)
}

The 1<sup>st</sup> class probability file is `1st_class_prob_info.npy`. The structure is as follows:

```json
{
    "time_bin": 5,
    "start": "5:00:00",
    "end": "9:00:00",
    "data": {
        0: {
            67: 0.5,
            74: 0.5,
            82: 0.25,
            83: 0.2,
            88: 0.16,
            89: 0.18,
            90: 0.25,
            91: 0.22, 
            92: 0.2,
            93: 0.16,
            94: 0.2,
            95: 0.16,
            96: 0.16,
            97: 0.11,
            98: 0.08,
            100: 0.12,
            105: 0.22,
            106: 0.2,
            107: 0.2
        }
        ...
    }
}
```
The attribute `data` stores for each node id the probabilities associated with the appearance of 1<sup>st</sup> class probabilities at each time bin.
When a pair (node id, time bin) does not exist, the probability is zero.

## Pickup and delivery requests

Excerpts from the Manhatan taxicab dataset where coordinates are approximated to the closest node in N within a 50-meter range.

### Training data

The ADP estimates value functions by randomly sampling 10\% of the request demand of a single day (Tuesday, 2011-04-12) in in the interval [5AM, 9AM).

The training data correspond to file:
    
    tripdata_ids_2011-04-12_000000_2011-04-12_235959.csv

### Testing data

The quality of the value functions estimated using the training data is tested against the remaining 51 Tuesdays of the year.
Again, a 10\% sample in the interval [5AM, 9AM) is used for comparison.

The testing data correspond to  files:

    tripdata_ids_2011-01-04_000000_2011-01-04_235959.csv
    tripdata_ids_2011-01-11_000000_2011-01-11_235959.csv
    tripdata_ids_2011-01-18_000000_2011-01-18_235959.csv
    tripdata_ids_2011-01-25_000000_2011-01-25_235959.csv
    tripdata_ids_2011-02-01_000000_2011-02-01_235959.csv
    tripdata_ids_2011-02-08_000000_2011-02-08_235959.csv
    tripdata_ids_2011-02-15_000000_2011-02-15_235959.csv
    tripdata_ids_2011-02-22_000000_2011-02-22_235959.csv
    tripdata_ids_2011-03-01_000000_2011-03-01_235959.csv
    tripdata_ids_2011-03-08_000000_2011-03-08_235959.csv
    tripdata_ids_2011-03-15_000000_2011-03-15_235959.csv
    tripdata_ids_2011-03-22_000000_2011-03-22_235959.csv
    tripdata_ids_2011-03-29_000000_2011-03-29_235959.csv
    tripdata_ids_2011-04-05_000000_2011-04-05_235959.csv
    tripdata_ids_2011-04-19_000000_2011-04-19_235959.csv
    tripdata_ids_2011-04-26_000000_2011-04-26_235959.csv
    tripdata_ids_2011-05-03_000000_2011-05-03_235959.csv
    tripdata_ids_2011-05-10_000000_2011-05-10_235959.csv
    tripdata_ids_2011-05-17_000000_2011-05-17_235959.csv
    tripdata_ids_2011-05-24_000000_2011-05-24_235959.csv
    tripdata_ids_2011-05-31_000000_2011-05-31_235959.csv
    tripdata_ids_2011-06-07_000000_2011-06-07_235959.csv
    tripdata_ids_2011-06-14_000000_2011-06-14_235959.csv
    tripdata_ids_2011-06-21_000000_2011-06-21_235959.csv
    tripdata_ids_2011-06-28_000000_2011-06-28_235959.csv
    tripdata_ids_2011-07-05_000000_2011-07-05_235959.csv
    tripdata_ids_2011-07-12_000000_2011-07-12_235959.csv
    tripdata_ids_2011-07-19_000000_2011-07-19_235959.csv
    tripdata_ids_2011-07-26_000000_2011-07-26_235959.csv
    tripdata_ids_2011-08-02_000000_2011-08-02_235959.csv
    tripdata_ids_2011-08-09_000000_2011-08-09_235959.csv
    tripdata_ids_2011-08-16_000000_2011-08-16_235959.csv
    tripdata_ids_2011-08-23_000000_2011-08-23_235959.csv
    tripdata_ids_2011-08-30_000000_2011-08-30_235959.csv
    tripdata_ids_2011-09-06_000000_2011-09-06_235959.csv
    tripdata_ids_2011-09-13_000000_2011-09-13_235959.csv
    tripdata_ids_2011-09-20_000000_2011-09-20_235959.csv
    tripdata_ids_2011-09-27_000000_2011-09-27_235959.csv
    tripdata_ids_2011-10-04_000000_2011-10-04_235959.csv
    tripdata_ids_2011-10-11_000000_2011-10-11_235959.csv
    tripdata_ids_2011-10-18_000000_2011-10-18_235959.csv
    tripdata_ids_2011-10-25_000000_2011-10-25_235959.csv
    tripdata_ids_2011-11-01_000000_2011-11-01_235959.csv
    tripdata_ids_2011-11-08_000000_2011-11-08_235959.csv
    tripdata_ids_2011-11-15_000000_2011-11-15_235959.csv
    tripdata_ids_2011-11-22_000000_2011-11-22_235959.csv
    tripdata_ids_2011-11-29_000000_2011-11-29_235959.csv
    tripdata_ids_2011-12-06_000000_2011-12-06_235959.csv
    tripdata_ids_2011-12-13_000000_2011-12-13_235959.csv
    tripdata_ids_2011-12-20_000000_2011-12-20_235959.csv
    tripdata_ids_2011-12-27_000000_2011-12-27_235959.csv