# Mobility-on-demand fleet management application

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
    pip install h3
    conda install gurobipy
    conda install -c conda-forge --strict-channel-priority h3-py
    conda istall bokeh


To activate the environment use:

    C:\Users\LocalAdmin\Anaconda353\Scripts\activate env_slevels
    C:\Users\breno\Anaconda3\Scripts\activate env_slevels

## Execution

### Logging

| Keyword | Function |
|---------:|----------|
|level LOG_LEVEL| Choose logging level LOG_LEVEL = [INFO, DEBUG].|
|log_mip | Save mip model (`.lp`) and mip solution log (`.log`) for each time step and iteration.|
|log_adp | Log all ADP phases (MIP decisions, dual extraction, and VFA update).|
|log_fleet | For each iteration `n`, save `cars_n.csv`  and `cars_result_n.csv` log files in `fleet/fleet_data/` comprising the information of each car in the beginning and in the end of iteration `n`, respectively. The following fields are considered for each vehicle: <br> `id, type, point, waypoint, previous, origin, middle_point, elapsed_distance, time_o_m, distance_o_m, elapsed, remaining_distance, step_m, idle_step_count, interrupted_rebalance_count, tabu, battery_level, trip, point_list, arrival_time, previous_arrival, previous_step, step, revenue, n_trips, distance_traveled, status, curret_trip, time_status, contract_duration`.|
|log_trips | For each iteration `n`, save (1) `trips_n.csv`  and (2) `trips_result_n.csv` log files in `trip_samples_data/` comprising the information of each trip request in the beginning and in the end of iteration `n`, respectively. <br>The following fields are considered in (1): <br> `placement_datetime, pk_id, dp_id, sq_class, max_delay, tolerance, passenger_count, pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude`.<br>The following fields are considered in (2): <br> `placement_datetime,pk_id,dp_id,sq_class,max_delay,tolerance,passenger_count,pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,pickup_step,pickup_delay,pickup_datetime,dropoff_time,dropoff_datetime,picked_by`.|
|log_times | Log the times (per iteration) to create decisions, extract duals, realize decisions, update duals, setup costs, setup constraints, and optimize MIP. Also include the total time.|
|save_plots| Save iteration's demand and fleet statuses for each step.|
|save_df  | Save .csv dataframes with fleet (status, vehicle count) and demand (met, unmet) data|

### Scenario configuration

| Keyword | Function |
|---------:|----------|
|hire| Consider FAV hiring. Related settings: DEPOT_SHARE, FAV_FLEET_SIZE, MAX_CONTRACT_DURATION.|
|n "number of iterations"| Set the number of iterations.|
|FLEET_SIZE "number of vehicles" | Set the fleet size.|
|backlog| Activate user backlogging. Rejected users are re-inserted into the demand with discounted maximum waiting time. E.g., if the period is `1 min` and a unmatched user can wait up to `10 min`, this user is re-inserted in the deamnd pool with maximum waiting time of `9 min`.|

### Optimization methods
The folowing keywords can be used to define the optimization method to assign and rebalance vehicles. For a single instance, each keyword creates a separate folder where you can find the results of each method.

| Keyword | Method |
|--------:|----------|
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

Example to create a case study named "R" considering 300 vehicles. The method will run on the training instance (defined in the file `config.py`) for 500 iterations. Additionally, all plots will be saved (fleet usage, demand, and user delays).

    python mod\ml\adp_network_server.py R -FLEET_SIZE 300 - n 500 -train -save_plots

Hiring:

    python mod\ml\adp_network_server.py R -FLEET_SIZE 300 - n 500 -train -save_plots -FAV_FLEET_SIZE 200

Batch file (`.bat`):

    call C:\Users\LocalAdmin\Anaconda3\Scripts\activate env_slevels
    call cd C:\Users\LocalAdmin\OneDrive\leap_forward\phd_project\reb\code\mod\
    call python mod\ml\adp_network_server.py R -FLEET_SIZE 300 - n 500 -train -save_plots

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

## Tuning


| Keyword | Method |
|--------:|----------|
|TEST_LABEL| Label of all tuning instances (default "TUNE") .|
|N_PROCESSES| Number of tuning istances executed in parallel (by default, 2).|
|FOCUS| Key to select pre-configured tuning instances.|
|METHOD| Which optimization method is used in the tuning (-train, -test, -reactive, -myopic).|

Example:

    python tests\ml\tuning.py <TEST_LABEL> <N_PROCESSES>  <METHOD> <FOCUS>
    python tests\ml\tuning.py SH 9 -train hiring
