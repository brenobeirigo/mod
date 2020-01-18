# Mobility-on-demand fleet management application
## ADP config

state = TIME, LOCATION, BATTERY, CONTRACT, CARTYPE, CARORIGIN
decision = ACTION, POSITION, BATTERY, CONTRACT_DURATION, CAR_TYPE, CAR_ORIGIN, ORIGIN, DESTINATION, SQ_CLASS, N_DECISIONS

## Running the simulator

    bokeh serve --show --port 5002 tests\ml\live_simulation.py

## Running the service rate and fleet status graphs

    bokeh serve --show --port 5003 mod\visual\slide_episode.py

## Activating the environment

    C:\Users\LocalAdmin\Anaconda353\Scripts\activate env_slevels
    C:\Users\breno\Anaconda3\Scripts\activate env_slevels

## Execution


| Keyword | Function |
|---------|----------|
|save_df  | Save .csv dataframes with fleet (status, vehicle count) and demand (met, unmet) data|
|use_duas | Extract duals from solution and use them to approximate the value function of the states.|
|save_progress| Update a progress file (progress.npy) after n iterations (default n=1) with the current value functions.|
|level LOG_LEVEL| Choose logging level LOG_LEVEL = [INFO, DEBUG].|
|log_mip | Save mip model (`.lp`) and mip solution log (`.log`) for each time step and iteration.|
|save_plots| Save iteration's demand and fleet statuses for each step.|
|n N_ITERATON| Set the number of iterations.|
|FLEET_SIZE | Set the fleet size.|

Example to log adp and mip execution, save progress.npy file, and plots.

    python mod\ml\adp_network_server.py TEST_NAME -save_progress 10 -log_adp -log_mip -save_plots -save_df -level DEBUG -n 300 -FLEET_SIZE 500

Execution (only save progress):

    python mod\ml\adp_network_server.py REB_T -FLEET_SIZE 500 -n 2000 -save_df -save_progress 10

Hiring:

    python mod\ml\adp_network_server.py hiring -FLEET_SIZE 300 -n 500 -save_progress 10 -save_plots -hire

Batch file (`.bat`):

    call C:\Users\LocalAdmin\Anaconda3\Scripts\activate env_slevels
    call cd C:\Users\LocalAdmin\OneDrive\leap_forward\phd_project\reb\code\mod\
    call python mod\ml\adp_network_server.py REB_T -FLEET_SIZE 500 -n 2000 -save_df -save_progress

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