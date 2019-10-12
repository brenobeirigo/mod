# Mobility-on-demand fleet management application

## Running the simulator

    bokeh serve --show --port 5002 tests\ml\live_simulation.py

## Running the service rate and fleet status graphs

    bokeh serve --show --port 5003 mod\visual\slide_episode.py

## Activating the environment

    C:\Users\LocalAdmin\Anaconda353\Scripts\activate env_slevels

## Execution
Example to log adp and mip execution, save progress.npy file, and plots.

    python mod\ml\adp_network_server.py TEST_NAME -use_duals -save_progress -log_adp -log_mip -save_plots -save_df -level DEBUG -n 300 -FLEET_SIZE 500

| Keyword | Function |
|---------|----------|
|save_df  | Save .csv dataframes with fleet (status, vehicle count) and demand (met, unmet) data|
|use_duas | Extract duals from solution and use them to approximate the value function of the states.|
|save_progress| Update a progress file (progress.npy) after each iteration with the current value functions.|
|level LOG_LEVEL| Choose logging level LOG_LEVEL = [INFO, DEBUG].|
|log_mip | Save mip model (`.lp`) and mip solution log (`.log`) for each time step and iteration.|
|save_plots| Save iteration's demand and fleet statuses for each step.|
|n N_ITERATON| Set the number of iterations.|
|FLEET_SIZE | Set the fleet size.|

Execution (only save progress):

    python mod\ml\adp_network_server.py REB_T -FLEET_SIZE 500 -n 2000 -save_df -use_duals -save_progress

Batch file (`.bat`):

    call C:\Users\LocalAdmin\Anaconda3\Scripts\activate env_slevels
    call cd C:\Users\LocalAdmin\OneDrive\leap_forward\phd_project\reb\code\mod\
    call python mod\ml\adp_network_server.py REB_T -FLEET_SIZE 500 -n 2000 -save_df -use_duals -save_progress

## Memory profiling

Decorate the function you would like to profile with `@profile` and execute passing the option `-m memory_profiler` (e..g, `python -m memory_profiler tenv\util.py`)

## Printing functools

print(amod.cost_func.cache_info())

## Installing cython

https://wiki.python.org/moin/WindowsCompilers

Execution:
`python setup.py build_ext --inplace`


## Gurobi error codes

Once an optimize call has returned, the Gurobi optimizer sets the Status attribute of the model to one of several [possible values](https://www.gurobi.com/documentation/6.0/refman/optimization_status_codes.html). E.g.: 


| Status code     | Value | Description |
|-----------------|-------|-------------|
| LOADED          | 1     | Model is loaded but no solution information is available.|
| OPTIMAL         | 2     | Model was solved to optimality (subject to tolerances)  and an optimal solution is available.|
| INFEASIBLE      | 3     | Model was proven to be infeasible.|
| INF_OR_UNBD     | 4     | Model was proven to be either infeasible or unbounded.|
