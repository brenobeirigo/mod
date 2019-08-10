# Mobility-on-demand fleet management application

## Running the simulator

    bokeh serve --show --port 5002 mod\env\simulator.py

## Running the service rate and fleet status graphs

    bokeh serve --show --port 5003 mod\visual\slide_episode.py

## Activating the environment

    C:\Users\LocalAdmin\Anaconda353\Scripts\activate env_slevels

## Execution
Example to log adp and mip execution, save progress.npy file, and plots.

    python mod\ml\adp_network_server.py TEST_NAME -save_progress -log_adp -log_mip -save_plots -level DEBUG -n 300 -FLEET_SIZE 500

Execution (only save progress):

    python mod\ml\adp_network_server.py TEST_NAME -save_progress -n 300 -FLEET_SIZE 500

## Memory profiling

Decorate the function you would like to profile with `@profile` and execute passing the option `-m memory_profiler`.
