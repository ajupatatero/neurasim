Interface
=========

The way of interacting with NeuraSim is with Command Line Inputs (CLI) and configuration files.
The CLIs consists of a series of predefined commands that executes different functions of the software. And that at the same time, accept argumnets. Wheter they serve to configure some option for the command itself, or wheter they serve to configure the function that the command executes. For instance the simulation function.
On the other hand, the files consists of a set of YAML files for the different aspects. 


Commands
--------

Simulate
^^^^^^^^
*Description: Command to execute the simulation of the given case defined in config_simulation.yaml.

*Usage: simulate [-key <arg>] [--long_key <arg>]

+------------------------------------------------+---------------+---------------------+---------------+
| Argument Description                           | Usage (Short) | Usage (long)        | Default Value |
+================================================+===============+=====================+===============+
| Path of the configuration yaml files directory | -cd <'path'>  | --conf_dir <'path'> | './'          |
+------------------------------------------------+---------------+---------------------+---------------+
| Path of the output results files directory     | -co <'path'>  | --out_dir <'path'>  | './results/'  |
+------------------------------------------------+---------------+---------------------+---------------+
| Path of the input results files directory      | -id <'path'>  | --in_dir <'path'>   | './results/'  |
+------------------------------------------------+---------------+---------------------+---------------+


Analyze
^^^^^^^

*Usage: analyze [-key <arg>] [--long_key <arg>]

+-------------------------------------------------------+---------------+-----------------------+---------------+
| Argument Description                                  | Usage (Short) | Usage (long)          | Default Value |
+=======================================================+===============+=======================+===============+
| Path of the configuration yaml files directory        | -cd <'path'>  | --conf_dir <'path'>   | './'          |
+-------------------------------------------------------+---------------+-----------------------+---------------+
| Path of the output results files directory            | -co <'path'>  | --out_dir <'path'>    | './results/'  |
+-------------------------------------------------------+---------------+-----------------------+---------------+
| Path of the input results files directory             | -id <'path'>  | --in_dir <'path'>     | './results/'  |
+-------------------------------------------------------+---------------+-----------------------+---------------+
| Type of analysis to perform. Append as many as needed | -a <type>     | --analysis_type <type>|       --      |
+-------------------------------------------------------+---------------+-----------------------+---------------+

*Types:


Train
^^^^^^^

*Usage: train [-key <arg>] [--long_key <arg>]

+-------------------------------------------------------+---------------+-----------------------+---------------+
| Argument Description                                  | Usage (Short) | Usage (long)          | Default Value |
+=======================================================+===============+=======================+===============+
| Path of the configuration yaml files directory        | -cd <'path'>  | --conf_dir <'path'>   | './'          |
+-------------------------------------------------------+---------------+-----------------------+---------------+
| Path of the output results files directory            | -co <'path'>  | --out_dir <'path'>    | './results/'  |
+-------------------------------------------------------+---------------+-----------------------+---------------+
| Path of the input results files directory             | -id <'path'>  | --in_dir <'path'>     | './results/'  |
+-------------------------------------------------------+---------------+-----------------------+---------------+


Iterate
^^^^^^^

*Usage: iterate [-key <arg>] [--long_key <arg>]

+-------------------------------------------------------+---------------+-----------------------+---------------+
| Argument Description                                  | Usage (Short) | Usage (long)          | Default Value |
+=======================================================+===============+=======================+===============+
| Path of the configuration yaml files directory        | -cd <'path'>  | --conf_dir <'path'>   | './'          |
+-------------------------------------------------------+---------------+-----------------------+---------------+
| Path of the output results files directory            | -co <'path'>  | --out_dir <'path'>    | './results/'  |
+-------------------------------------------------------+---------------+-----------------------+---------------+
| Path of the input results files directory             | -id <'path'>  | --in_dir <'path'>     | './results/'  |
+-------------------------------------------------------+---------------+-----------------------+---------------+
| Callable(function, command) or script to iterate over | -e <callable> | --execute <callable>  |   simulate    |
+-------------------------------------------------------+---------------+-----------------------+---------------+

*Example: iterate -e "analyze -a velocity"







Configuration files
-------------------
The following are the parameters accepted in the config_simulation.yaml file. Notice not all are necessary to run a simulation.
At the same time the notation goes as follows:

yaml property name [magnitude/type] {default value}: explanation.


INPUT/OUTPUT PROPERTIES 
^^^^^^^^^^^^^^^^^^^^^^^
in_dir [path] {--}:
out_dir [path] {--}:


RUNNING OPTIONS
^^^^^^^^^^^^^^^
DEBUG [bool] {False}: to activate the debugging tag within the code.

MAX_TIME [ms] {23 hours}: maximum time of simulation before cutting it and preparing the relaunch of another job until completion.
This is useful when working with PANDO since its nodes have a limit of time. 

GPU [bool] {False}: Tag to run it on GPU instead of CPU.

FP64 [bool] {False}: Tag to activate Floating Point precision of 64 bit instead of 32.

save_performance [bool] {False}: Option to calculate and save the performance.

save_field [bool] {False}: Option to save the fields.

save_field_x_ite [int] {200}: save the fields each x iterations.

plot_field [bool] {True}: Option to plot fields during execution of run.

plot_x_ite [int] {10}: plot and save the plots each x iterations.

plot_field_gif [bool] {True}: if plot_field, plot gif option.

plot_field_steps [bool] {True}: if plot_field, plot timesteps.   

post_computations [bool] {True}: Option to perform the post calculations regarding forces, and probes, etc.

save_post_x_ite [int] {100}: Save the post computation calculations variables each x iterations.


SIMULATION OPTIONS
^^^^^^^^^^^^^^^^^^
sim_method [] {--}: simulation class to be executed.

load_path [path] {--}: if convnet, the path of the saved architecture.

network_name [str] {--}: if convnet, the name of the network.

new_train [str] {--}: if convnet, if it is a network trained using fluidnet or phiflow(neurasim) [new->neurasim, old->fluidnet].

normalization [] {--}: if convnet + new_train, the parametrization parameter.

ite_transition [int] {0}: if convnet, the iteration to pass from CG start to NN.


precision [int] {1e-3}: TOLERANCE OR PRECISION OF THE LAPLACE MATRIX SOLVER (used in all solvers. i.e. in convnet before ite transition!).

max_iterations [int] {1e5}: MAX NUMBER OF ITERATIONS OF THE CG METHOD TO SOLVE THE LAPLACE EQUATION.


DISCRETITATION & GEOMETRY
^^^^^^^^^^^^^^^^^^^^^^^^^
Lx [distance ref] {--}: Longitude of the domain in the x axis. It can be mm or m if consistently changed. 

Ly [distance ref] {--}: Longitude of the domain in the y axis. It can be mm or m if consistently changed. 

Nx [int] {--}: number of control volumes in the x axis.

Ny [int] {--}: number of control volumes in the y axis.

Nt [int] {--}: number of time iterations.

CFL [float] {--}: CFL condition