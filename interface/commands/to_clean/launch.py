from neurasim import *
from datetime import date
import os

''' 
Treats the input parameters, wheter it is a yaml file with the pando parameter to launc. Or the variables as arguments.
And then creates the slurm file and launches it.

Notice it also has as an argument the script/command to execute
'''

def main():
    #Read yaml and arguments and treatment
    parser = LaunchParser()
    command, config = parser.parse()

    for i in range(20):
        if not os.path.isfile(f'{config['conf_dir']}/launch_{os.path.basename(os.path.dirname(config['conf_dir']))}_run_{i}.slurm'):
            break

    run_launch=f'./launch_{os.path.basename(os.path.dirname(config['conf_dir']))}_run_{i}.slurm'


    #TODO: launch only creates the sbatch file and launch the command and folder

    #Create launch slurm file
    with open(run_launch) as fh:
        fh.writelines("#!/bin/bash\n")

        fh.writelines(f"\n\n# LAUNCHER FILE AUTOMATICALLY GENERATED ON {date.now().strftime('%d/%m/%Y %H:%M:%S')} {}\n")

        for key,value in config.items():
            if not key == 'conf_dir':
                fh.writelines(f"#SBATCH --{key}={value}\n")
        #fh.writelines("module load python/3.7")   
        fh.writelines("source $HOME/fluidnet_env/bin/activate")  
        fh.writelines(f"{command}")  #arguments join in command as dict

    #Launch slurm file
    os.system(f"sbatch {run_launch}")

if __name__ == '__main__':
    main()