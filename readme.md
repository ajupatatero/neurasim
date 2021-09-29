# FluidNet (version 3)
Project to accelerate the resolution of the Poisson step in CFD solvers, by means of convolutional neural networks.
This project is based on PhiFlow v2 and [FluidNet](https://github.com/google/FluidNet) 

Please check the original [PhiFlow v2](https://github.com/tum-pbs/PhiFlow) code, which has been used to complement this code. This work has been developed respecting the Phiflow license agreement.


---

# Installation

## 1.Setup
To install the software please follow the following instructions and commands. First of all, clone the git repository to its local folder.
```
cd <path where to clone the repository>

git clone [--branch <branchname] https://reponame.git

```
Then, create a new environment with venv (or conda) and execute the installer which will install all the missing dependencies, configure the enviorement, and install the fluidnet package (Please make sure that Python3 is available, setting tested with python 3.6/7/8). 


```
python -m venv path/to/your/env  

source path/to/your/env/bin/activate

cd neurasim

python install.py
```

## 2. Launching test cases with trained networks

Plase note that the shown cases are developed to be launched in GPU cards, thus make sure that a GPU card is available in your computer or supercomputer. The networks used on this work are included in the git repository in the ```trained_networks/``` folder. Following the paper's notation, the networks with  the short term loss (STL), partially frozen long term loss (PF-LTL) and full long term loss (F-LTL) for several look ahead iterations (LAI) are:

Training Strategy |   LAI  | Network name
----------------- | ------ | ------------
STL               |    -   | ```nolt/Unet_nolt_3```
PF-LTL            |   2-4  | ```lt_nograd_2_4/Unet_nolt_grad_2_4```
PF-LTL            |   4-8  | ```lt_nograd_4_8/Unet_nolt_grad_4_8```
PF-LTL            |   4-16 | ```lt_nograd_4_16/Unet_nolt_grad_4_16```
F-LTL             |   1-2  | ```lt_grad_1_2/Unet_lt_grad_1_2```
F-LTL             |   2-4  | ```lt_grad_2_4/Unet_lt_grad_2_4```
F-LTL             |   2-6  | ```lt_grad_2_6/Unet_lt_grad_2_6```

To launch the cases create a ```cases``` folder outside the repository. Then, create a folder for the desired test case, and copy the desired confi_file from ```doc/config_files```. For example to recreate a plume with cylinder case:

```
cd /your/path/to/launch

mkdir cases
mkdir cases/plume

cp /path/to/neurasim/doc/config_files/config_simulation_plume_cyl.yaml ./cases/plume

cd cases/plume
```

Once the wanted configuration file is copied and modified to match the desired configuration, entry-points are used to launch the simulation, so just type:


```
simulate -cd config_simulation_plume_cyl.yaml
```

## 3. Retraining networks

To retrain new network architectures, or to try new hyperparameter configurations, a training dataset should be created. In this works case, the same **2D dataset** as the original FluidNet [Section 1: Generating the data - Generating training data](https://github.com/google/FluidNet#1-generating-the-data) (generated with MantaFlow) is used. Please carefully follow the intructions to generate the dataset, which should occupy around 48Gb.

The dataset file structure should be located in ```<dataDir>``` folder with the following structure: 
```
.
└── dataDir
    └── dataset
        ├── te
        └── tr

```
To train a network, first create a folder outside of the repository (for example in the previously introduced ```cases```folder, and copy the training congifuration found in ```path/to/neurasim/doc/config_files/config_train.yaml```

```
cd cases
mkdir train
cd train
cp path/to/neurasim/doc/config_files/config_train.yaml .
```

Precise the location of the dataset in ```pytorch/config_files/trainConfig.yaml``` writing the folder location at ```dataDir``` (__use absolute paths__).
Precise also ```dataset``` (name of the dataset), and output folder ```modelDir```where the trained model and loss logs will be stored and the model name ```modelFilename```. Once this is done, the hyperparameteres should be tuned. Particularly, the following options are highlighted:

* ```modelParam: divLongTermLambda:``` If set to 0 no long term is used (STL training), whereas for LTL trainings usually it is set to 5.
* ```modelParam: ltGrad:``` If true the intermediate gradients are computed (F-LTL) and if set to False the PF-LTL training is performed
* ```longTermDivNumSteps:``` Number of LAI.

Once this is done, thanks to the entry points, to launch the training just type:

```
train
```
