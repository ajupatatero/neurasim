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


