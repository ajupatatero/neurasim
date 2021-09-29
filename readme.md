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

The networks used on this work are included in the git repository in the ```trained_networks/``` folder. 

# Contributors

# Acknowledgements
