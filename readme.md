# FluidNet (version 3)
Project to accelerate the resolution of the Poisson step in CFD solvers, by means of convolutional neural networks.
This project is based on PhiFlow v2 and [FluidNet](https://github.com/google/FluidNet) 

Please check the original [PhiFlow v2](https://github.com/tum-pbs/PhiFlow) code, which has been used to complement this code. This work has been developed respecting the Phiflow license agreement.


---

# Installation

## 1.Requirements

Dependencies:

import argparse
import yaml
import glob

import re
import traceback
from colorama import init, Fore, Back, Style


scipy
numpy
matplotlib
imageio
torch

Memory:



## 2.Setup
To install the software please follow the following instructions and commands. First of all, clone the git repository to its local folder. Then, execute the installer which will install all the missing dependencies, configure the enviorement, and install the fluidnet package. Make sure to specify whether it is a developing setup or not. Meaning if the pip install is editable or not. 

```
cd <path where to clone the repository>

git clone [--branch <branchname] repository

module load python/3.7

python ./fluidnet_3/scripts/install.py {--pando, -p} <True>  {--develope, -d} <True> {--update, -u} <False>

```


# Usage and Guides
[To consult the full documentation as well as all the guides and miscellanious. Go to the Sphinx generated HTML index. Or click on this link.](doc/build/html/index.html)


# Contributors

# Acknowledgements
