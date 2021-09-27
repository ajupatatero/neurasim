Setup
=====

To install the software please follow the following instructions and commands. First of all, clone the git repository to its local folder. Then, execute the installer which will install all the missing dependencies, configure the enviorement in case of working in PANDO, and install the fluidnet package. Make sure to specify whether it is a developing setup or not. Meaning if the pip install is editable or not. As well as, if it is installed on PANDO.


    cd <path where to clone the repository>
    
    git clone [--branch <branchname] https://gitlab.com/daep-ia/fluidnet_3.git
    
    module load python/3.7
    
    python ./fluidnet_3/scripts/install.py {--pando, -p} <True>  {--develope, -d} <True> {--update, -u} <False>

