import os, sys


dependencies =['pyyaml', 'argparse', 'colorama', 'scipy', 'numpy', 'torch', 'matplotlib', 'sphinx', 'myst-parser', 'ptflops']

FPATH="./"


print(f"NEURASIM {sys.platform} Easy Installer")


#Detect Platform
#Windows (win32, win64), Linux (linux2) y Mac (darwin)
if sys.platform in ('linux2', 'linux', 'darwin'):
    LINUX = True
elif sys.platform in ('win32', 'win64'):
    WINDOWS = True
else:
    print('>>Neurasim [FATAL ERROR]: Operating Sytem not suported.')
    exit()

#Install Dependencies
for i, dependency in enumerate(dependencies):
    print(f">>Neurasim: Installing {dependency}...")
    os.system(f"pip install {dependency}")

#Install/Update Neurasim
print(f">>Neurasim: Installing Neurasim...")
os.system(f"python {FPATH}setup.py {''}")
