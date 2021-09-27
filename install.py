import os, sys


#TODO: import from requirements

dependencies =['pyyaml', 'argparse', 'colorama', 'scipy', 'numpy', 'torch', 'matplotlib', 'sphinx', 'myst-parser', 'ptflops'] 

EPATH="$HOME/"
FPATH="./"

#Default values
PANDO=False
DEVELOPE=False
UPDATE=False

print(f"NEURASIM {sys.platform} Easy Installer")

#Treat Arguments Parsed
try:
    for arg in sys.argv[1:]:
      if arg in ('-p','--pando'):
          PANDO=arg
      elif arg in ('-d','--develope'):
          DEVELOPE=arg
      elif arg in ('-u','--update'):
          UPDATE=arg
except:
    print ('install.py {--pando, -p} <True>  {--develope, -d} <True> {--update, -u} <False>')
    sys.exit(2)

#Detect Platform
#Windows (win32, win64), Linux (linux2) y Mac (darwin)
if sys.platform in ('linux2', 'linux', 'darwin'):
    LINUX = True
elif sys.platform in ('win32', 'win64'):
    WINDOWS = True
else:
    print('>>Neurasim [FATAL ERROR]: Operating Sytem not suported.')
    exit()

#Create Python Enviorement and or Load Envoirement
if PANDO and LINUX:
    if not UPDATE:
        print(f">>Neurasim: Creating enviorement...")
        os.system(f"python3 -m venv {EPATH}fluidnet_env")
    print(f">>Neurasim: Loading enviorement...")
    os.system(f"source {EPATH}fluidnet_env/bin/activate")

#Install Dependencies
for i, dependency in enumerate(dependencies):
    print(f">>Neurasim: Installing {dependency}...")
    os.system(f"pip install {dependency}")

#Install/Update Neurasim                        #TODO error in installing on raiman test???
if not UPDATE:
    print(f">>Neurasim: Installing Neurasim...")
    os.system(f"python {FPATH}setup.py {'develop' if DEVELOPE else ''}")
elif UPDATE:
    print(f">>Neurasim: Updating Neurasim from git...")
    os.system(f"git {FPATH}pull")

#Configure Bash on PANDO
if PANDO and LINUX:
    # configure .bashrc etc alias, colors, etc
    NotImplemented