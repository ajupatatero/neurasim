import os

import pdb

from progress.bar import Bar #pip install progress


import numpy as np #pip install numpy

from scipy.signal import hilbert
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.interpolate import griddata

import matplotlib #pip install matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import BoundaryNorm

import imageio #pip install imageio


from engines.phi.torch.flow import *
from engines.phi.physics._boundaries import STICKY
from engines.phi.field._grid import Grid, CenteredGrid, StaggeredGrid, unstack_staggered_tensor
from engines.phi.math._tensors import NativeTensor
from engines.phi.geom._inverse_sphere import InverseSphere

from interface.terminal.cmd_info import *
from interface.parser.command_parser import *
from interface.scripts.iterate import *

from simulation import *

from analysis.mesure import *
from analysis.meta_analysis import *
from analysis.post_proc import *

from util.plot.plot_tools import *
from util.operations.field_operate import *

from neural_models.architectures import *
from neural_models.dataset_managers.dataset import *
from neural_models.train.trainer import *
from neural_models.train.plot_train import *
from neural_models.train.util_train import *
from neural_models.train.train import *
