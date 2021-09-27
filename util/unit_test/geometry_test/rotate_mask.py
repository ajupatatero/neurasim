import numpy as np
from engines.phi.torch.flow import *
from numpy.core import shape_base
from scipy.signal.filter_design import _vratio
from util.plot.plot_tools import *
from analysis.mesure import *
from neurasim import *


from util.operations.field_operate import *

out_dir='./'
Nx=100
Ny=100
Ly=100
Lx=100
dx=Lx/Nx
dy=Ly/Ny

xD=5
D=2

TORCH_BACKEND.set_default_device('GPU')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

DOMAIN = Domain(x=Nx, y=Ny, boundaries=[OPEN, STICKY], bounds=Box[0:Lx, 0:Ly])

INT = HardGeometryMask(Sphere([xD, Ly/2], radius=D/2 )) >> DOMAIN.scalar_grid() 
EXT = HardGeometryMask(InverseSphere([xD, Ly/2], radius=D*2 )) >> DOMAIN.scalar_grid() 
BLADES = get_blades_mask(DOMAIN, 0, 0, 0)

WR=0.05*D

velocity = ((DOMAIN.staggered_grid(Noise(batch=1)) * 0 )+1) *(0.0,0.0)


bc_mask, bc_value = get_obstacles_bc([[EXT, WR, True],[INT, -0.5*WR, False]  ]) 
velocity = apply_boundaries(velocity, bc_mask, bc_value)



plot_field(BLADES, #INT+EXT,
        plot_type=['surface'],
        options=[ ['limits', [0, 1]],
        ['aux_contourn', False],   
        ['grid', True],
        #['velocity', velocity],
        ],
        Lx=Lx, Ly=Ly, dx=dx, dy=dy,
        lx='x', ly='y',lbar='mask',
        save=True, filename=f'{out_dir}BLADES.png')