import numpy as np
from engines.phi.torch.flow import *
from util.plot.plot_field import *
from analysis.mesure import *
from util.operations.field_operate import *


out_dir='./'
Nx=50
Ny=50
Ly=50
Lx=50
dx=Lx/Nx
dy=Ly/Ny

xD=25
D=10


TORCH_BACKEND.set_default_device('GPU')
DOMAIN = Domain(x=Nx, y=Ny, boundaries=[OPEN, STICKY], bounds=Box[0:Lx, 0:Ly])
velocity = ((DOMAIN.staggered_grid(Noise(batch=1)) * 0 )+1) *(0.0,0.0)
 

INT = HardGeometryMask(Sphere([xD, Ly/2], radius=D/2 )) >> DOMAIN.scalar_grid() 
EXT = HardGeometryMask(InverseSphere([xD, Ly/2], radius=D*2 )) >> DOMAIN.scalar_grid() 




WR=0.05*D/2
#velocity = set_rotate_bc(EXT, velocity = velocity, w=0, wr=WR)

bc_mask, bc_value = get_obstacles_bc([ [INT, WR], ]) #self.w*(self.D/2)


velocity = apply_boundaries(velocity, bc_mask, bc_value)


plot_field(INT+EXT,
        plot_type=['surface'],
        options=[ ['limits', [0, 1]],
        ['aux_contourn', False],   
        ['grid', True],
        ['velocity', velocity],
        ],
        Lx=Lx, Ly=Ly, dx=dx, dy=dy,
        lx='x', ly='y',lbar='mask',
        save=True, filename=f'{out_dir}NEW_BC.png')
