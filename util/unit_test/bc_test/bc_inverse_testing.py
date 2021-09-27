import numpy as np
from engines.phi.torch.flow import *
from numpy.core import shape_base
from scipy.signal.filter_design import _vratio
from util.plot.plot_tools import *
from analysis.mesure import *
from neurasim import *


from util.operations.field_operate import *

out_dir='./'
Nx=10
Ny=10
Ly=10
Lx=10
dx=Lx/Nx
dy=Ly/Ny

xD=5
D=2


DOMAIN = Domain(x=Nx, y=Ny, boundaries=[OPEN, STICKY], bounds=Box[0:Lx, 0:Ly])

INT = HardGeometryMask(Sphere([xD, Ly/2], radius=D/2 )) >> DOMAIN.scalar_grid() 
EXT = HardGeometryMask(InverseSphere([xD, Ly/2], radius=D*2 )) >> DOMAIN.scalar_grid() 

WR=0.05*D/2


#NORMAL
velocity = ((DOMAIN.staggered_grid(Noise(batch=1)) * 0 )+1) *(1,1) #*(0.0,0.0)
velocity = set_normal_bc(EXT,velocity=velocity,velocity_BC=[0,0,0,0]) #[1,1,1,1])

plot_field(INT+EXT,
        plot_type=['surface'],
        options=[ ['limits', [0, 1]],
        ['aux_contourn', False],   
        ['grid', True],
        ['velocity', velocity],
        ],
        Lx=Lx, Ly=Ly, dx=dx, dy=dy,
        lx='x', ly='y',lbar='mask',
        save=True, filename=f'{out_dir}normal_test_out_INVERSE.png')


velocity2 = ((DOMAIN.staggered_grid(Noise(batch=1)) * 0 )+1) *(1,1) #*(0.0,0.0)
velocity2 = set_normal_bc(INT,velocity=velocity2,velocity_BC=[0,0,0,0]) #[1,1,1,1])

plot_field(INT+EXT,
        plot_type=['surface'],
        options=[ ['limits', [0, 1]],
        ['aux_contourn', False],   
        ['grid', True],
        ['velocity', velocity2],
        ],
        Lx=Lx, Ly=Ly, dx=dx, dy=dy,
        lx='x', ly='y',lbar='mask',
        save=True, filename=f'{out_dir}normal_test_out_INVERSE_REF.png')


#TANGENTIAL
velocity = ((DOMAIN.staggered_grid(Noise(batch=1)) * 0 )+1) *(1,1) #*(0.0,0.0)
velocity = set_tangential_bc(EXT,velocity=velocity,velocity_BC=[0,0,0,0], inv_geom=True) #[1,1,1,1])

plot_field(INT+EXT,
        plot_type=['surface'],
        options=[ ['limits', [0, 1]],
        ['aux_contourn', False],   
        ['grid', True],
        ['velocity', velocity],
        ],
        Lx=Lx, Ly=Ly, dx=dx, dy=dy,
        lx='x', ly='y',lbar='mask',
        save=True, filename=f'{out_dir}tangential_test_out_INVERSE.png')



velocity2 = ((DOMAIN.staggered_grid(Noise(batch=1)) * 0 )+1) *(1,1) #*(0.0,0.0)
velocity2 = set_tangential_bc(INT,velocity=velocity2,velocity_BC=[0,0,0,0]) #[1,1,1,1])

plot_field(INT+EXT,
        plot_type=['surface'],
        options=[ ['limits', [0, 1]],
        ['aux_contourn', False],   
        ['grid', True],
        ['velocity', velocity2],
        ],
        Lx=Lx, Ly=Ly, dx=dx, dy=dy,
        lx='x', ly='y',lbar='mask',
        save=True, filename=f'{out_dir}tangential_test_out_INVERSE_REF.png')


#INTERIOR
velocity = ((DOMAIN.staggered_grid(Noise(batch=1)) * 0 )+1) *(1,1) 
velocity = set_interior_bc(EXT,velocity=velocity,velocity_BC=[0,0], inv_geom=True)

edges = get_interior_edges(EXT)
plot_field(INT+EXT,
        plot_type=['surface'],
        options=[ ['limits', [0, 1]],
        ['aux_contourn', False],   
        ['grid', True],
        ['velocity', velocity],
        ['edges', edges]
        ],
        Lx=Lx, Ly=Ly, dx=dx, dy=dy,
        lx='x', ly='y',lbar='mask',
        save=True, filename=f'{out_dir}interior_test_out_INVERSE.png')



velocity2 = ((DOMAIN.staggered_grid(Noise(batch=1)) * 0 )+1) *(1,1) 
velocity2 = set_interior_bc(INT,velocity=velocity2,velocity_BC=[0,0]) 

plot_field(INT+EXT,
        plot_type=['surface'],
        options=[ ['limits', [0, 1]],
        ['aux_contourn', False],   
        ['grid', True],
        ['velocity', velocity2],
        ],
        Lx=Lx, Ly=Ly, dx=dx, dy=dy,
        lx='x', ly='y',lbar='mask',
        save=True, filename=f'{out_dir}interior_test_out_INVERSE_REF.png')