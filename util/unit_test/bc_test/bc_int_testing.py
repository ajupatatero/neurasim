import numpy as np
from engines.phi.torch.flow import *
from numpy.core import shape_base
from scipy.signal.filter_design import _vratio
from util.plot.plot_field import *
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
D=4


DOMAIN = Domain(x=Nx, y=Ny, boundaries=[OPEN, STICKY], bounds=Box[0:Lx, 0:Ly])

velocity = ((DOMAIN.staggered_grid(Noise(batch=1)) * 0 )+1) *(1,1)
 
BOX_MASK = HardGeometryMask(Box[xD-D:xD+D, Ly/2-D:Ly/2+D]) >> DOMAIN.scalar_grid()  
FORCES_MASK = HardGeometryMask(Sphere([xD, Ly/2], radius=D/2 )) >> DOMAIN.scalar_grid() 

zoom_pos=[xD + D -1, xD + D +1, 
        Ly/2 + D -1, Ly/2 + D +1 ] 





#FUNCTION

[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_exterior_edges(BOX_MASK)
[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = exterior_edge_to_interior_edge(edge_hl_x=edge_hl_x, 
        edge_hl_y=edge_hl_y, edge_hr_x=edge_hr_x, edge_hr_y=edge_hr_y, edge_vb_x=edge_vb_x, edge_vb_y=edge_vb_y, edge_vt_x=edge_vt_x, edge_vt_y=edge_vt_y)



#velocity = velocity*(0,0)

#velocity = set_normal_bc(BOX_MASK, velocity = velocity, velocity_BC = [0,0,0,0])
#velocity = set_tangential_bc(BOX_MASK, velocity = velocity, velocity_BC = [0,0,0,0])


velocity = to_numpy(velocity)
u = velocity[0]
v = velocity[1]

plot_field(BOX_MASK,
        plot_type=['surface'],
        options=[ ['limits', [0, 1]],
        #['full_zoom', True],
        ['zoom_position', zoom_pos],
        ['aux_contourn', False],   
        ['grid', True],
        ['edges', [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ]],
        ['velocity', velocity],
        ],
        Lx=Lx, Ly=Ly, dx=dx, dy=dy,
        lx='x', ly='y',lbar='mask',
        save=True, filename=f'{out_dir}interior_test_zoom_in2.png')


#Set normal velocities

line_x_sorted, line_y_sorted, angle_sorted = get_line_distribution(edges=[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ])

up_x = line_x_sorted[:int(len(line_x_sorted)/2)]
up_y = line_y_sorted[:int(len(line_y_sorted)/2)]

down_x = line_x_sorted[int(len(line_x_sorted)/2):]
down_y = line_y_sorted[int(len(line_y_sorted)/2):]

for i in range(len(up_x)):
        u[up_x[i], down_y[i]:up_y[i]+1] = 0
        v[up_x[i], down_y[i]:up_y[i]+1] = 0





#Pass to phiflow
velocity = to_staggered([u,v], Lx, Ly)

# END FUNCTION

plot_field(BOX_MASK,
        plot_type=['surface'],
        options=[ ['limits', [0, 1]],
        #['full_zoom', True],
        ['zoom_position', zoom_pos],
        ['aux_contourn', False],   
        ['grid', True],
        ['edges', [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ]],
        ['velocity', velocity],
        ],
        Lx=Lx, Ly=Ly, dx=dx, dy=dy,
        lx='x', ly='y',lbar='mask',
        save=True, filename=f'{out_dir}interior_test_zoom_out2.png')
