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
D=4


DOMAIN = Domain(x=Nx, y=Ny, boundaries=[OPEN, STICKY], bounds=Box[0:Lx, 0:Ly])

velocity = ((DOMAIN.staggered_grid(Noise(batch=1)) * 0 )+1) *(1,1)
 
BOX_MASK = HardGeometryMask(Box[xD-D:xD+D, Ly/2-D:Ly/2+D]) >> DOMAIN.scalar_grid()  
FORCES_MASK = HardGeometryMask(Sphere([xD, Ly/2], radius=D/2 )) >> DOMAIN.scalar_grid() 

zoom_pos=[xD + D -1, xD + D +1, 
        Ly/2 + D -1, Ly/2 + D +1 ] 


vl = -1
vr = 2
vb = -1
vt = 2


vl = 0
vr = 0
vb = 0
vt = 0


#FUNCTION

[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_exterior_edges(FORCES_MASK)
[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = exterior_edge_to_interior_edge(edge_hl_x=edge_hl_x, 
        edge_hl_y=edge_hl_y, edge_hr_x=edge_hr_x, edge_hr_y=edge_hr_y, edge_vb_x=edge_vb_x, edge_vb_y=edge_vb_y, edge_vt_x=edge_vt_x, edge_vt_y=edge_vt_y)


velocity = to_numpy(velocity)
u = velocity[0]
v = velocity[1]

plot_field(FORCES_MASK,
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
        save=True, filename=f'{out_dir}normal_test_in.png')


#Set normal velocities
u[edge_hl_x, edge_hl_y] = vl
u[edge_hr_x +1, edge_hr_y] = vr

v[edge_vb_x, edge_vb_y] = vb
v[edge_vt_x, edge_vt_y + 1] = vt



#Pass to phiflow
velocity = to_staggered([u,v], Lx, Ly)

# vel= torch.zeros((1, 2, Nx+1, Ny+1))
# vel[0,0,:,:] = torch.from_numpy(u)
# vel[0,1,:,:] = torch.from_numpy(v)

# velocity_init =  DOMAIN.staggered_grid(1)
# tensor_U = math.wrap(vel.cuda(), 'batch,vector,x,y')
# lower = math.wrap(velocity_init.box.lower)
# upper = math.wrap(velocity_init.box.upper)
# extrapolation = math.extrapolation.ZERO
# tensor_U_unstack = unstack_staggered_tensor(tensor_U)

# velocity =  StaggeredGrid(tensor_U_unstack, geom.Box(lower, upper), extrapolation)


# END FUNCTION


####################################################
######################################################
#FINAL FUNCTION
##################################3


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

velocity = set_normal_bc(FORCES_MASK, velocity = velocity, velocity_BC = [0,0,0,0])

plot_field(FORCES_MASK,
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
        save=True, filename=f'{out_dir}normal_test_out.png')
