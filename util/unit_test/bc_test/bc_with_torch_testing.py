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


[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_exterior_edges(FORCES_MASK)


[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = exterior_edge_to_interior_edge(edge_hl_x=edge_hl_x, 
        edge_hl_y=edge_hl_y, edge_hr_x=edge_hr_x, edge_hr_y=edge_hr_y, edge_vb_x=edge_vb_x, edge_vb_y=edge_vb_y, edge_vt_x=edge_vt_x, edge_vt_y=edge_vt_y)



get_line_distribution(object_mask=FORCES_MASK)