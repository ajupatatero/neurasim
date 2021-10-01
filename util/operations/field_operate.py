import numpy as np
from scipy.signal.ltisys import dimpulse
from torch.functional import Tensor
from torch.nn.functional import batch_norm
from neurasim import *


from engines.phi import *
from engines.phi.field._grid import Grid, CenteredGrid, StaggeredGrid, unstack_staggered_tensor
from engines.phi.physics import *
from engines.phi.geom import *

import torch

#TYPE TRANSFORMATION FUNCTIONS
def to_torch(field):
    '''Function to make a detached copy on cpu from a tensor of StaggeredGrid or CenteredGrid.'''

    if isinstance(field,StaggeredGrid):
        Z0 = field.staggered_tensor().tensors[0]._native.detach().squeeze().cpu()
        Z1 = field.staggered_tensor().tensors[1]._native.detach().squeeze().cpu()
        Z = [Z0, Z1]
    elif isinstance(field,CenteredGrid):
        Z=field.values._native.detach().squeeze().cpu()
    else:
        #print('Field already different than StaggeredGrid or CenteredGrid.')
        Z = field

    return Z

def to_numpy2(field):
    if isinstance(field,StaggeredGrid) or isinstance(field,CenteredGrid):
        Z = to_torch(field)
        Z = Z.numpy()
    else:
        #print('Field already different than StaggeredGrid or CenteredGrid.')
        Z = field

    return Z

def to_numpy(field):
    if isinstance(field,StaggeredGrid):
        if list(field.shape.batch)[0] >= 1 if list(field.shape.batch) else False:
            Z0 = field.staggered_tensor().tensors[0]._native.detach().cpu().numpy()#[0]
            Z1 = field.staggered_tensor().tensors[1]._native.detach().cpu().numpy()#[0]
        else: #NB: in case there is some field without bactch size specified as 1. i.e. masks
            Z0 = field.staggered_tensor().tensors[0]._native.cpu().numpy()
            Z1 = field.staggered_tensor().tensors[1]._native.cpu().numpy()
        Z = [Z0, Z1]
    elif isinstance(field,CenteredGrid):
        if list(field.shape.batch)[0] >= 1 if list(field.shape.batch) else False:
            Z=field.values._native.detach().cpu().numpy()#[0]
        else:
            Z=field.values._native.cpu().numpy()
    else:
        Z=field



    return Z


def to_staggered(field, Lx, Ly):
    if isinstance(field,StaggeredGrid) or isinstance(field,CenteredGrid):
        error()
    else:
        if len(np.shape(field)) == 4:
            batch = np.shape(field)[1]
            dim = np.shape(field)[0]
            nnx = np.shape(field)[2]
            nny = np.shape(field)[3]
            u = field[0]
            v = field[1]
        elif len(np.shape(field)) == 3:
            batch = 1
            dim = np.shape(field)[0]
            nnx = np.shape(field)[1]
            nny = np.shape(field)[2]
            u = field[0]
            v = field[1]
        else:
            print('The input field must be at minimum a stacked list of 2D fields.')
            exit()


        vel= torch.zeros((batch, dim, nnx, nny))
        if isinstance(field,Tensor):
            vel[:,0,:,:] = u
            vel[:,1,:,:] = v
        else:
            vel[:,0,:,:] = torch.from_numpy(u)
            vel[:,1,:,:] = torch.from_numpy(v)


        # Auxiliary tensor to get correct atributes
        domain = Domain(x=nnx, y=nny, boundaries=CLOSED, bounds=Box[0:Lx, 0:Ly])

        # Velocity tensor created
        tensor_U = math.wrap(vel, 'batch,vector,x,y') #cuda() if putted as default better it already works. And then no problems

        # Useful elements for tensor generation
        extrapolation = math.extrapolation.ZERO

        tensor_U_unstack = unstack_staggered_tensor(tensor_U)
        velocity =  StaggeredGrid(tensor_U_unstack, domain.bounds, extrapolation) #geom.Box(lower, upper)

        return velocity

def to_centered(field):
    pass


#FIELD DATA FUNCTIONS
def get_dimensions(field):
    if isinstance(field,StaggeredGrid) or isinstance(field,CenteredGrid):

        if len(np.shape(field)) == 3 or len(np.shape(field)) == 4: #in case is staggered, the last one, i.e. shape[3] is the vector which will be 2
            Nbatch = field.shape[0]
            Nx = field.shape[1]
            Ny = field.shape[2]
        else:
            Nbatch = 0
            Nx = field.shape[0]
            Ny = field.shape[1]

        Lx = field.box.upper[0] - 0
        Ly = field.box.upper[1] - 0
        dx = field.dx[0]
        dy = field.dx[1]

    assert Lx/Nx == dx and Ly/Ny == dy, 'Problem with dx calculation'

    return Nbatch,Lx,Ly,Nx,Ny,dx,dy

def get_grid(field, Lx=None, Ly=None, dx=None, dy=None):
    #1.Get field physical properties
    if isinstance(field,StaggeredGrid) or isinstance(field,CenteredGrid):
        Nbatch,Lx,Ly,Nx,Ny,dx,dy = get_dimensions(field)
    else:
        assert Lx != None and Ly != None and dx != None and dy != None , 'If field is array or numpy, then it requires to pass the {Lx,Ly,dx,dy} values.'

    #2.1.Staggered grid == Cells walls
    cx = np.arange(0, Lx +dx/2, dx)
    cy = np.arange(0, Ly +dy/2, dy)
    #NOTICE: we include +dx in order to consider the last step in arrange

    #2.2.Value grid -> Centered or Staggered in function of value type
    if isinstance(field,StaggeredGrid):
        x = cx
        y = cy
    elif isinstance(field,CenteredGrid):
        x = np.arange(dx/2, Lx-dx/2 +dx/2, dx)
        y = np.arange(dy/2, Ly-dy/2 +dy/2, dy)
    elif isinstance(field,(list, tuple, np.ndarray)) and len(field)==Lx/dx + 1:
        x = cx
        y = cy
    elif isinstance(field,(list, tuple, np.ndarray)):
        x = np.arange(dx/2, Lx-dx/2 +dx/2, dx)
        y = np.arange(dy/2, Ly-dy/2 +dy/2, dy)

    Y, X = np.meshgrid(y,x)
    #NOTICE: Correct row-major, colum-major notation of numpy to adapt it to the phiflow notation
    #i.e. in numpy array(y,x), in phiflow and derivated array(x,y). Where x,y physical direction
    #i.e. x horizontal positive to the right, y vertical positive to the up.

    return x,y,X,Y,cx,cy


#OBJECTS MASKS & TREATMENT FUNCTIONS
def get_exterior_edges(object_mask):
    '''Function to extract the indeces of the exterior edges nodes corresponding with the input mask of the geometry.'''

    #1.Correct mask field in case is phiflow type and not array
    object_mask = to_torch(object_mask)

    #2.Extract the edges nodes
    #VERTICAL EDGES or WALLS (HORIZONTAL DIFFERENCE)
    edgesh = torch.where(object_mask[1:,:] != object_mask[:-1,:])

    #Vertical-Left walls
    edge_hl_x = edgesh[0][:int(len(edgesh[0])/2)]
    edge_hl_y = edgesh[1][:int(len(edgesh[1])/2)]

    #Vertical-Right walls
    edge_hr_x = edgesh[0][int(len(edgesh[0])/2):] + 1
    edge_hr_y = edgesh[1][int(len(edgesh[1])/2):]

    #HORIZONTAL EDGES or WALLS (VERTICAL DIFFERENCE)
    edgesv = torch.where(object_mask[:,1:] != object_mask[:,:-1])

    #Horizontal-Bottom walls
    edge_vb_x = edgesv[0][0::2]
    edge_vb_y = edgesv[1][0:-1:2]

    #Horizontal-Top walls
    edge_vt_x = edgesv[0][1::2]
    edge_vt_y = edgesv[1][1::2] + 1

    return [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ]

def exterior_edge_to_interior_edge(edge_hl_x=[], edge_hl_y=[], edge_hr_x=[], edge_hr_y=[], edge_vb_x=[], edge_vb_y=[], edge_vt_x=[], edge_vt_y=[]):
    '''Function to pass the nodes of an exterior edge to their corresponding interior ones.
    USAGE:

    '''

    #Apply the transformation correction
    edge_hl_x = edge_hl_x + 1
    edge_hr_x = edge_hr_x - 1
    edge_vb_y = edge_vb_y + 1
    edge_vt_y = edge_vt_y - 1

    return [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ]

def get_interior_edges(object_mask):
    [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_exterior_edges(object_mask)
    [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = exterior_edge_to_interior_edge(edge_hl_x=edge_hl_x,
        edge_hl_y=edge_hl_y, edge_hr_x=edge_hr_x, edge_hr_y=edge_hr_y, edge_vb_x=edge_vb_x, edge_vb_y=edge_vb_y, edge_vt_x=edge_vt_x, edge_vt_y=edge_vt_y)

    return [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ]

def get_line_distribution(object_mask=None, edges=None):

    #1.Obtain the edges of the object
    if edges is not None:
        [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = edges
    elif object_mask is not None:
        [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_exterior_edges(object_mask)
    else:
        raise Error

    edge_hl_x = edge_hl_x.cpu().numpy()
    edge_hl_y = edge_hl_y.cpu().numpy()
    edge_hr_x = edge_hr_x.cpu().numpy()
    edge_hr_y = edge_hr_y.cpu().numpy()
    edge_vb_x = edge_vb_x.cpu().numpy()
    edge_vb_y = edge_vb_y.cpu().numpy()
    edge_vt_x = edge_vt_x.cpu().numpy()
    edge_vt_y = edge_vt_y.cpu().numpy()


    #2.Sort each individual group of edges into clock-wise order
    #Sort Left
    edge_hl_x_sorted = []
    edge_hl_y_sorted = []

    for i, idx in enumerate(np.unique(edge_hl_x)):
        edge_hl_x_sorted = [*edge_hl_x[np.where(edge_hl_x == idx)][:int(np.array(np.where(edge_hl_x == idx)).size/2)],
                            *edge_hl_x_sorted,
                            *edge_hl_x[np.where(edge_hl_x == idx)][int(np.array(np.where(edge_hl_x == idx)).size/2):] ]

        edge_hl_y_sorted = [*edge_hl_y[np.where(edge_hl_x == idx)][:int(np.array(np.where(edge_hl_x == idx)).size/2)],
                            *edge_hl_y_sorted,
                            *edge_hl_y[np.where(edge_hl_x == idx)][int(np.array(np.where(edge_hl_x == idx)).size/2):] ]

    #Sort Top (already in order)
    edge_vt_x_sorted = edge_vt_x
    edge_vt_y_sorted = edge_vt_y

    #Sort Right
    edge_hr_x_sorted = []
    edge_hr_y_sorted = []
    for i, idx in enumerate(np.unique(edge_hr_x)[::-1]):
        edge_hr_x_sorted = [*edge_hr_x[np.where(edge_hr_x == idx)][:int(np.array(np.where(edge_hr_x == idx)).size/2)],
                            *edge_hr_x_sorted,
                            *edge_hr_x[np.where(edge_hr_x == idx)][int(np.array(np.where(edge_hr_x == idx)).size/2):] ]

        edge_hr_y_sorted = [*edge_hr_y[np.where(edge_hr_x == idx)][:int(np.array(np.where(edge_hr_x == idx)).size/2)],
                            *edge_hr_y_sorted,
                            *edge_hr_y[np.where(edge_hr_x == idx)][int(np.array(np.where(edge_hr_x == idx)).size/2):] ]

    edge_hr_y_sorted = edge_hr_y_sorted[::-1]

    #Sort Bottom
    edge_vb_x_sorted = edge_vb_x[::-1]
    edge_vb_y_sorted = edge_vb_y

    #3.Combine all sorted and Set Origin
    edge_x_sorted_full = [*edge_hl_x_sorted[int(len(edge_hl_x_sorted)/2):], *edge_vt_x_sorted, *edge_hr_x_sorted, *edge_vb_x_sorted, *edge_hl_x_sorted[:int(len(edge_hl_x_sorted)/2)]]
    edge_y_sorted_full = [*edge_hl_y_sorted[int(len(edge_hl_y_sorted)/2):], *edge_vt_y_sorted, *edge_hr_y_sorted, *edge_vb_y_sorted, *edge_hl_y_sorted[:int(len(edge_hl_y_sorted)/2)]]

    #4.Eliminate copied items
    edge_x_sorted = []
    edge_y_sorted = []

    for x, idx in enumerate(edge_x_sorted_full):
        jdx = edge_y_sorted_full[x]

        present = False
        for i in range(len(edge_x_sorted)): #if 1 element, len error
            if edge_x_sorted[i] == idx and edge_y_sorted[i] == jdx:
                present = True
                break

        if not present:
            edge_x_sorted.append(idx)
            edge_y_sorted.append(jdx)

    #5.Calculate corresponding angle
    angle = np.zeros_like(edge_x_sorted)
    for x, idx in enumerate(edge_x_sorted):
        jdx = edge_y_sorted[x]

        c_x = min(edge_x_sorted) + (max(edge_x_sorted)-min(edge_x_sorted))/2
        c_y = min(edge_y_sorted) + (max(edge_y_sorted)-min(edge_y_sorted))/2

        angle[x] = np.arctan2(jdx-c_y,c_x-idx)*(180/np.pi)
        if angle[x]<0:
            angle[x] = angle[x] + 360
    angle_sorted = sorted(angle)

    #6.Final sorting to correct order
    line_x_sorted = []
    line_y_sorted = []
    for i, ang in enumerate(angle_sorted):
        for j, jdx in enumerate(angle):
            if ang == jdx:
                line_x_sorted.append(edge_x_sorted[j])
                line_y_sorted.append(edge_y_sorted[j])
                break

    return line_x_sorted, line_y_sorted, angle_sorted

def rotate_mask():
    pass


#OBJECT CREATION FUNCTIONS
def get_blades_mask(domain, center, radius, num_blades):
    obstacle_bar = Box[45:55, 20:80]
    obstacle_bar_2 = Box[20:80, 45:55]
    obstacle = Obstacle(union([obstacle_bar, obstacle_bar_2]) ,angular_velocity=0)

    obstacle_mask = domain.scalar_grid(obstacle.geometry)

    return obstacle_mask


#BOUNDARIES CONDITIONS FUNCTIONS
def set_normal_bc(object_mask, velocity = None, velocity_BC = [0,0,0,0]):
    '''Function to impose the velocity boundary conditions in the normal direction of a given object.

    USAGE:
        -object_mask:  type scalar_grid//centerredgrid
        -velocity
        -velocity_BC: [velocity_left, velocity_right, velocity_bottom, velocity_up]
        -domain: simulation domain

    '''

    #1.Get the edges of the object
    [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_interior_edges(object_mask)

    #2.Pass to numpy the velocity field (in case none is passed, suppose it is not defined yet.)
    if velocity is None:
        u = np.zeros_like(object_mask+1)
        v = np.zeros_like(object_mask+1)
    else:
        [u, v] = to_numpy(velocity)

    #3.Set the normal velocities
    u[:, edge_hl_x, edge_hl_y] = velocity_BC[0]
    u[:, edge_hr_x +1, edge_hr_y] = velocity_BC[1]


    v[:, edge_vb_x, edge_vb_y] = velocity_BC[2]
    v[:, edge_vt_x, edge_vt_y + 1] = velocity_BC[3]


    #4.Pass back to PhiFlow Staggered
    _,Lx,Ly,_,_,_,_ = get_dimensions(object_mask)
    velocity = to_staggered([u,v], Lx, Ly)

    return velocity

def set_tangential_bc(object_mask, velocity = None, velocity_BC = [0,0,0,0], inv_geom = False):
    '''Function to impose the velocity boundary conditions in the tangential direction of a given object.

    USAGE:
        -object_mask:  type scalar_grid//centerredgrid
        -velocity
        -velocity_BC: [velocity_left, velocity_right, velocity_bottom, velocity_up]
        -domain: simulation domain

    '''

    #1.Get the edges of the object
    [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_interior_edges(object_mask)

    #2.Pass to numpy the velocity field (in case none is passed, suppose it is not defined yet.)
    if velocity is None:
        u = np.zeros_like(object_mask+1)
        v = np.zeros_like(object_mask+1)
    else:
        [u, v] = to_numpy(velocity)

    # If bsz =1, squeeze batch dimension
    if u.shape[0] == 1:
        u = u[0]
        v = v[0]


    #3.Set the normal velocities
    if not inv_geom:
        v[edge_hl_x - 1, edge_hl_y] = velocity_BC[0]
        v[edge_hr_x + 1, edge_hr_y] = velocity_BC[1]

        u[edge_vb_x, edge_vb_y - 1] = velocity_BC[2]
        u[edge_vt_x, edge_vt_y + 1] = velocity_BC[3]
    elif inv_geom:
        v[edge_hl_x, edge_hl_y] = velocity_BC[0]
        v[edge_hr_x, edge_hr_y] = velocity_BC[1]

        u[edge_vb_x, edge_vb_y] = velocity_BC[2]
        u[edge_vt_x, edge_vt_y] = velocity_BC[3]

    #4.Pass back to PhiFlow Staggered
    _,Lx,Ly,_,_,_,_ = get_dimensions(object_mask)
    velocity = to_staggered([u,v], Lx, Ly)

    return velocity

def set_tangential_w_bc(object_mask, velocity = None, w_BC = 0):
    '''Function to impose the velocity boundary conditions in the tangential direction of a given object.

    USAGE:
        -object_mask:  type scalar_grid//centerredgrid
        -velocity
        -w_BC:   ! w positive in clock-wise direction
        -domain: simulation domain

    '''

    exit()
    pass

    #1.Get the edges of the object
    [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_interior_edges(object_mask)

    #2.Pass to numpy the velocity field (in case none is passed, suppose it is not defined yet.)
    if velocity is None:
        u = np.zeros_like(object_mask+1)
        v = np.zeros_like(object_mask+1)
    else:
        [u, v] = to_numpy(velocity)

    # If bsz =1, squeeze batch dimension
    if u.shape[0] == 1:
        u = u[0]
        v = v[0]

    #3.Set the normal velocities
    v[edge_hl_x - 1, edge_hl_y] = w_BC*r
    v[edge_hr_x + 1, edge_hr_y] = -w_BC*r

    u[edge_vb_x, edge_vb_y - 1] = -w_BC*r
    u[edge_vt_x, edge_vt_y + 1] = w_BC*r

    #4.Pass back to PhiFlow Staggered
    _,Lx,Ly,_,_,_,_ = get_dimensions(object_mask)
    velocity = to_staggered([u,v], Lx, Ly)

    return velocity

def set_interior_bc(object_mask, velocity = None, velocity_BC = [0,0], inv_geom = False):
    '''Function to impose the velocity boundary conditions inside of a given object.

    USAGE:
        -object_mask:  type scalar_grid//centerredgrid
        -velocity
        -velocity_BC: [velocity_x, velocity_y]
        -domain: simulation domain

    '''

    #1.Get the edges of the object
    [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_interior_edges(object_mask)

    #2.Pass to numpy the velocity field (in case none is passed, suppose it is not defined yet.)
    if velocity is None:
        u = np.zeros_like(object_mask+1)
        v = np.zeros_like(object_mask+1)
    else:
        [u, v] = to_numpy(velocity)

    # If bsz =1, squeeze batch dimension
    if u.shape[0] == 1:
        u = u[0]
        v = v[0]

    #4.Set the normal velocities
    if not inv_geom:
        line_x_sorted, line_y_sorted, _ = get_line_distribution(edges=[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ])

        up_x = line_x_sorted[:int(len(line_x_sorted)/2)]
        up_y = line_y_sorted[:int(len(line_y_sorted)/2)]

        #down_x = line_x_sorted[int(len(line_x_sorted)/2):]
        down_y = line_y_sorted[int(len(line_y_sorted)/2):]

        for i in range(len(up_x)):
            u[up_x[i], down_y[i]:up_y[i]+1] = velocity_BC[0]
            v[up_x[i], down_y[i]:up_y[i]+1] = velocity_BC[1]

    elif inv_geom:
        u_aux = np.ones_like(u) * velocity_BC[0]
        v_aux = np.ones_like(v) * velocity_BC[1]

        line_x_sorted, line_y_sorted, _ = get_line_distribution(edges=[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ])

        up_x = line_x_sorted[:int(len(line_x_sorted)/2)]
        up_y = line_y_sorted[:int(len(line_y_sorted)/2)]

        #down_x = line_x_sorted[int(len(line_x_sorted)/2):]
        down_y = line_y_sorted[int(len(line_y_sorted)/2):]

        for i in range(len(up_x)):
            u_aux[up_x[i], down_y[i]:up_y[i]+1] = u[up_x[i], down_y[i]:up_y[i]+1]
            v_aux[up_x[i], down_y[i]:up_y[i]+1] = v[up_x[i], down_y[i]:up_y[i]+1]

        u = u_aux
        v = v_aux

    #5.Pass back to PhiFlow Staggered
    _,Lx,Ly,_,_,_,_ = get_dimensions(object_mask)
    velocity = to_staggered([u,v], Lx, Ly)

    return velocity

def set_wall_bc(object_mask, velocity = None):
    return set_normal_bc(object_mask, velocity = velocity)

def set_flow_bc():
    pass

def set_rotate_bc(object_mask, velocity = None, w=0, wr=0):
    velocity = set_interior_bc(object_mask, velocity = velocity)
    velocity = set_wall_bc(object_mask, velocity = velocity)

    #Ahra mismo, es w*r !!! si se usa la set_w entonces w
    velocity = set_tangential_bc(object_mask, velocity = velocity, velocity_BC = [wr,-wr,-wr,wr])

    return velocity


def set_normal_bc2(bc_mask_x, bc_mask_y, bc_value_x, bc_value_y, object_mask, velocity_BC = [0,0,0,0]):
    '''Function to impose the velocity boundary conditions in the normal direction of a given object.

    '''

    [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_interior_edges(object_mask)

    #Set the normal velocities
    bc_value_x[edge_hl_x, edge_hl_y] += velocity_BC[0]
    bc_value_x[edge_hr_x +1, edge_hr_y] += velocity_BC[1]

    bc_value_y[edge_vb_x, edge_vb_y] += velocity_BC[2]
    bc_value_y[edge_vt_x, edge_vt_y + 1] += velocity_BC[3]

    bc_mask_x[edge_hl_x, edge_hl_y] = 0
    bc_mask_x[edge_hr_x +1, edge_hr_y] = 0

    bc_mask_y[edge_vb_x, edge_vb_y] = 0
    bc_mask_y[edge_vt_x, edge_vt_y + 1] = 0

    return bc_mask_x, bc_mask_y, bc_value_x, bc_value_y

def set_tangential_bc2(bc_mask_x, bc_mask_y, bc_value_x, bc_value_y, object_mask, velocity_BC = [0,0,0,0], inv_geom = False):
    '''Function to impose the velocity boundary conditions in the tangential direction of a given object.

    '''

    #1.Get the edges of the object
    [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_interior_edges(object_mask)

    #Set the normal velocities
    if not inv_geom:
        bc_value_y[edge_hl_x - 1, edge_hl_y] += velocity_BC[0]
        bc_value_y[edge_hr_x + 1, edge_hr_y] += velocity_BC[1]

        bc_value_x[edge_vb_x, edge_vb_y - 1] += velocity_BC[2]
        bc_value_x[edge_vt_x, edge_vt_y + 1] += velocity_BC[3]

        bc_mask_y[edge_hl_x - 1, edge_hl_y] = 0
        bc_mask_y[edge_hr_x + 1, edge_hr_y] = 0

        bc_mask_x[edge_vb_x, edge_vb_y - 1] = 0
        bc_mask_x[edge_vt_x, edge_vt_y + 1] = 0

    elif inv_geom:
        bc_value_y[edge_hl_x, edge_hl_y] += velocity_BC[0]
        bc_value_y[edge_hr_x, edge_hr_y] += velocity_BC[1]

        bc_value_x[edge_vb_x, edge_vb_y] += velocity_BC[2]
        bc_value_x[edge_vt_x, edge_vt_y] += velocity_BC[3]

        bc_mask_y[edge_hl_x, edge_hl_y] = 0
        bc_mask_y[edge_hr_x, edge_hr_y] = 0

        bc_mask_x[edge_vb_x, edge_vb_y] = 0
        bc_mask_x[edge_vt_x, edge_vt_y] = 0


    return bc_mask_x, bc_mask_y, bc_value_x, bc_value_y,

def set_interior_bc2(bc_mask_x, bc_mask_y, bc_value_x, bc_value_y, object_mask, velocity_BC = [0,0], inv_geom = False):
    '''Function to impose the velocity boundary conditions inside of a given object.
    '''

    #1.Get the edges of the object
    [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_interior_edges(object_mask)

    #2.Set the normal velocities
    if not inv_geom:
        line_x_sorted, line_y_sorted, _ = get_line_distribution(edges=[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ])

        up_x = line_x_sorted[:int(len(line_x_sorted)/2)]
        up_y = line_y_sorted[:int(len(line_y_sorted)/2)]

        #down_x = line_x_sorted[int(len(line_x_sorted)/2):]
        down_y = line_y_sorted[int(len(line_y_sorted)/2):]

        for i in range(len(up_x)):
            bc_value_x[up_x[i], down_y[i]:up_y[i]+1] += velocity_BC[0]
            bc_value_y[up_x[i], down_y[i]:up_y[i]+1] += velocity_BC[1]

            bc_mask_x[up_x[i], down_y[i]:up_y[i]+1] = 0
            bc_mask_y[up_x[i], down_y[i]:up_y[i]+1] = 0

    elif inv_geom:
        bc_value_x_aux  = torch.ones_like(bc_mask_y) * velocity_BC[0]
        bc_value_y_aux = torch.ones_like(bc_mask_y) * velocity_BC[1]
        bc_mask_x_aux = torch.zeros_like(bc_mask_y)
        bc_mask_y_aux = torch.zeros_like(bc_mask_y)

        line_x_sorted, line_y_sorted, _ = get_line_distribution(edges=[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ])

        up_x = line_x_sorted[:int(len(line_x_sorted)/2)]
        up_y = line_y_sorted[:int(len(line_y_sorted)/2)]

        #down_x = line_x_sorted[int(len(line_x_sorted)/2):]
        down_y = line_y_sorted[int(len(line_y_sorted)/2):]

        for i in range(len(up_x)):
            bc_value_x_aux[up_x[i], down_y[i]:up_y[i]+1] = bc_value_x[up_x[i], down_y[i]:up_y[i]+1]
            bc_value_y_aux[up_x[i], down_y[i]:up_y[i]+1] = bc_value_y[up_x[i], down_y[i]:up_y[i]+1]

            bc_mask_x_aux[up_x[i], down_y[i]:up_y[i]+1] = bc_mask_x[up_x[i], down_y[i]:up_y[i]+1]
            bc_mask_y_aux[up_x[i], down_y[i]:up_y[i]+1] = bc_mask_y[up_x[i], down_y[i]:up_y[i]+1]

        bc_value_x = bc_value_x_aux
        bc_value_y = bc_value_y_aux

        bc_mask_x = bc_mask_x_aux
        bc_mask_y = bc_mask_y_aux

    return bc_mask_x, bc_mask_y, bc_value_x, bc_value_y


def get_obstacles_bc(obstacles):
    '''obstacles: [ [obstacle_mask, 'wr', 'inv_geom'], [obstacle_mask, 'wr', 'inv_geom'], ...]
    '''
    Z = to_numpy(obstacles[0][0])

    # If Bsz not esual to 1!
    if len(Z.shape) == 3 and Z.shape[0] > 1:
        Z = (Z.shape[0], Z.shape[1]+1, Z.shape[2]+1)

        bc_mask_x = torch.ones(Z)
        bc_mask_y = torch.ones(Z)
        bc_value_x = torch.zeros(Z)
        bc_value_y = torch.zeros(Z)

        for object_mask, wr, inverse in obstacles:
            for i in range(Z[0]):
                print("I {}".format(i))
                bc_mask_x[i], bc_mask_y[i], bc_value_x[i], bc_value_y[i] = set_normal_bc2(bc_mask_x[i], bc_mask_y[i], bc_value_x[i], bc_value_y[i], object_mask[i])
                bc_mask_x[i], bc_mask_y[i], bc_value_x[i], bc_value_y[i] = set_interior_bc2(bc_mask_x[i], bc_mask_y[i], bc_value_x[i], bc_value_y[i], object_mask[i], inv_geom=inverse)
                if wr is not None:
                    bc_mask_x[i], bc_mask_y[i], bc_value_x[i], bc_value_y[i] = set_tangential_bc2(bc_mask_x[i], bc_mask_y[i], bc_value_x[i], bc_value_y[i], object_mask[i], velocity_BC=[wr,-wr,-wr,wr], inv_geom=inverse)

        #Pass back to PhiFlow Staggered
        _,Lx,Ly,_,_,_,_ = get_dimensions(obstacles[0][0])
        bc_value = to_staggered([bc_value_x,bc_value_y], Lx, Ly)
        bc_mask = to_staggered([bc_mask_x,bc_mask_y], Lx, Ly)

    else:
        Z = (Z.shape[0]+1, Z.shape[1]+1)

        bc_mask_x = torch.ones(Z)
        bc_mask_y = torch.ones(Z)
        bc_value_x = torch.zeros(Z)
        bc_value_y = torch.zeros(Z)

        for object_mask, wr, inverse in obstacles:
            bc_mask_x, bc_mask_y, bc_value_x, bc_value_y = set_normal_bc2(bc_mask_x, bc_mask_y, bc_value_x, bc_value_y, object_mask)
            bc_mask_x, bc_mask_y, bc_value_x, bc_value_y = set_interior_bc2(bc_mask_x, bc_mask_y, bc_value_x, bc_value_y, object_mask, inv_geom=inverse)
            if wr is not None:
                bc_mask_x, bc_mask_y, bc_value_x, bc_value_y = set_tangential_bc2(bc_mask_x, bc_mask_y, bc_value_x, bc_value_y, object_mask, velocity_BC=[-wr,wr,wr,-wr], inv_geom=inverse)

            #Pass back to PhiFlow Staggered
            _,Lx,Ly,_,_,_,_ = get_dimensions(obstacles[0][0])

            value = torch.cat((bc_value_x[None,:,:], bc_value_y[None,:,:]), dim=0)
            mask = torch.cat((bc_mask_x[None,:,:], bc_mask_y[None,:,:]), dim=0)

            bc_value = to_staggered(value, Lx, Ly)
            bc_mask = to_staggered(mask, Lx, Ly)

    return bc_mask, bc_value

def get_flows_bc():
    pass #inside flows

def get_domain_bc_masks():
    pass #inflows, etc en la pared

def apply_boundaries(velocity, bc_mask, bc_value):
    return velocity*bc_mask + bc_value


#FRAME OF REFERENCE FUNCTIONS
def to_relative():
    pass

def to_inertial():
    pass