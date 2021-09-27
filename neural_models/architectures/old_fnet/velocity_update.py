
import torch

from . import CellType
from . import velocityDivergence
import matplotlib.pyplot as plt

def velocityUpdateNot(pressure, U, flags):

    r""" Calculate the pressure gradient and subtract it into (i.e. calculate
    U' = U - grad(p)). Some care must be taken with handling boundary conditions.
    This function mimics correctVelocity in Manta.
    Velocity update is done IN-PLACE.

    Arguments:
        p (Tensor): scalar pressure field.
        U (Tensor): velocity field (size(2) can be 2 or 3, indicating 2D / 3D)
        flags (Tensor): input occupancy grid
    """
    # Check arguments.
    assert U.dim() == 5 and flags.dim() == 5 and pressure.dim() == 5, \
               "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"
    b = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (U.size(1) == 3)
    if not is3D:
        assert d == 1, "d > 1 for a 2D domain"
        assert U.size(4) == w, "2D velocity field must have only 2 channels"

    assert U.size(0) == b and U.size(2) == d and U.size(3) == h \
        and U.size(4) == w, "size mismatch"
    assert pressure.is_same_size(flags), "size mismatch"
    assert U.is_contiguous() and flags.is_contiguous() and \
              pressure.is_contiguous(), "Input is not contiguous"

    # First, we build the mask for detecting fluid cells. Borders are left untouched.
    # mask_fluid   Fluid cells.
    # mask_fluid_i Fluid cells with (i-1) neighbour also a fluid.
    # mask_fluid_j Fluid cells with (j-1) neighbour also a fluid.
    # mask_fluid_k FLuid cells with (k-1) neighbour also a fluid.

    # Second, we detect obstacle cells
    # See Bridson p44 for algorithm and boundaries treatment.

    if not is3D:
        # Current cell is fluid
        mask_fluid = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid)
        # Current is fluid and neighbour to left or down are fluid
        mask_fluid_i = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid))
        mask_fluid_j = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeFluid))
        # Current cell is fluid and neighbours to left or down are obstacle
        mask_fluid_obstacle_im1 = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeEmpty))
        mask_fluid_obstacle_jm1 = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeEmpty))
        # Current cell is obstacle and not outflow
        mask_obstacle = flags.narrow(4, 1, w-2).narrow(3, 1, h-2) \
                            .eq(CellType.TypeEmpty).__and__ \
                     (flags.narrow(4, 1, w-2).narrow(3, 1, h-2) \
                            .ne(CellType.TypeOutflow))
        # Current cell is obstacle and neighbours to left or down are fluid
        mask_obstacle_fluid_im1 = mask_obstacle.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid))
        mask_obstacle_fluid_jm1 = mask_obstacle.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeFluid))
        # Current cell is obstacle and neighbours to left or down are not fluid
        mask_no_fluid_im1 = mask_obstacle.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeEmpty))
        mask_no_fluid_jm1 = mask_obstacle.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeEmpty))

    else:
        # TODO: implement 3D bcs well.
        # TODO: add outlfow (change in advection required)
        mask_fluid  = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2).eq(CellType.TypeFluid)
        mask_fluid_i = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2).eq(CellType.TypeFluid))
        mask_fluid_j = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 0, h-1).narrow(2, 1, d-2).eq(CellType.TypeFluid))
        mask_fluid_k = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 0, d-2).eq(CellType.TypeFluid))

    # Cast into float or double tensor and cat into a single mask along chan.

    mask_fluid_i_f = mask_fluid_i.type(U.type())
    mask_fluid_j_f = mask_fluid_j.type(U.type())

    mask_fluid_obstacle_i_f = mask_fluid_obstacle_im1.type(U.type())
    mask_fluid_obstacle_j_f = mask_fluid_obstacle_jm1.type(U.type())

    mask_obstacle_fluid_i_f = mask_obstacle_fluid_im1.type(U.type())
    mask_obstacle_fluid_j_f = mask_obstacle_fluid_jm1.type(U.type())

    mask_no_fluid_i_f = mask_no_fluid_im1.type(U.type())
    mask_no_fluid_j_f = mask_no_fluid_jm1.type(U.type())

    if is3D:
        mask_fluid_k_f = mask_fluid_k.type(U.type())

    if not is3D:
        mask_fluid = torch.cat((mask_fluid_i_f, mask_fluid_j_f), 1).contiguous()
        mask_fluid_obstacle = torch.cat((mask_fluid_obstacle_i_f, mask_fluid_obstacle_j_f), 1).contiguous()
        mask_obstacle_fluid = torch.cat((mask_obstacle_fluid_i_f, mask_obstacle_fluid_j_f), 1).contiguous()
        mask_no_fluid = torch.cat((mask_no_fluid_i_f, mask_no_fluid_j_f), 1).contiguous()
    else:
        mask_fluid = torch.cat((mask_fluid_i_f, mask_fluid_j_f, mask_fluid_k_f), 1).contiguous()

    # pressure tensor.
    #Pijk    Pressure at (i,j,k) in 3 channels (2 for 2D).
    #Pijk_m  Pressure at chan 0: (i-1, j, k)
              #          chan 1: (i, j-1, k)
              #          chan 2: (i, j, k-1)

    if not is3D:
        Pijk = pressure.narrow(4, 1, w-2).narrow(3, 1, h-2)
        Pijk = Pijk.clone().expand(b, 2, d, h-2, w-2)
        Pijk_m = Pijk.clone().expand(b, 2, d, h-2, w-2)
        Pijk_m[:,0] = pressure.narrow(4, 0, w-2).narrow(3, 1, h-2).squeeze(1)
        Pijk_m[:,1] = pressure.narrow(4, 1, w-2).narrow(3, 0, h-2).squeeze(1)
    else:
        Pijk = pressure.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2)
        Pijk = Pijk.clone().expand(b, 3, d-2, h-1, w-1)
        Pijk_m = Pijk.clone().expand(b, 3, d-2, h-1, w-1)
        Pijk_m[:,0] = pressure.narrow(4, 0, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2).squeeze(1)
        Pijk_m[:,1] = pressure.narrow(4, 1, w-1).narrow(3, 0, h-1).narrow(2, 1, d-2).squeeze(1)
        Pijk_m[:,2] = pressure.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 0, d-2).squeeze(1)

    # grad(p) = [[ p(i,j,k) - p(i-1,j,k) ]
    #            [ p(i,j,k) - p(i,j-1,k) ]
    #            [ p(i,j,k) - p(i,j,k-1) ]]
    if not is3D:
        # Three cases:
        # 1) Cell is fluid and left neighbour is fluid:
        # u = u - grad(p)
        # 2) Cell is fluid and left neighbour is obstacle
        # u = u - p(i,j)
        # 3) Cell is obstacle and left neighbour is fluid
        # u = u + p(i-1,j)

        U[:,:,:,1:(h-1),1:(w-1)] = (mask_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
             mask_fluid_obstacle * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - Pijk) + \
            mask_obstacle_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) + Pijk_m) + \
            mask_no_fluid * (0))
        #print('******************************************************')

        #print('masks')
        #print('*    *         *            *')
        #print(mask_fluid)
        #print('*    *         *            *')
        #print(mask_fluid_obstacle)
        #print(mask_obstacle_fluid)
        #print(mask_no_fluid)
        #print('******************************************************')
    else:
        U[:,:,1:(d-1),1:(h-1),1:(w-1)] =  mask * \
            (U.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2) - (Pijk - Pijk_m))


def velocityUpdate(pressure, U, flags):

    r""" Calculate the pressure gradient and subtract it into (i.e. calculate
    U' = U - grad(p)). Some care must be taken with handling boundary conditions.
    This function mimics correctVelocity in Manta.
    Velocity update is done IN-PLACE.
    Arguments:
        p (Tensor): scalar pressure field.
        U (Tensor): velocity field (size(2) can be 2 or 3, indicating 2D / 3D)
        flags (Tensor): input occupancy grid
    """
    # Check arguments.
    assert U.dim() == 5 and flags.dim() == 5 and pressure.dim() == 5, \
               "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"
    b = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (U.size(1) == 3)
    if not is3D:
        assert d == 1, "d > 1 for a 2D domain"
        assert U.size(4) == w, "2D velocity field must have only 2 channels"

    assert U.size(0) == b and U.size(2) == d and U.size(3) == h \
        and U.size(4) == w, "size mismatch"
    assert pressure.is_same_size(flags), "size mismatch"
    assert U.is_contiguous(), "U is not contiguous"
    assert flags.is_contiguous(), "Flags is not contiguous"
    assert pressure.is_contiguous(), "Pressure is not contiguous"


    # First, we build the mask for detecting fluid cells. Borders are left untouched.
    # mask_fluid   Fluid cells.
    # mask_fluid_i Fluid cells with (i-1) neighbour also a fluid.
    # mask_fluid_j Fluid cells with (j-1) neighbour also a fluid.
    # mask_fluid_k FLuid cells with (k-1) neighbour also a fluid.

    # Second, we detect obstacle cells
    # See Bridson p44 for algorithm and boundaries treatment.

    if not is3D:
        # Current cell is fluid
        mask_fluid = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid)

        # Current cell is inflow
        mask_inflow = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).eq(CellType.TypeInflow)

        # Current cell is outflow
        mask_outflow = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).eq(CellType.TypeOutflow)

        # Current is fluid and neighbour to left or down are fluid
        mask_fluid_i = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid))
        mask_fluid_j = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeFluid))

        # Current is inflow and neighbour to left or down are inflow
        mask_inflow_i = mask_inflow.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeInflow))
        mask_inflow_j = mask_inflow.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeInflow))

        # Current is inflow and neighbour to left or down are inflow
        mask_outflow_i = mask_outflow.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeOutflow))
        mask_outflow_j = mask_outflow.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeOutflow))

        # Current cell is fluid and neighbours to left or down are obstacle
        mask_fluid_obstacle_im1 = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeEmpty))
        mask_fluid_obstacle_jm1 = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeEmpty))

        # Current cell is inflow and neighbours to left or down are obstacle
        mask_inflow_obstacle_im1 = mask_inflow.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeEmpty))
        mask_inflow_obstacle_jm1 = mask_inflow.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeEmpty))

        # Current cell is fluid and neighbours to left or down are Inflows
        mask_fluid_inflow_im1 = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeInflow))
        mask_fluid_inflow_jm1 = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeInflow))

        # Current cell is fluid and neighbours to left or down are Outflows
        mask_fluid_outflow_im1 = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeOutflow))
        mask_fluid_outflow_jm1 = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeOutflow))


        # Current cell is obstacle and not outflow
        mask_obstacle = flags.narrow(4, 1, w-2).narrow(3, 1, h-2) \
                            .eq(CellType.TypeEmpty).__and__ \
                     (flags.narrow(4, 1, w-2).narrow(3, 1, h-2) \
                            .ne(CellType.TypeOutflow))

        # Current cell is obstacle and neighbours to left or down are fluid
        mask_obstacle_fluid_im1 = mask_obstacle.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid))
        mask_obstacle_fluid_jm1 = mask_obstacle.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeFluid))

        # Current cell is obstacle and neighbours to left or down are inflow
        mask_obstacle_inflow_im1 = mask_obstacle.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeInflow))
        mask_obstacle_inflow_jm1 = mask_obstacle.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeInflow))

        # Current cell is inflow and neighbours to left or down are fluid
        mask_inflow_fluid_im1 = mask_inflow.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid))
        mask_inflow_fluid_jm1 = mask_inflow.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeFluid))

        # Current cell is outflow and neighbours to left or down are fluid
        mask_outflow_fluid_im1 = mask_outflow.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid))
        mask_outflow_fluid_jm1 = mask_outflow.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeFluid))


        # Current cell is outflow and neighbours to left or down are obstacle
        mask_outflow_obstacle_im1 = mask_outflow.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeEmpty))
        mask_outflow_obstacle_jm1 = mask_outflow.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeEmpty))

        # Current cell is obstacle and neighbours to left or down are not fluid
        mask_no_fluid_im1 = mask_obstacle.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeEmpty))
        mask_no_fluid_jm1 = mask_obstacle.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeEmpty))

        # Current cell is outflow and neighbours to left or down are not fluid
        mask_outflow_no_fluid_im1 = mask_outflow.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).ne(CellType.TypeFluid))
        mask_outflow_no_fluid_jm1 = mask_outflow.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).ne(CellType.TypeFluid))

    else:
        # TODO: implement 3D bcs well.
        # TODO: add outlfow (change in advection required)
        mask_fluid  = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2).eq(CellType.TypeFluid)
        mask_fluid_i = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2).eq(CellType.TypeFluid))
        mask_fluid_j = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 0, h-1).narrow(2, 1, d-2).eq(CellType.TypeFluid))
        mask_fluid_k = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 0, d-2).eq(CellType.TypeFluid))

    # Cast into float or double tensor and cat into a single mask along chan.
    mask_fluid_i_f = mask_fluid_i.type(U.type())
    mask_fluid_j_f = mask_fluid_j.type(U.type())

    mask_inflow_i_f = mask_inflow_i.type(U.type())
    mask_inflow_j_f = mask_inflow_j.type(U.type())

    mask_outflow_i_f = mask_outflow_i.type(U.type())
    mask_outflow_j_f = mask_outflow_j.type(U.type())

    mask_fluid_obstacle_i_f = mask_fluid_obstacle_im1.type(U.type())
    mask_fluid_obstacle_j_f = mask_fluid_obstacle_jm1.type(U.type())

    mask_fluid_inflow_im1 = mask_fluid_inflow_im1.type(U.type())
    mask_fluid_inflow_jm1 = mask_fluid_inflow_jm1.type(U.type())

    mask_fluid_outflow_im1 = mask_fluid_outflow_im1.type(U.type())
    mask_fluid_outflow_jm1 = mask_fluid_outflow_jm1.type(U.type())

    mask_inflow_fluid_im1 = mask_inflow_fluid_im1.type(U.type())
    mask_inflow_fluid_jm1 = mask_inflow_fluid_jm1.type(U.type())

    mask_inflow_obstacle_im1 = mask_inflow_obstacle_im1.type(U.type())
    mask_inflow_obstacle_jm1 = mask_inflow_obstacle_jm1.type(U.type())

    mask_outflow_fluid_im1 = mask_outflow_fluid_im1.type(U.type())
    mask_outflow_fluid_jm1 = mask_outflow_fluid_jm1.type(U.type())

    mask_obstacle_fluid_i_f = mask_obstacle_fluid_im1.type(U.type())
    mask_obstacle_fluid_j_f = mask_obstacle_fluid_jm1.type(U.type())

    mask_obstacle_inflow_i_f = mask_obstacle_inflow_im1.type(U.type())
    mask_obstacle_inflow_j_f = mask_obstacle_inflow_jm1.type(U.type())

    mask_no_fluid_i_f = mask_no_fluid_im1.type(U.type())
    mask_no_fluid_j_f = mask_no_fluid_jm1.type(U.type())


 
    if is3D:
        mask_fluid_k_f = mask_fluid_k.type(U.type())

    if not is3D:
        mask_fluid = torch.cat((mask_fluid_i_f, mask_fluid_j_f), 1).contiguous()
        mask_inflow = torch.cat((mask_inflow_i_f, mask_inflow_j_f), 1).contiguous()
        mask_outflow = torch.cat((mask_outflow_i_f, mask_outflow_j_f), 1).contiguous()
        mask_fluid_obstacle = torch.cat((mask_fluid_obstacle_i_f, mask_fluid_obstacle_j_f), 1).contiguous()
        mask_fluid_inflow = torch.cat((mask_fluid_inflow_im1, mask_fluid_inflow_jm1), 1).contiguous()
        mask_fluid_outflow = torch.cat((mask_fluid_outflow_im1, mask_fluid_outflow_jm1), 1).contiguous()
        mask_obstacle_fluid = torch.cat((mask_obstacle_fluid_i_f, mask_obstacle_fluid_j_f), 1).contiguous()
        mask_obstacle_inflow = torch.cat((mask_obstacle_inflow_i_f, mask_obstacle_inflow_j_f), 1).contiguous()        
        mask_inflow_fluid = torch.cat((mask_inflow_fluid_im1, mask_inflow_fluid_jm1), 1).contiguous()
        mask_outflow_fluid = torch.cat((mask_outflow_fluid_im1, mask_outflow_fluid_jm1), 1).contiguous()
        mask_inflow_obstacle = torch.cat((mask_inflow_obstacle_im1, mask_inflow_obstacle_jm1), 1).contiguous()
        mask_no_fluid = torch.cat((mask_no_fluid_i_f, mask_no_fluid_j_f), 1).contiguous()
    else:
        mask_fluid = torch.cat((mask_fluid_i_f, mask_fluid_j_f, mask_fluid_k_f), 1).contiguous()

    # pressure tensor.
    #Pijk    Pressure at (i,j,k) in 3 channels (2 for 2D).
    #Pijk_m  Pressure at chan 0: (i-1, j, k)
              #          chan 1: (i, j-1, k)
              #          chan 2: (i, j, k-1)
    #Pijk_p  Pressure at chan 0: (i+1, j, k)
              #          chan 1: (i, j+1, k)
              #          chan 2: (i, j, k+1)


    if not is3D:
        Pijk = pressure.narrow(4, 1, w-2).narrow(3, 1, h-2)
        Pijk = Pijk.clone().expand(b, 2, d, h-2, w-2)
        Pijk_m = Pijk.clone().expand(b, 2, d, h-2, w-2)
        Pijk_p = Pijk.clone().expand(b, 2, d, h-2, w-2)
        Pijk_m[:,0] = pressure.narrow(4, 0, w-2).narrow(3, 1, h-2).squeeze(1)
        Pijk_m[:,1] = pressure.narrow(4, 1, w-2).narrow(3, 0, h-2).squeeze(1)
        Pijk_p[:,0] = pressure.narrow(4, 2, w-2).narrow(3, 1, h-2).squeeze(1)
        Pijk_p[:,1] = pressure.narrow(4, 1, w-2).narrow(3, 2, h-2).squeeze(1)

    else:
        Pijk = pressure.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2)
        Pijk = Pijk.clone().expand(b, 3, d-2, h-1, w-1)
        Pijk_m = Pijk.clone().expand(b, 3, d-2, h-1, w-1)
        Pijk_m[:,0] = pressure.narrow(4, 0, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2).squeeze(1)
        Pijk_m[:,1] = pressure.narrow(4, 1, w-1).narrow(3, 0, h-1).narrow(2, 1, d-2).squeeze(1)
        Pijk_m[:,2] = pressure.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 0, d-2).squeeze(1)

    # grad(p) = [[ p(i,j,k) - p(i-1,j,k) ]
    #            [ p(i,j,k) - p(i,j-1,k) ]
    #            [ p(i,j,k) - p(i,j,k-1) ]]
    if not is3D:
        # Three cases:
        # 1) Cell is fluid and left neighbour is fluid:
        # u = u - grad(p)
        # 2) Cell is fluid and left neighbour is obstacle
        # u = u - p(i,j)
        # 3) Cell is obstacle and left neighbour is fluid
        # u = u + p(i-1,j)


        """
        U[:,:,:,1:(h-1),1:(w-1)] = (mask_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
            mask_fluid_outflow * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - Pijk) + \
            mask_fluid_obstacle * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - Pijk) + \
            mask_obstacle_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) + Pijk_m) + \
            mask_outflow_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) + Pijk_m) + \
            mask_inflow * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) +            
            mask_fluid_inflow * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
            mask_inflow_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
            mask_obstacle_inflow * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
            mask_inflow_obstacle * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
            mask_no_fluid * (0))
        """

        #import matplotlib.pyplot as plt
        #fig, axs = plt.subplots(1, 2, figsize=(6,3))
        #im0 = axs[0].imshow((mask_fluid[0, 0]).detach().cpu()[0, 180:260, 120:180], cmap=plt.get_cmap('seismic'))
        #fig.colorbar(im0, ax=axs[0])
        #im1 = axs[1].imshow((mask_fluid[0, 1]).detach().cpu()[0, 180:260, 120:180], cmap=plt.get_cmap('seismic'))
        #fig.colorbar(im1, ax=axs[1])
        #fig.tight_layout()
        #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results_lt_nograd/mask_fluid.png')
        #fig, axs = plt.subplots(1, 2, figsize=(6,3))
        #im0 = axs[0].imshow((mask_fluid_obstacle[0, 0]).detach().cpu()[0, 180:260, 120:180], cmap=plt.get_cmap('seismic'))
        #fig.colorbar(im0, ax=axs[0])
        #im1 = axs[1].imshow((mask_fluid_obstacle[0, 1]).detach().cpu()[0, 180:260, 120:180], cmap=plt.get_cmap('seismic'))
        #fig.colorbar(im1, ax=axs[1])
        #fig.tight_layout()
        #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results_lt_nograd/mask_fluid_obstacle.png')
        #fig, axs = plt.subplots(1, 2, figsize=(6,3))
        #im0 = axs[0].imshow((mask_obstacle_fluid[0, 0]).detach().cpu()[0, 180:260, 120:180], cmap=plt.get_cmap('seismic'))
        #fig.colorbar(im0, ax=axs[0])
        #im1 = axs[1].imshow((mask_obstacle_fluid[0, 1]).detach().cpu()[0, 180:260, 120:180], cmap=plt.get_cmap('seismic'))
        #fig.colorbar(im1, ax=axs[1])
        #fig.tight_layout()
        #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results_lt_nograd/mask_obstacle_fluid.png')
        #fig, axs = plt.subplots(1, 2, figsize=(6,3))
        #im0 = axs[0].imshow((mask_no_fluid[0, 0]).detach().cpu()[0, 180:260, 120:180], cmap=plt.get_cmap('seismic'))
        #fig.colorbar(im0, ax=axs[0])
        #im1 = axs[1].imshow((mask_no_fluid[0, 1]).detach().cpu()[0, 180:260, 120:180], cmap=plt.get_cmap('seismic'))
        #fig.colorbar(im1, ax=axs[1])
        #fig.tight_layout()
        #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results_lt_nograd/mask_no_fluid.png')
        #fig, axs = plt.subplots(1, 3, figsize=(9,3))
        #maxgpx = torch.max((Pijk - Pijk_m)[0, 0])
        #maxgpy = torch.max((Pijk - Pijk_m)[0, 1])
        #maxp = torch.max((Pijk)[0, 0])
        #im0 = axs[0].imshow(((Pijk)[0, 0]).detach().cpu()[0, :, :], vmin = -maxp , vmax = maxp, cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180] [0, 438:, 502:]
        #fig.colorbar(im0, ax=axs[0])
        #im1 = axs[1].imshow(((Pijk - Pijk_m)[0, 0]).detach().cpu()[0, :, :], vmin = -maxgpx , vmax = maxgpx, cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180][0, 438:, 502:]
        #fig.colorbar(im1, ax=axs[1])
        #im2 = axs[2].imshow(((Pijk - Pijk_m)[0, 1]).detach().cpu()[0, :, :], vmin = -maxgpy , vmax = maxgpy, cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180] [0, 438:, 502:]
        #fig.colorbar(im2, ax=axs[2])
        #fig.tight_layout()
        #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results_lt_nograd/grad_p.png')
        #fig, axs = plt.subplots(1, 2, figsize=(6,3))
        #im0 = axs[0].imshow((U[0, 0]).detach().cpu()[0, :, :], cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im0, ax=axs[0])
        #im1 = axs[1].imshow((U[0, 1]).detach().cpu()[0, :, :], cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im1, ax=axs[1])
        #fig.tight_layout()
        #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results_lt_nograd/U_prev.png')
        #fig, axs = plt.subplots(1, 1, figsize=(6,3))
        #maxdiv = torch.max((velocityDivergence(U, flags)[0, 0]).detach().cpu()[0, :, :])
        #im0 = axs.imshow((velocityDivergence(U, flags)[0, 0]).detach().cpu()[0, :, :], vmin = -0.1*maxdiv , vmax = 0.1*maxdiv, cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im0, ax=axs)
        #fig.tight_layout()
        #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results_lt_nograd/div_in.png')
        #plt.close()

        mask_fluid = mask_fluid*0 +1

        U[:,:,:,1:(h-1),1:(w-1)] = (mask_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
            mask_fluid_obstacle * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - Pijk) + \
            mask_obstacle_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) + Pijk_m) + \
            mask_inflow * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) +            
            mask_fluid_inflow * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
            mask_inflow_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
            mask_obstacle_inflow * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
            mask_inflow_obstacle * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
            mask_no_fluid * (0))
        
        #fig, axs = plt.subplots(2, 4, figsize=(20,6))
        #im00 = axs[0, 0].imshow(((U)[0, 0]).detach().cpu()[0, 118:, 118:], cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im00, ax=axs[0, 0])
        #im10 = axs[1, 0].imshow(((U)[0, 1]).detach().cpu()[0, 118:, 118:], cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im10, ax=axs[1, 0])
        #im01 = axs[0, 1].imshow(((U)[0, 0]).detach().cpu()[0, :10, 118:], cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im01, ax=axs[0, 1])
        #im11 = axs[1, 1].imshow(((U)[0, 1]).detach().cpu()[0, :10, 118:], cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im11, ax=axs[1, 1])
        #im02 = axs[0, 2].imshow(((U)[0, 0]).detach().cpu()[0, 118:, :10], cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im02, ax=axs[0, 2])
        #im12 = axs[1, 2].imshow(((U)[0, 1]).detach().cpu()[0, 118:, :10], cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im12, ax=axs[1, 2]) 
        #im03 = axs[0, 3].imshow(((U)[0, 0]).detach().cpu()[0, :10, :10], cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im03, ax=axs[0, 3])
        #im13 = axs[1, 3].imshow(((U)[0, 1]).detach().cpu()[0, :10, :10], cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im13, ax=axs[1, 3])
        #fig.tight_layout()
        #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results_lt_nograd/U_after.png')
        #fig, axs = plt.subplots(1, 4, figsize=(10,3))
        #im0 = axs[0].imshow((velocityDivergence(U, flags)[0, 0]).detach().cpu()[0, 118:, 118:], vmin = -0.01*maxdiv, vmax = 0.01*maxdiv, cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im0, ax=axs[0])
        #im1 = axs[1].imshow((velocityDivergence(U, flags)[0, 0]).detach().cpu()[0, :10, 118:], vmin = -0.01*maxdiv, vmax = 0.01*maxdiv, cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im1, ax=axs[1])
        #im2 = axs[2].imshow((velocityDivergence(U, flags)[0, 0]).detach().cpu()[0, 118:, :10], vmin = -0.01*maxdiv, vmax = 0.01*maxdiv, cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im2, ax=axs[2])
        #im3 = axs[3].imshow((velocityDivergence(U, flags)[0, 0]).detach().cpu()[0, :10, :10], vmin = -0.01*maxdiv, vmax = 0.01*maxdiv, cmap=plt.get_cmap('seismic')) #[0, 180:260, 120:180]
        #fig.colorbar(im3, ax=axs[3])
        #fig.tight_layout()
        #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results_lt_nograd/div_out.png')
        #plt.close()

        #import pdb
        #pdb.set_trace()

        #print("flags After Adding 7", flags[0,0,0,0:5,30:64])
        #U[:,:,:,1:(h-1),1:(w-1)] = (mask_fluid * \
        #    (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
        #     mask_fluid_obstacle * \
        #    (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - Pijk) + \
        #    mask_obstacle_fluid * \
        #    (U.narrow(4, 1, w-2).narrow(3, 1, h-2) + Pijk_m) + \
        #    mask_inflow * \
        #    (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
        #    mask_fluid_inflow * \
        #    (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
        #    mask_no_fluid * (0))
        #print('******************************************************')

        #print('masks')
        #print('*    *         *            *')
        #print(mask_fluid)
        #print('*    *         *            *')
        #print(mask_fluid_obstacle)
        #print(mask_obstacle_fluid)
        #print(mask_no_fluid)
        #print('******************************************************')
    else:
        U[:,:,1:(d-1),1:(h-1),1:(w-1)] =  mask * \
            (U.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2) - (Pijk - Pijk_m))



def velocityUpdate_Density(pressure, U, flags, density):

    r""" Calculate the pressure gradient and subtract it into (i.e. calculate
    U' = U - grad(p)/rho). Some care must be taken with handling boundary conditions.
    This function mimics correctVelocity in Manta.
    Velocity update is done IN-PLACE.

    Arguments:
        p (Tensor): scalar pressure field.
        U (Tensor): velocity field (size(2) can be 2 or 3, indicating 2D / 3D)
        flags (Tensor): input occupancy grid
        density (Tensor): scalar density field.
    """
    # Check arguments.
    assert U.dim() == 5 and flags.dim() == 5 and pressure.dim() == 5 and density.dim(), \
               "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"
    b = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (U.size(1) == 3)
    if not is3D:
        assert d == 1, "d > 1 for a 2D domain"
        assert U.size(4) == w, "2D velocity field must have only 2 channels"

    assert U.size(0) == b and U.size(2) == d and U.size(3) == h \
        and U.size(4) == w, "size mismatch"
    assert pressure.is_same_size(flags), "size mismatch"
    assert density.is_same_size(flags), "size mismatch"
    assert U.is_contiguous() and flags.is_contiguous() and \
              pressure.is_contiguous() and density.is_contiguous(), "Input is not contiguous"


    # First, we build the mask for detecting fluid cells. Borders are left untouched.
    # mask_fluid   Fluid cells.
    # mask_fluid_i Fluid cells with (i-1) neighbour also a fluid.
    # mask_fluid_j Fluid cells with (j-1) neighbour also a fluid.
    # mask_fluid_k FLuid cells with (k-1) neighbour also a fluid.

    # Second, we detect obstacle cells
    # See Bridson p44 for algorithm and boundaries treatment.

    if not is3D:
        # Current cell is fluid
        mask_fluid = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid)

        # Current is fluid and neighbour to left or down are fluid
        mask_fluid_i = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid))
        mask_fluid_j = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeFluid))

        # Current cell is fluid and neighbours to left or down are obstacle
        mask_fluid_obstacle_im1 = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeObstacle))
        mask_fluid_obstacle_jm1 = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeEmpty))
        #mask_fluid_obstacle_im1 = mask_fluid.__and__ \
        #    (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeEmpty))
        #mask_fluid_obstacle_jm1 = mask_fluid.__and__ \
        #    (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeEmpty))

        mask_obstacle = flags.narrow(4, 1, w-2).narrow(3, 1, h-2) \
                            .eq(CellType.TypeObstacle).__and__ \
                     (flags.narrow(4, 1, w-2).narrow(3, 1, h-2) \
                            .ne(CellType.TypeOutflow))

        # Current cell is obstacle and neighbours to left or down are fluid
        mask_obstacle_fluid_im1 = mask_obstacle.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid))
        mask_obstacle_fluid_jm1 = mask_obstacle.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeFluid))

        # Current cell is obstacle and neighbours to left or down are not fluid
        mask_no_fluid_im1 = mask_obstacle.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeEmpty))
        mask_no_fluid_jm1 = mask_obstacle.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeEmpty))

    else:
        # TODO: implement 3D bcs well.
        # TODO: add outlfow (change in advection required)
        mask_fluid  = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2).eq(CellType.TypeFluid)
        mask_fluid_i = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2).eq(CellType.TypeFluid))
        mask_fluid_j = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 0, h-1).narrow(2, 1, d-2).eq(CellType.TypeFluid))
        mask_fluid_k = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 0, d-2).eq(CellType.TypeFluid))

    # Cast into float or double tensor and cat into a single mask along chan.
    mask_fluid_i_f = mask_fluid_i.type(U.type())
    mask_fluid_j_f = mask_fluid_j.type(U.type())

    mask_fluid_obstacle_i_f = mask_fluid_obstacle_im1.type(U.type())
    mask_fluid_obstacle_j_f = mask_fluid_obstacle_jm1.type(U.type())

    mask_obstacle_fluid_i_f = mask_obstacle_fluid_im1.type(U.type())
    mask_obstacle_fluid_j_f = mask_obstacle_fluid_jm1.type(U.type())

    mask_no_fluid_i_f = mask_no_fluid_im1.type(U.type())
    mask_no_fluid_j_f = mask_no_fluid_jm1.type(U.type())


 
    if is3D:
        mask_fluid_k_f = mask_fluid_k.type(U.type())

    if not is3D:
        mask_fluid = torch.cat((mask_fluid_i_f, mask_fluid_j_f), 1).contiguous()
        mask_fluid_obstacle = torch.cat((mask_fluid_obstacle_i_f, mask_fluid_obstacle_j_f), 1).contiguous()
        mask_obstacle_fluid = torch.cat((mask_obstacle_fluid_i_f, mask_obstacle_fluid_j_f), 1).contiguous()
        mask_no_fluid = torch.cat((mask_no_fluid_i_f, mask_no_fluid_j_f), 1).contiguous()
    else:
        mask_fluid = torch.cat((mask_fluid_i_f, mask_fluid_j_f, mask_fluid_k_f), 1).contiguous()

    # pressure tensor.
    #Pijk    Pressure at (i,j,k) in 3 channels (2 for 2D).
    #Pijk_m  Pressure at chan 0: (i-1, j, k)
              #          chan 1: (i, j-1, k)
              #          chan 2: (i, j, k-1)
    #Pijk_p  Pressure at chan 0: (i+1, j, k)
              #          chan 1: (i, j+1, k)
              #          chan 2: (i, j, k+1)


    if not is3D:
        Pijk = pressure.narrow(4, 1, w-2).narrow(3, 1, h-2)
        Pijk = Pijk.clone().expand(b, 2, d, h-2, w-2)
        Pijk_m = Pijk.clone().expand(b, 2, d, h-2, w-2)
        Pijk_p = Pijk.clone().expand(b, 2, d, h-2, w-2)
        Pijk_m[:,0] = pressure.narrow(4, 0, w-2).narrow(3, 1, h-2).squeeze(1)
        Pijk_m[:,1] = pressure.narrow(4, 1, w-2).narrow(3, 0, h-2).squeeze(1)
        Pijk_p[:,0] = pressure.narrow(4, 2, w-2).narrow(3, 1, h-2).squeeze(1)
        Pijk_p[:,1] = pressure.narrow(4, 1, w-2).narrow(3, 2, h-2).squeeze(1)

        Rhoijk = density.narrow(4, 1, w-2).narrow(3, 1, h-2)
        Rhoijk = Rhoijk.clone().expand(b, 2, d, h-2, w-2)
        Rhoijk_m = Rhoijk.clone().expand(b, 2, d, h-2, w-2)
        Rhoijk_p = Rhoijk.clone().expand(b, 2, d, h-2, w-2)
        Rhoijk_m[:,0] = density.narrow(4, 0, w-2).narrow(3, 1, h-2).squeeze(1)
        Rhoijk_m[:,1] = density.narrow(4, 1, w-2).narrow(3, 0, h-2).squeeze(1)
        Rhoijk_p[:,0] = density.narrow(4, 2, w-2).narrow(3, 1, h-2).squeeze(1)
        Rhoijk_p[:,1] = density.narrow(4, 1, w-2).narrow(3, 2, h-2).squeeze(1)

        #print("Pijk ", Pijk)
        #print("Pijk_m ", Pijk_m)
        #print("Pijk_p ", Pijk_p)


    else:
        Pijk = pressure.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2)
        Pijk = Pijk.clone().expand(b, 3, d-2, h-1, w-1)
        Pijk_m = Pijk.clone().expand(b, 3, d-2, h-1, w-1)
        Pijk_m[:,0] = pressure.narrow(4, 0, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2).squeeze(1)
        Pijk_m[:,1] = pressure.narrow(4, 1, w-1).narrow(3, 0, h-1).narrow(2, 1, d-2).squeeze(1)
        Pijk_m[:,2] = pressure.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 0, d-2).squeeze(1)

    # grad(p) = [[ p(i,j,k) - p(i-1,j,k) ]
    #            [ p(i,j,k) - p(i,j-1,k) ]
    #            [ p(i,j,k) - p(i,j,k-1) ]]
    if not is3D:
        # Three cases:
        # 1) Cell is fluid and left neighbour is fluid:
        # u = u - grad(p)
        # 2) Cell is fluid and left neighbour is obstacle
        # u = u - p(i,j)
        # 3) Cell is obstacle and left neighbour is fluid
        # u = u + p(i-1,j)

                
        U[:,:,:,1:(h-1),1:(w-1)] = (mask_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)/(1-Rhoijk)) + \
            mask_fluid_obstacle * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk_p - Pijk)) + \
            #mask_fluid_obstacle * \
            #(U.narrow(4, 1, w-2).narrow(3, 1, h-2) - Pijk) + \
            mask_obstacle_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) + Pijk_m)) 

    else:
        U[:,:,1:(d-1),1:(h-1),1:(w-1)] =  mask * \
            (U.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2) - (Pijk - Pijk_m))


