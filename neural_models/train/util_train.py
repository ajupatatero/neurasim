import torch
import random
import glob
import pdb
import numpy as np
from shutil import copyfile
from engines.phi.torch.flow import *
from engines.phi.field import divergence
from engines.phi.field._grid import Grid, CenteredGrid, StaggeredGrid, unstack_staggered_tensor
from util.operations.field_operate import *

def lt_setting(config, is3D):

    # Check if additional buoyancy or gravity is added to future frames.
    # Adding Buoyancy means adding a source term in the momentum equation, of
    # the type f = delta_rho*g and rho = rho_0 + delta_rho (constant term + fluctuation)
    # with rho' << rho_0
    # Adding gravity: source of the type f = rho_0*g
    # Source term is a vector (direction and magnitude).

    oldBuoyancyScale = config['buoyancyScale']
    # rand(1) is an uniform dist on the interval [0,1)
    if torch.rand(1)[0] < config['trainBuoyancyProb']:
        # Add buoyancy to this batch (only in the long term frames)
        var = torch.tensor([1.]).cuda()
        config['buoyancyScale'] = torch.normal(config['trainBuoyancyScale'], var)

    oldGravityScale = config['gravityScale']
    # rand(1) is an uniform dist on the interval [0,1)
    if torch.rand(1)[0] < config['trainGravityProb']:
        # Add gravity to this batch (only in the long term frames)
        var = torch.tensor([1.]).cuda()
        config['gravityScale'] = torch.normal(config['trainGravityScale'], var)

    oldGravity = config['gravityVec']
    if config['buoyancyScale'] > 0 or config['gravityScale'] > 0:
        # Set to 0 gravity vector (direction of gravity)
        config['gravityVec']['x'] = 0
        config['gravityVec']['y'] = 0
        config['gravityVec']['z'] = 0

        # Chose randomly one of three cardinal directions and set random + or - dir
        card_dir = 0
        if is3D:
            card_dir = random.randint(0,2)
        else:
            card_dir = random.randint(0,1)

        updown = random.randint(0,1) * 2 - 1
        if card_dir == 0:
            config['gravityVec']['x'] = updown
        elif card_dir == 1:
            config['gravityVec']['y'] = updown
        elif card_dir == 2:
            config['gravityVec']['z'] = updown

    base_dt = config['dt']

    if config['timeScaleSigma'] > 0:
        # FluidNet: randn() returns normal distribution with mean 0 and var 1.
        # The mean of abs(randn) ~= 0.7972, hence the 0.2028 value below.
        scale_dt = 0.2028 + torch.abs(torch.randn(1))[0] * \
                config['timeScaleSigma']
        config['dt'] = base_dt * scale_dt

    num_future_steps = config['longTermDivNumSteps'][0]
    # rand(1) is an uniform dist on the interval [0,1)
    # longTermDivProbability is the prob that longTermDivNumSteps[0] is taken.
    # otherwise, longTermDivNumSteps[1] is taken with prob 1 - longTermDivProbability
    if torch.rand(1)[0] > config['longTermDivProbability']:
        num_future_steps = config['longTermDivNumSteps'][1]

    return base_dt, num_future_steps, config

def convert_phi_to_torch(velocity, pressure, div_out):
    """ Useful function to convert from Phiflow to torch the fields for the losses

    Args:
        velocity ([type]): [description]
        pressure ([type]): [description]
        div_out ([type]): [description]

    Returns:
        [type]: [description]
    """
    out_p = pressure
    out_U = velocity

    div_out_t = div_out.values._native
    out_p_t = out_p.values._native
    out_U_t = torch.cat((out_U.staggered_tensor().tensors[0]._native.unsqueeze(1),
                            out_U.staggered_tensor().tensors[1]._native.unsqueeze(1)), dim=1)[:,:,:-1,:-1]

    # Change order from (nnx, nny) to (nny, nnx) !
    div_out_t = div_out_t.transpose(-1, -2)
    out_p_t = out_p_t.transpose(-1, -2)
    out_U_t = out_U_t.transpose(-1, -2)

    return div_out_t, out_p_t, out_U_t

def convert_torch_to_phi(p, U, in_U_t, flags, domain):
    """ Useful function to convert from torch values to Phiflow fields for the losses

    Args:
        p ([type]): [description]
        U ([type]): [description]
        in_U_t ([type]): [description]

    Returns:
        [type]: [description]
    """

    pressure = CenteredGrid(math.tensor(p.squeeze(1).squeeze(1).transpose(-1,-2), names=['batch', 'x', 'y']), domain.bounds)

    velocity, vel_mask = load_values(U, flags, domain)
    velocity_in, _ = load_values(in_U_t.unsqueeze(2)[:,:,:,:-1,:-1] , flags, domain)

    # divergence
    div_in = divergence(velocity_in)
    div_out = divergence(velocity)

    return pressure, velocity, vel_mask, div_out, div_in

def get_std_phiflow(velocity, domain, std_norm):
    lower = math.wrap(velocity.box.lower)
    upper = math.wrap(velocity.box.upper)
    extrapolation = math.extrapolation.ZERO
    #tensor_std = math.wrap(torch.cat((std_norm, std_norm), dim=1).expand(-1, -1, velocity.box.upper._native[0] +1 , velocity.box.upper._native[1] +1), 'batch,vector,x,y')
    tensor_std = math.wrap(torch.cat((std_norm, std_norm), dim=1).expand(-1, -1, velocity.cells._shape[0] +1 , velocity.cells._shape[1] +1), 'batch,vector,x,y')

    tensor_std_unstack = unstack_staggered_tensor(tensor_std)
    std_mask_v =  StaggeredGrid(tensor_std_unstack, geom.Box(lower, upper), extrapolation)

    std_mask_p = CenteredGrid(math.tensor(std_norm[:,0].expand(-1, velocity.box.upper._native[0] +1 , velocity.box.upper._native[1] +1), names=['batch', 'x', 'y']), domain.bounds)

    return std_mask_v, std_mask_p

def create_from_flags(flags, velocity):

    bsz = flags.size(0)
    nnx = flags.size(-2)
    nny = flags.size(-1)
    flags_big = torch.zeros((bsz, 2, 1, nnx+1, nny+1)).cuda()
    flags_big_next = torch.zeros((bsz, 2, 1, nnx+1, nny+1)).cuda()
    flags_big[:,0,:,:-1,:-1] = (1-flags.squeeze(1))
    flags_big_next[:,0,:,1:, :-1] = (1-flags.squeeze(1))
    flags_big[:,0] *= flags_big_next[:,0]

    flags_big[:,1,:,:-1,:-1] = (1-flags.squeeze(1))
    flags_big_next[:,1,:,:-1, 1:] = (1-flags.squeeze(1))
    flags_big[:,1] *= flags_big_next[:,1]

    tensor_flags = math.wrap(flags_big.squeeze(2), 'batch,vector,x,y')
    tensor_flag_unstack = unstack_staggered_tensor(tensor_flags)
    lower = math.wrap(velocity.box.lower)
    upper = math.wrap(velocity.box.upper)
    extrapolation = math.extrapolation.ZERO
    vel_mask =  StaggeredGrid(tensor_flag_unstack, geom.Box(lower, upper), extrapolation)

    return vel_mask

def save_checkpoint(state, is_best, save_path, filename):
    filename = glob.os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = glob.os.path.join(save_path, 'convModel_lastEpoch_best.pth')
        copyfile(filename, bestname)

def load_values(UDiv, flags, domain):

    # Get dimensions for tensor creation
    bsz = UDiv.size(0)
    dim = UDiv.size(1)
    nnz = UDiv.size(2)
    nny = UDiv.size(3)
    nnx = UDiv.size(4)

    # Create velocity tensor with size nny+1 nnx+1, as FluidNet assumes that value = 0
    UDiv_big = torch.zeros((bsz, dim, nnz, nny+1, nnx+1)).cuda()
    UDiv_big[:,:,:,:-1,:-1] = UDiv

    # Transpose as the structure in torch is (bsz, channel, nnz, nny, nnx)
    # In Phiflow the structure is rather (bsz, channel, nnx, nny)
    UDiv_big = UDiv_big.transpose(-1, -2)

    velocity_init =  domain.staggered_grid(1)
    tensor_U = math.wrap(UDiv_big.squeeze(2), 'batch,vector,x,y')
    lower = math.wrap(velocity_init.box.lower)
    upper = math.wrap(velocity_init.box.upper)
    extrapolation = math.extrapolation.ZERO
    tensor_U_unstack = unstack_staggered_tensor(tensor_U)
    velocity =  StaggeredGrid(tensor_U_unstack, geom.Box(lower, upper), extrapolation)


    flags_big = torch.zeros((bsz, dim, nnz, nny+1, nnx+1)).cuda()
    flags_big_next = torch.zeros((bsz, dim, nnz, nny+1, nnx+1)).cuda()
    flags_loaded = torch.zeros((bsz, dim, nnz, nny+1, nnx+1)).cuda()
    flags_big[:,0,:,:-1,:-1] = (2-flags.squeeze(1))
    flags_loaded[:,0,:,:-1,:-1] = (2-flags.squeeze(1))

    flags_big_next[:,0,:,:-1,1:] = (2-flags.squeeze(1))
    flags_big[:,0] *= flags_big_next[:,0]

    flags_big[:,1,:,:-1,:-1] = (2-flags.squeeze(1))
    flags_big_next[:,1,:,1:,:-1] = (2-flags.squeeze(1))
    flags_loaded[:,1,:,:-1,:-1] = (2-flags.squeeze(1))
    flags_big[:,1] *= flags_big_next[:,1]

    # Transpose as the structure in torch is (bsz, channel, nnz, nny, nnx)
    # In Phiflow the structure is rather (bsz, channel, nnx, nny)
    flags_big = flags_big.transpose(-1, -2)

    tensor_flags = math.wrap(flags_big.squeeze(2), 'batch,vector,x,y')
    tensor_flag_unstack = unstack_staggered_tensor(tensor_flags)
    vel_mask =  StaggeredGrid(tensor_flag_unstack, geom.Box(lower, upper), extrapolation)

    return velocity, vel_mask

def change_nan_zero(velocity, domain):

    ux_torch = velocity.staggered_tensor().tensors[0]._native.unsqueeze(1)
    uy_torch = velocity.staggered_tensor().tensors[1]._native.unsqueeze(1)

    ux_torch = torch.nan_to_num(ux_torch)
    uy_torch = torch.nan_to_num(uy_torch)

    velocity_big = torch.cat((ux_torch, uy_torch), dim=1)

    velocity_init =  domain.staggered_grid(1)
    tensor_U = math.wrap(velocity_big.squeeze(2), 'batch,vector,x,y')
    lower = math.wrap(velocity_init.box.lower)
    upper = math.wrap(velocity_init.box.upper)
    extrapolation = math.extrapolation.ZERO
    tensor_U_unstack = unstack_staggered_tensor(tensor_U)
    velocity =  StaggeredGrid(tensor_U_unstack, geom.Box(lower, upper), extrapolation)

    return velocity