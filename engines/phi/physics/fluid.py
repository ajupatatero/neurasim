"""
Definition of Fluid, IncompressibleFlow as well as fluid-related functions.
"""
from ...phi import math, field
from ...phi.field import GeometryMask, AngularVelocity, Grid, StaggeredGrid, divergence, CenteredGrid, spatial_gradient, where, HardGeometryMask
from ...phi.geom import union
from ._boundaries import Domain

from functools import wraps
from typing import TypeVar, Tuple
import pdb

from ...phi.torch.flow import *
from ...phi.torch._torch_backend import TorchBackend
from ...phi.math._tensors import NativeTensor

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from util.operations.field_operate import *



def make_incompressible(velocity: Grid,
                        domain: Domain,
                        obstacles: tuple or list = (),
                        solve_params: math.LinearSolve = math.LinearSolve(None, 1e-3),
                        pressure_guess: CenteredGrid = None):
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.
    
    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
      velocity: Vector field sampled on a grid
      domain: Used to specify boundary conditions
      obstacles: List of Obstacles to specify boundary conditions inside the domain (Default value = ())
      pressure_guess: Initial guess for the pressure solve
      solve_params: Parameters for the pressure solve

    Returns:
      velocity: divergence-free velocity of type `type(velocity)`
      pressure: solved pressure field, `CenteredGrid`
      iterations: Number of iterations required to solve for the pressure
      divergence: divergence field of input velocity, `CenteredGrid`

    """
    input_velocity = velocity
    active = domain.grid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), extrapolation=domain.boundaries['active_extrapolation'])
    accessible = domain.grid(active, extrapolation=domain.boundaries['accessible_extrapolation'])
    hard_bcs = field.stagger(accessible, math.minimum, domain.boundaries['accessible_extrapolation'], type=type(velocity))
    velocity = layer_obstacle_velocities(velocity * hard_bcs, obstacles).with_(extrapolation=domain.boundaries['near_vector_extrapolation'])
    div = divergence(velocity)
    if domain.boundaries['near_vector_extrapolation'] == math.extrapolation.BOUNDARY:
        div -= field.mean(div)

    # Solve pressure

    def laplace(p):
        grad = spatial_gradient(p, type(velocity))
        grad *= hard_bcs
        grad = grad.with_(extrapolation=domain.boundaries['near_vector_extrapolation'])
        div = divergence(grad)
        lap = where(active, div, p)
        return lap

    pressure_guess = pressure_guess if pressure_guess is not None else domain.scalar_grid(0)
    converged, pressure, iterations = field.solve(laplace, y=div, x0=pressure_guess, solve_params=solve_params, constants=[active, hard_bcs])
    if math.all_available(converged) and not math.all(converged):
        raise AssertionError(f"pressure solve did not converge after {iterations} iterations\nResult: {pressure.values}")
    # Subtract grad pressure
    gradp = field.spatial_gradient(pressure, type=type(velocity)) * hard_bcs
    velocity = (velocity - gradp).with_(extrapolation=input_velocity.extrapolation)
    return velocity, pressure, iterations, div


def make_incompressible_BC(velocity: Grid,
                            domain: Domain,
                            obstacles: tuple or list = (),
                            solve_params: math.LinearSolve = math.LinearSolve(None, 1e-3),
                            pressure_guess: CenteredGrid = None,
                            solver = None):
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.
    
    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
      velocity: Vector field sampled on a grid
      domain: Used to specify boundary conditions
      obstacles: List of Obstacles to specify boundary conditions inside the domain (Default value = ())
      pressure_guess: Initial guess for the pressure solve
      solve_params: Parameters for the pressure solve
      solver: CG or network solver ('CG' or 'NN' are accepted)

    Returns:
      velocity: divergence-free velocity of type `type(velocity)`
      pressure: solved pressure field, `CenteredGrid`
      iterations: Number of iterations required to solve for the pressure
      divergence: divergence field of input velocity, `CenteredGrid`

    """
    input_velocity = velocity
    active = domain.grid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), extrapolation=domain.boundaries['active_extrapolation'])
    accessible = domain.grid(active, extrapolation=domain.boundaries['accessible_extrapolation'])
    hard_bcs = field.stagger(accessible, math.minimum, domain.boundaries['accessible_extrapolation'], type=type(velocity))
    velocity = layer_obstacle_velocities(velocity * hard_bcs, obstacles).with_(extrapolation=domain.boundaries['near_vector_extrapolation'])
    div = divergence(velocity)
    div_CG = div
    if domain.boundaries['near_vector_extrapolation'] == math.extrapolation.BOUNDARY:
        div -= field.mean(div)

    def laplace(p):
        grad = spatial_gradient(p, type(velocity))
        grad = grad.with_(extrapolation=domain.boundaries['near_vector_extrapolation'])
        div = divergence(grad)
        lap = where(active, div, p)
        return lap

    pressure_guess = pressure_guess if pressure_guess is not None else domain.scalar_grid(0)
    if solver == 'CG':

        t1 = torch.cuda.Event(enable_timing=True)
        t1.record()

        converged, pressure, iterations = field.solve(laplace, y=div, x0=pressure_guess, solve_params=solve_params, constants=[active, hard_bcs])
        
        t2 = torch.cuda.Event(enable_timing=True)
        t2.record()
        torch.cuda.synchronize()
        elapsed_time_ms = t1.elapsed_time(t2)

        if math.all_available(converged) and not math.all(converged):
            raise AssertionError(f"pressure solve did not converge after {iterations} iterations\nResult: {pressure.values}")
    elif solver == 'NN':
        raise NotImplementedError

    # Subtract grad pressure
    gradp = field.spatial_gradient(pressure, type=type(velocity)) #* hard_bcs
    velocity = (velocity - gradp).with_(extrapolation=input_velocity.extrapolation)

    div_out = divergence(velocity)
    div_out_CG = div_out

    return velocity, pressure, iterations, div, elapsed_time_ms


def make_incompressible_with_network(velocity: Grid,
                        flags: Grid,
                        DOMAIN: Domain,
                        neural_net: nn.Module,
                        normalize: bool,
                        scale_factor: NativeTensor
                        ):
    """
    Projects the given velocity field by solving for the pressure and subtracting its gradient.
    
    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
      velocity: Vector field sampled on a grid
      domain: Used to specify boundary conditions
      obstacles: List of Obstacles to specify boundary conditions inside the domain (Default value = ())
      neural_net: Network used to solve the Poisson Equation

    Returns:
      velocity: divergence-free velocity of type `type(velocity)`
      pressure: solved pressure field, `CenteredGrid`
      divergence: divergence field of input velocity, `CenteredGrid`

    """

    input_velocity = velocity
    obstacles = ()

    if normalize:
        velocity /= scale_factor
    #active = domain.grid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), extrapolation=domain.boundaries.active_extrapolation)
    #accessible = domain.grid(active, extrapolation=domain.boundaries.accessible_extrapolation)
    #hard_bcs = field.stagger(accessible, math.minimum, domain.boundaries.accessible_extrapolation, type=type(velocity))
    #velocity = layer_obstacle_velocities(velocity * hard_bcs, obstacles).with_(extrapolation=domain.boundaries.near_vector_extrapolation)
    # For debug
    velocity_CG = velocity

    div_in = divergence(velocity)

    #if domain.boundaries.near_vector_extrapolation == math.extrapolation.BOUNDARY:
    #    div_in -= field.mean(div_in)

    # Convert divergence into torch tensor! values.vector[i]
    #divergence_torch = div_in.copy(div_in.values)
    #print('div_in ', divergence_torch.shape) 


    components = []
    #components = [div_in.values.vector[i] for i in range(div_in.shape.spatial.names)]
    
    for i, dim in enumerate(div_in.shape.spatial.names):
        dim_in = div_in.values.vector[i]
        components.append(dim_in)

    div_torch = div_in.values._native.unsqueeze(1) #.transpose(-1, -2)
    # Div torch modif
    #div_torch[:,:,0] = 0

    flags_torch = flags.values._native
    div_torch = div_torch * (1-flags_torch.unsqueeze(1))

    network_in = torch.cat((div_torch, flags_torch.unsqueeze(1)), dim=1)

    #pdb.set_trace()
    pressure_torch, time_Unet = neural_net(network_in)
    #pdb.set_trace()  

    with torch.no_grad():

        # Now solve with torch to debug
        active = DOMAIN.grid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), extrapolation=DOMAIN.boundaries['active_extrapolation'])
        accessible = DOMAIN.grid(active, extrapolation=DOMAIN.boundaries['accessible_extrapolation'])
        hard_bcs = field.stagger(accessible, math.minimum, DOMAIN.boundaries['accessible_extrapolation'], type=type(velocity))
        #velocity = layer_obstacle_velocities(velocity * hard_bcs, obstacles).with_(extrapolation=domain.boundaries['near_vector_extrapolation'])
        div_CG = divergence(velocity_CG)
        if DOMAIN.boundaries['near_vector_extrapolation'] == math.extrapolation.BOUNDARY:
            div_CG -= field.mean(div_CG)  

        # Solve pressure

        def laplace(p):
            grad = spatial_gradient(p, type(velocity))
            grad = grad.with_(extrapolation=DOMAIN.boundaries['near_vector_extrapolation'])
            div = divergence(grad)
            lap = where(active, div, p)
            return lap

        pressure_guess = None
        solve_params= math.LinearSolve(None, 1e-3)
        pressure_guess = pressure_guess if pressure_guess is not None else DOMAIN.scalar_grid(0)
        converged, pressure_CG, iterations = field.solve(laplace, y=div_CG, x0=pressure_guess, solve_params=solve_params, constants=[active, hard_bcs])
        if math.all_available(converged) and not math.all(converged):
            raise AssertionError(f"pressure solve did not converge after {iterations} iterations\nResult: {pressure_CG.values}")

        # Subtract grad pressure
        gradp_CG = field.spatial_gradient(pressure_CG, type=type(velocity)) * hard_bcs
        gradp_x_CG_np = gradp_CG.staggered_tensor().tensors[0]._native.cpu().detach().numpy()[0]
        gradp_y_CG_np = gradp_CG.staggered_tensor().tensors[1]._native.cpu().detach().numpy()[0]
        velocity_CG = (velocity_CG - gradp_CG).with_(extrapolation=input_velocity.extrapolation)
        div_out_CG = divergence(velocity_CG)


    # Pressure in obstacles!
    pressure_torch *= (1-flags_torch.unsqueeze(1))

    pressure = CenteredGrid(math.tensor(pressure_torch.squeeze(1), names=['batch', 'x', 'y']), DOMAIN.bounds)

    # Subtract grad pressure
    gradp = field.spatial_gradient(pressure, type=type(input_velocity))
    gradp_x_np = gradp.staggered_tensor().tensors[0]._native.cpu().detach().numpy()[0]
    gradp_y_np = gradp.staggered_tensor().tensors[1]._native.cpu().detach().numpy()[0]
    gradp = field.gradient(pressure, type=type(velocity)) * hard_bcs
    velocity = (velocity - gradp).with_(extrapolation=velocity.extrapolation)

    velocity = layer_obstacle_velocities(velocity * hard_bcs, obstacles).with_(extrapolation=domain.boundaries.near_vector_extrapolation)
    div_out = divergence(velocity)

    plot = False
    # epoch = np.load('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results/epoch.npy')
    # lt = np.load('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results/lt.npy')
    # save_file = '/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results/Net_inside_mask_{}_0_{}.png'.format(lt, epoch)
    # valid = np.load('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results/valid.npy')

    # if plot and not os.path.isfile(save_file) and lt > 0 and valid >0 and epoch %5 ==0:

    #     for i in range(0, 1):
    #         fig, axs = plt.subplots(1, 6, figsize=(20,3))
    #         axs[0].set_title('Div in torch')
    #         im0 = axs[0].imshow((div_torch.detach().cpu().numpy())[i, 0, :, :], vmin= -np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im0, ax=axs[0])
    #         axs[1].set_title('Div in CG')
    #         im1 = axs[1].imshow((div_CG.values._native).detach().cpu()[i, :, :], vmin= -np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im1, ax=axs[1])
    #         axs[2].set_title('Div out torch')
    #         im2 = axs[2].imshow((div_out.values._native).detach().cpu()[i, :, :] , vmin= -np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im2, ax=axs[2])
    #         axs[3].set_title('Div out CG')
    #         im3 = axs[3].imshow((div_out_CG.values._native).detach().cpu()[i, :, :], vmin= -np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im3, ax=axs[3])
    #         axs[4].set_title('Pressure torch')
    #         im4 = axs[4].imshow(pressure.values._native.detach().cpu()[i, :, :], vmin= -np.max(np.abs(pressure_CG.values._native.detach().cpu().numpy())[i, :, :]), vmax = np.max(np.abs(pressure_CG.values._native.detach().cpu().numpy())[i, :, :]), cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im4, ax=axs[4])
    #         axs[5].set_title('Pressure CG')
    #         im5 = axs[5].imshow(pressure_CG.values._native.detach().cpu()[i, :, :] , vmin= -np.max(np.abs(pressure_CG.values._native.detach().cpu().numpy())[i, :, :]), vmax = np.max(np.abs(pressure_CG.values._native.detach().cpu().numpy())[i, :, :]), cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im5, ax=axs[5])
    #         fig.tight_layout()
    #         print("Inside")
    #         #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/13_lt_inference/results/Net_torch_vs_CG_{}_{}_{}.png'.format(lt, i, epoch))
    #         fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results/Net_torch_vs_CG_{}_{}.png'.format(i, epoch))
    #         plt.close()

    #     for i in range(0, 1):
    #         fig, axs = plt.subplots(1, 7, figsize=(20,3))
    #         im0 = axs[0].imshow(np.abs((div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('Reds'))
    #         axs[0].set_title('Div in ')
    #         fig.colorbar(im0, ax=axs[0])
    #         axs[1].set_title('Div out')
    #         im1 = axs[1].imshow(np.abs((div_out.values._native).detach().cpu()[i, :, :] ), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('Reds'))
    #         fig.colorbar(im1, ax=axs[1])
    #         axs[2].set_title('velocity x in')
    #         im2 = axs[2].imshow(np.abs(input_velocity.staggered_tensor().tensors[0]._native[i].detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
    #         fig.colorbar(im2, ax=axs[2])
    #         axs[3].set_title('velocity y in')
    #         im3 = axs[3].imshow(np.abs(input_velocity.staggered_tensor().tensors[1]._native[i].detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
    #         fig.colorbar(im3, ax=axs[3])
    #         axs[4].set_title('velocity x out')
    #         im4 = axs[4].imshow(np.abs(velocity.staggered_tensor().tensors[0]._native[i].detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
    #         fig.colorbar(im4, ax=axs[4])
    #         axs[5].set_title('velocity y out')
    #         im5 = axs[5].imshow(np.abs(velocity.staggered_tensor().tensors[1]._native[i].detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
    #         fig.colorbar(im5, ax=axs[5])
    #         axs[6].set_title('Pressure')
    #         im6 = axs[6].imshow(np.abs(pressure.values._native.detach().cpu()[i, :, :] ), cmap=plt.get_cmap('Reds'))
    #         fig.colorbar(im6, ax=axs[6])
    #         fig.tight_layout()
    #         #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/13_lt_inference/results/Net_inside_mask_{}_{}_{}.png'.format(lt, i, epoch))
    #         fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results/Net_inside_mask_{}_{}.png'.format(i, epoch))
    #         plt.close()
           
    #     for i in range(0, 1):
    #         fig, axs = plt.subplots(1, 7, figsize=(20,3))
    #         im0 = axs[0].imshow(np.abs((div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())), cmap=plt.get_cmap('Reds'))
    #         axs[0].set_title('Div in ')
    #         fig.colorbar(im0, ax=axs[0])
    #         axs[1].set_title('Div out')
    #         im1 = axs[1].imshow(np.abs((div_out.values._native).detach().cpu()[i, :, :] ), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())), cmap=plt.get_cmap('Reds'))
    #         fig.colorbar(im1, ax=axs[1])
    #         axs[2].set_title('velocity x in')
    #         im2 = axs[2].imshow(np.abs(input_velocity.staggered_tensor().tensors[0]._native[i].detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
    #         fig.colorbar(im2, ax=axs[2])
    #         axs[3].set_title('velocity y in')
    #         im3 = axs[3].imshow(np.abs(input_velocity.staggered_tensor().tensors[1]._native[i].detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
    #         fig.colorbar(im3, ax=axs[3])
    #         im4 = axs[4].imshow(gradp_x_np, vmin = -np.max(gradp_x_np), vmax= np.max(gradp_x_np),cmap=plt.get_cmap('seismic'))
    #         axs[4].set_title('Gradp x')
    #         fig.colorbar(im4, ax=axs[4])
    #         axs[5].set_title('Gradp y')
    #         im5 = axs[5].imshow(gradp_y_np, vmin = -np.max(gradp_y_np), vmax= np.max(gradp_y_np),cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im5, ax=axs[5])
    #         axs[6].set_title('Pressure')
    #         im6 = axs[6].imshow(np.abs(pressure.values._native.detach().cpu()[i, :, :] ), cmap=plt.get_cmap('Reds'))
    #         fig.colorbar(im6, ax=axs[6])
    #         fig.tight_layout()
    #         #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/13_lt_inference/results/Net_grad_mask_{}_{}_{}.png'.format(lt, i, epoch))
    #         fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results/Net_grad_mask_{}_{}.png'.format(i, epoch))
    #         plt.close()

    #     for i in range(0, 1):
    #         fig, axs = plt.subplots(1, 6, figsize=(20,3))
    #         im0 = axs[0].imshow(pressure.values._native.detach().cpu()[i, :, :], vmin = -np.max(np.abs(pressure_CG.values._native.detach().cpu()[i, :, :].numpy())), vmax= np.max(np.abs(pressure_CG.values._native.detach().cpu()[i, :, :].numpy())), cmap=plt.get_cmap('seismic'))
    #         axs[0].set_title('Pressure ')
    #         fig.colorbar(im0, ax=axs[0])

    #         axs[1].set_title('Pressure CG')
    #         im1 = axs[1].imshow(pressure_CG.values._native.detach().cpu()[i, :, :], vmin = -np.max(np.abs(pressure_CG.values._native.detach().cpu()[i, :, :].numpy())), vmax= np.max(np.abs(pressure_CG.values._native.detach().cpu()[i, :, :].numpy())), cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im1, ax=axs[1])

    #         axs[2].set_title('Gradp x')
    #         im2 = axs[2].imshow(gradp_x_np, vmin = -np.max(gradp_x_CG_np), vmax= np.max(gradp_x_CG_np),cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im2, ax=axs[2])

    #         axs[3].set_title('Gradp y')
    #         im3 = axs[3].imshow(gradp_y_np, vmin = -np.max(gradp_y_CG_np), vmax= np.max(gradp_y_CG_np),cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im3, ax=axs[3])

    #         axs[4].set_title('Gradp CG x')
    #         im4 = axs[4].imshow(gradp_x_CG_np, vmin = -np.max(gradp_x_CG_np), vmax= np.max(gradp_x_CG_np),cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im4, ax=axs[4])

    #         axs[5].set_title('Gradp CG y')
    #         im5 = axs[5].imshow(gradp_y_CG_np, vmin = -np.max(gradp_y_CG_np), vmax= np.max(gradp_y_CG_np),cmap=plt.get_cmap('seismic'))
    #         fig.colorbar(im5, ax=axs[5])

    #         fig.tight_layout()
    #         #fig.savefig('/tmpdir/ajuriail/neuralsim/cases/13_lt_inference/results/Net_grad_CG_mask_{}_{}_{}.png'.format(lt, i, epoch))
    #         fig.savefig('/tmpdir/ajuriail/neuralsim/cases/14_train_debug/results/Net_grad_CG_mask_{}_{}.png'.format(i, epoch))
    #         plt.close()


    if normalize:
        velocity *= scale_factor
        #pressure *= scale_factor


    return pressure_CG, velocity_CG, div_out_CG, div_CG 


def make_incompressible_vel_inflow(INFLOW: StaggeredGrid, velocity: Grid,
                        domain: Domain,
                        obstacles: tuple or list = (),
                        solve_params: math.LinearSolve = math.LinearSolve(None, 1e-3),
                        pressure_guess: CenteredGrid = None):
    input_velocity = velocity
    active = domain.grid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), extrapolation=domain.boundaries['active_extrapolation'])
    accessible = domain.grid(active, extrapolation=domain.boundaries['accessible_extrapolation'])
    hard_bcs = field.stagger(accessible, math.minimum, domain.boundaries['accessible_extrapolation'], type=type(velocity))
    velocity = layer_obstacle_velocities(velocity * hard_bcs, obstacles).with_(extrapolation=domain.boundaries['near_vector_extrapolation'])

    velocity = velocity + INFLOW

    div = divergence(velocity)
    if domain.boundaries['near_vector_extrapolation'] == math.extrapolation.BOUNDARY:
        div -= field.mean(div)

    # Solve pressure

    def laplace(p):
        grad = spatial_gradient(p, type(velocity))
        grad *= hard_bcs
        grad = grad.with_(extrapolation=domain.boundaries['near_vector_extrapolation'])
        div = divergence(grad)
        lap = where(active, div, p)
        return lap

    pressure_guess = pressure_guess if pressure_guess is not None else domain.scalar_grid(0)
    converged, pressure, iterations = field.solve(laplace, y=div, x0=pressure_guess, solve_params=solve_params, constants=[active, hard_bcs])
    if math.all_available(converged) and not math.all(converged):
        raise AssertionError(f"pressure solve did not converge after {iterations} iterations\nResult: {pressure.values}")
    # Subtract grad pressure
    gradp = field.spatial_gradient(pressure, type=type(velocity)) * hard_bcs
    velocity = (velocity - gradp).with_(extrapolation=input_velocity.extrapolation)
    return velocity, pressure, iterations, div
                        
def divergence_cuda(field: Grid) -> CenteredGrid:
    """
    Computes the divergence of a grid using finite differences.
    This function can operate in two modes depending on the type of `field`:
    * `CenteredGrid` approximates the divergence at cell centers using central differences
    * `StaggeredGrid` exactly computes the divergence at cell centers
    Args:Ò
        field: vector field as `CenteredGrid` or `StaggeredGrid`
    Returns:
        Divergence field as `CenteredGrid`
    """
    if isinstance(field, StaggeredGrid):
        components = []
        print('shape names ', field.shape.spatial.names) 
        for i, dim in enumerate(field.shape.spatial.names):
            div_dim = math.gradient(field.values.vector[i], dx=1.0, difference='forward', padding=None, dims=[dim]).gradient[0]
            print('div dim ', div_dim)
            components.append(div_dim)
        print('len ', len(components), components)
        data = math.sum(components, 0)
        return CenteredGrid(data, field.box, field.extrapolation.gradient())
    elif isinstance(field, CenteredGrid):
        left, right = shift(field, (-1, 1), stack_dim='div_')
        grad = (right - left) / (field.dx * 2)
        components = [grad.vector[i].div_[i] for i in range(grad.div_.size)]
        result = sum(components)
        return result
    else:
        raise NotImplementedError(f"{type(field)} not supported. Only StaggeredGrid allowed.")

def layer_obstacle_velocities(velocity: Grid, obstacles: tuple or list):
    """
    Enforces obstacle boundary conditions on a velocity grid.
    Cells inside obstacles will get their velocity from the obstacle movement.
    Cells outside will be unaffected.

    Args:
      velocity: centered or staggered velocity grid
      obstacles: sequence of Obstacles
      velocity: Grid: 
      obstacles: tuple or list: 

    Returns:
      velocity of same type as `velocity`

    """
    for obstacle in obstacles:
        if not obstacle.is_stationary:
            obs_mask = GeometryMask(obstacle.geometry)
            obs_mask = obs_mask.at(velocity)
            angular_velocity = AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity, falloff=None).at(velocity)
            obs_vel = angular_velocity + obstacle.velocity
            velocity = (1 - obs_mask) * velocity + obs_mask * obs_vel
    return velocity



