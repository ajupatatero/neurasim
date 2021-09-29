import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_models.architectures.Unet import UNet
from neural_models.architectures.multi_scale_net import MultiScaleNet
from neural_models.architectures.multi_scale_net_small import MultiScaleNetSmall
from math import inf
from engines.phi.torch.flow import *

from engines.phi.field._grid import Grid, CenteredGrid, StaggeredGrid, unstack_staggered_tensor
from engines.phi.field import GeometryMask, AngularVelocity, Grid, StaggeredGrid, divergence, CenteredGrid, spatial_gradient, where, HardGeometryMask
from engines.phi.math._tensors import NativeTensor
from neural_models.train.util_train import *
from neural_models.architectures.old_fnet import *
import pdb
import copy



class _ScaleNet(nn.Module):
    def __init__(self, mconf):
        super(_ScaleNet, self).__init__()
        self.mconf = mconf

    def forward(self, x):
        bsz = x.size(0)
        # Rehaspe form (b x chan x d x h x w) to (b x -1)
        y = x.view(bsz, -1)
        # Calculate std using Bessel's correction (correction with n/n-1)
        std = torch.std(y, dim=1, keepdim=True) # output is size (b x 1)
        scale = torch.clamp(std, \
            self.mconf['normalizeInputThreshold'] , inf)
        scale = scale.view(bsz, 1, 1, 1, 1)

        return scale

class _HiddenConvBlock(nn.Module):
    def __init__(self, dropout=True):
        super(_HiddenConvBlock, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3, padding = 0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3, padding = 0),
            nn.ReLU(),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class FluidNet(nn.Module):
    # For now, only 2D model. Add 2D/3D option. Only known from data!
    # Also, build model with MSE of pressure as loss func, therefore input is velocity
    # and output is pressure, to be compared to target pressure.
    def __init__(self, mconf, it, folder, dropout=True):
        super(FluidNet, self).__init__()


        self.dropout = dropout
        self.mconf = mconf
        self.inDims = mconf['inputDim']
        self.is3D = mconf['is3D']
        self.it = it
        self.folder =folder

        self.scale = _ScaleNet(self.mconf)
        # Input channels = 3 (inDims, flags)
        # We add padding to make sure that Win = Wout and Hin = Hout with ker_size=3
        self.conv1 = torch.nn.Conv2d(self.inDims, 16, kernel_size=3, padding=1)

        self.modDown1 = torch.nn.AvgPool2d(kernel_size=2)
        self.modDown2 = torch.nn.AvgPool2d(kernel_size=4)

        self.convBank = _HiddenConvBlock(dropout=False)

        #self.deconv1 = torch.nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        #self.deconv2 = torch.nn.ConvTranspose2d(16, 16, kernel_size=4, stride=4)

        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(16, 8, kernel_size=1)

        # Output channels = 1 (pressure)
        self.convOut = torch.nn.Conv2d(8, 1, kernel_size=1)

        # MultiScaleNet
        self.multiScale = MultiScaleNet(self.inDims)

    def forward(self, input_, it,folder):

        # data indexes     |           |
        #       (dim 1)    |    2D     |    3D
        # ----------------------------------------
        #   DATA:
        #       pDiv       |    0      |    0
        #       UDiv       |    1:3    |    1:4
        #       flags      |    3      |    4
        #       densityDiv |    4      |    5
        #   TARGET:
        #       p          |    0      |    0
        #       U          |    1:3    |    1:4
        #       density    |    3      |    4

        # For now, we work ONLY in 2d


        assert self.is3D == False, 'Input can only be 2D'

        assert self.mconf['inputChannels']['pDiv'] or \
                self.mconf['inputChannels']['UDiv'] or \
                self.mconf['inputChannels']['div'], 'Choose at least one field (U, div or p).'

        pDiv = None
        UDiv = None
        div = None

        # Flags are always loaded
        if self.is3D:
            flags = input_[:,4].unsqueeze(1)
        else:
            flags = input_[:,3].unsqueeze(1).contiguous()

        if (self.mconf['inputChannels']['pDiv'] or (self.mconf['normalizeInput'] \
            and self.mconf['normalizeInputChan'] == 'pDiv')):
            pDiv = input_[:,0].unsqueeze(1).contiguous()

        if (self.mconf['inputChannels']['UDiv'] or self.mconf['inputChannels']['div'] \
            or (self.mconf['normalizeInput'] \
            and self.mconf['normalizeInputChan'] == 'UDiv')):
            if self.is3D:
                UDiv = input_[:,1:4].contiguous()
            else:
                UDiv = input_[:,1:3].contiguous()

            # Apply setWallBcs to zero out obstacles velocities on the boundary
            UDiv = setWallBcs(UDiv, flags)

            if self.mconf['inputChannels']['div']:
                div = velocityDivergence(UDiv, flags)

        # Apply scale to input
        if self.mconf['normalizeInput']:
            if self.mconf['normalizeInputChan'] == 'UDiv':
                s = self.scale(UDiv)
            elif self.mconf['normalizeInputChan'] == 'pDiv':
                s = self.scale(pDiv)
            elif self.mconf['normalizeInputChan'] == 'div':
                s = self.scale(div)
            else:
                raise Exception('Incorrect normalize input channel.')

            if pDiv is not None:
                pDiv = torch.div(pDiv, s)
            if UDiv is not None:
                UDiv = torch.div(UDiv, s)
            if div is not None:
                div = torch.div(div, s)

        x = torch.FloatTensor(input_.size(0), \
                              self.inDims,    \
                              input_.size(2), \
                              input_.size(3), \
                              input_.size(4)).type_as(input_)

        chan = 0
        if self.mconf['inputChannels']['pDiv']:
            x[:, chan] = pDiv[:,0]
            chan += 1
        elif self.mconf['inputChannels']['UDiv']:
            if self.is3D:
                x[:,chan:(chan+3)] = UDiv
                chan += 3
            else:
                x[:,chan:(chan+2)] = UDiv
                chan += 2
        elif self.mconf['inputChannels']['div']:
            x[:, chan] = div[:,0]
            chan += 1

        # FlagsToOccupancy creates a [0,1] grid out of the manta flags
        x[:,chan,:,:,:] = flagsToOccupancy(flags).squeeze(1)

        if not self.is3D:
            # Squeeze unary dimension as we are in 2D
            x = torch.squeeze(x,2)

        if self.mconf['model'] == 'ScaleNet':

            # Declare variables
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Start recording
            start_event.record()

            print("is X contiguous ", x.is_contiguous())
            # Network it
            p_out = self.multiScale(x)

            # Finish recording
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)
            time = elapsed_time_ms

            p= p_out[:,0,...].unsqueeze(1).contiguous()
            print("is p contiguous ", p.is_contiguous())

        else:
            # Inital layers
            x = F.relu(self.conv1(x))

            # We divide the network in 3 banks, applying average pooling
            x1 = self.modDown1(x)
            x2 = self.modDown2(x)

            # Process every bank in parallel
            x0 = self.convBank(x)
            x1 = self.convBank(x1)
            x2 = self.convBank(x2)

            # Upsample banks 1 and 2 to bank 0 size and accumulate inputs

            #x1 = self.upscale1(x1)
            #x2 = self.upscale2(x2)

            x1 = F.interpolate(x1, scale_factor=2)
            x2 = F.interpolate(x2, scale_factor=4)
            #x1 = self.deconv1(x1)
            #x2 = self.deconv2(x2)

            #x = torch.cat((x0, x1, x2), dim=1)
            x = x0 + x1 + x2

            # Apply last 2 convolutions
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

            # Output pressure (1 chan)
            p = self.convOut(x)

        # Add back the unary dimension
        if not self.is3D:
            p = torch.unsqueeze(p, 2)

        # Correct U = UDiv - grad(p)
        # flags is the one with Manta's values, not occupancy in [0,1]

        print("IS P contiguous  model line 212", p.is_contiguous())
        print("P shape model line 213", p.shape)

        velocityUpdate(pressure=p, U=UDiv, flags=flags)

        # We now UNDO the scale factor we applied on the input.
        if self.mconf['normalizeInput']:
            p = torch.mul(p,s)  # Applies p' = *= scale
            UDiv = torch.mul(UDiv,s)

        # Set BCs after velocity update.
        UDiv = setWallBcs(UDiv, flags)


        return p, UDiv, time


class PhiflowNet_old(nn.Module):
    # For now, only 2D model. Add 2D/3D option. Only known from data!
    # Also, build model with MSE of pressure as loss func, therefore input is velocity
    # and output is pressure, to be compared to target pressure.
    def __init__(self, mconf, it, folder, dropout=True):
        super(PhiflowNet_old, self).__init__()


        self.dropout = dropout
        self.mconf = mconf
        self.inDims = mconf['inputDim']
        self.is3D = mconf['is3D']
        self.it = it
        self.folder =folder
        self.scale = _ScaleNet(self.mconf)
        self.call_num = 0

        # Network
        self.multiScale = UNet(self.inDims)

    #def forward(self, input_, velocity, vel_mask, DOMAIN, lt_bool, it,folder):
    def forward(self, input_, velocity, vel_mask, DOMAIN, lt_bool, it, folder):

        # data indexes     |           |
        #       (dim 1)    |    2D     |    3D
        # ----------------------------------------
        #   DATA:
        #       pDiv       |    0      |    0
        #       UDiv       |    1:3    |    1:4
        #       flags      |    3      |    4
        #       densityDiv |    4      |    5
        #   TARGET:
        #       p          |    0      |    0
        #       U          |    1:3    |    1:4
        #       density    |    3      |    4

        # For now, we work ONLY in 2d


        assert self.is3D == False, 'Input can only be 2D'

        assert self.mconf['inputChannels']['pDiv'] or \
                self.mconf['inputChannels']['UDiv'] or \
                self.mconf['inputChannels']['div'], 'Choose at least one field (U, div or p).'

        pDiv = None
        UDiv = None
        div = None

        # Declare sizes
        self.nnx = len(input_[0,0,0,0])
        self.nny = len(input_[0,0,0,:,0])

        # Flags are always loaded
        if self.is3D:
            flags = input_[:,4].unsqueeze(1)
        else:
            flags = input_[:,3].unsqueeze(1).contiguous()

        if (self.mconf['inputChannels']['pDiv'] or (self.mconf['normalizeInput'] \
            and self.mconf['normalizeInputChan'] == 'pDiv')):
            pDiv = input_[:,0].unsqueeze(1).contiguous()

        if (self.mconf['inputChannels']['UDiv'] or self.mconf['inputChannels']['div'] \
            or (self.mconf['normalizeInput'] \
            and self.mconf['normalizeInputChan'] == 'UDiv')):
            if self.is3D:
                UDiv = input_[:,1:4].contiguous()
            else:
                UDiv = input_[:,1:3].contiguous()

            # Apply setWallBcs to zero out obstacles velocities on the boundary
            UDiv = setWallBcs(UDiv, flags)

            if self.mconf['inputChannels']['div']:
                div = velocityDivergence(UDiv, flags)

        # Apply scale to input
        if self.mconf['normalizeInput']:
            if self.mconf['normalizeInputChan'] == 'UDiv':
                s = self.scale(UDiv)
            elif self.mconf['normalizeInputChan'] == 'pDiv':
                s = self.scale(pDiv)
            elif self.mconf['normalizeInputChan'] == 'div':
                s = self.scale(div)
            else:
                raise Exception('Incorrect normalize input channel.')

            if pDiv is not None:
                pDiv = torch.div(pDiv, s)
            #if UDiv is not None:
            #    UDiv = torch.div(UDiv, s)
            if div is not None:
                div = torch.div(div, s)

        x = torch.FloatTensor(input_.size(0), \
                              self.inDims,    \
                              input_.size(2), \
                              input_.size(3), \
                              input_.size(4)).type_as(input_)

        chan = 0
        if self.mconf['inputChannels']['pDiv']:
            x[:, chan] = pDiv[:,0]
            chan += 1
        elif self.mconf['inputChannels']['UDiv']:
            if self.is3D:
                x[:,chan:(chan+3)] = UDiv
                chan += 3
            else:
                x[:,chan:(chan+2)] = UDiv
                chan += 2
        elif self.mconf['inputChannels']['div']:
            x[:, chan] = div[:,0]
            chan += 1

        # FlagsToOccupancy creates a [0,1] grid out of the manta flags
        x[:,chan,:,:,:] = flagsToOccupancy(flags).squeeze(1)

        if not self.is3D:
            # Squeeze unary dimension as we are in 2D
            x = torch.squeeze(x,2)

        # Declare variables
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # When performing the lt loss, no need for reinitializing a velocity field from the torch field.
        if not lt_bool:
            bsz = UDiv.size(0)
            dim = UDiv.size(1)
            nnz = UDiv.size(2)
            nny = UDiv.size(3)
            nnx = UDiv.size(4)

            UDiv_big = torch.zeros((bsz, dim, nnz, nny+1, nnx+1)).cuda()
            UDiv_big[:,:,:,:-1,:-1] = UDiv

            #velocity = DOMAIN.vector_grid(math.wrap(UDiv.squeeze(2), 'batch,vector,x,y'))
            #velocity = DOMAIN.staggered_grid(math.wrap(UDiv_big.squeeze(2), 'batch,vector,x,y'))
            #tensor_U = NativeTensor(UDiv_big.squeeze(2), 'batch,vector,y,x')

            velocity_init =  DOMAIN.staggered_grid(1)
            tensor_U = math.wrap(UDiv_big.squeeze(2), 'batch,vector,y,x')
            lower = math.wrap(velocity_init.box.lower)
            upper = math.wrap(velocity_init.box.upper)
            extrapolation = math.extrapolation.ZERO
            tensor_U_unstack = unstack_staggered_tensor(tensor_U)
            velocity =  StaggeredGrid(tensor_U_unstack, geom.Box(lower, upper), extrapolation)

            flags_big = torch.zeros((bsz, dim, nnz, nny+1, nnx+1)).cuda()
            flags_big_next = torch.zeros((bsz, dim, nnz, nny+1, nnx+1)).cuda()
            flags_big[:,0,:,:-1,:-1] = (2-flags.squeeze(1))
            flags_big_next[:,0,:,:-1,1:] = (2-flags.squeeze(1))
            flags_big[:,0] *= flags_big_next[:,0]

            flags_big[:,1,:,:-1,:-1] = (2-flags.squeeze(1))
            flags_big_next[:,1,:,1:,:-1] = (2-flags.squeeze(1))
            flags_big[:,1] *= flags_big_next[:,1]

            tensor_flags = math.wrap(flags_big.squeeze(2), 'batch,vector,y,x')
            tensor_flag_unstack = unstack_staggered_tensor(tensor_flags)
            vel_mask =  StaggeredGrid(tensor_flag_unstack, geom.Box(lower, upper), extrapolation)


        flags_mask = CenteredGrid(tensor((2-flags).squeeze(2).squeeze(1), names=['batch', 'y', 'x']), DOMAIN.bounds)
        # Apply scale to input
        if self.mconf['normalizeInput']:
            normalize = True
            vel_nor = torch.cat((velocity.staggered_tensor().tensors[0]._native.unsqueeze(1),
                                      velocity.staggered_tensor().tensors[1]._native.unsqueeze(1)), dim=1)
            vel_reduced = vel_nor[:,:,:-1,:-1].contiguous()
            scale_factor = self.scale(vel_reduced)[:,0,0,0,0]
            scale_factor_phi = tensor(scale_factor,names=['batch'])

        else:
            normalize = False
            scale_factor_phi = tensor(torch.ones_like(flags[:,0,0,0,0]),names=['batch'])

        # Start recording
        start_event.record()

        self.call_num +=1

        pressure, velocity, div_out, div_in = fluid.make_incompressible_with_network(velocity, flags_mask, DOMAIN, self.multiScale, normalize, scale_factor_phi)

        # Network it
        #p_out = self.multiScale(x)


        # Finish recording
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        time = elapsed_time_ms

        # After Correction set BC!
        velocity *= vel_mask

        return pressure, velocity, div_out, div_in, vel_mask, time


class PhiflowNet(nn.Module):
    # For now, only 2D model. Add 2D/3D option. Only known from data!
    # Also, build model with MSE of pressure as loss func, therefore input is velocity
    # and output is pressure, to be compared to target pressure.
    def __init__(self, mconf, it, folder, dropout=True):
        super(PhiflowNet, self).__init__()


        self.dropout = dropout
        self.mconf = mconf
        self.inDims = mconf['inputDim']
        self.is3D = mconf['is3D']
        self.it = it
        self.folder =folder
        #self.scale = _ScaleNet(self.mconf)
        self.scale = mconf.get("scale_factor", None)
        if self.scale == None:
            self.scale = _ScaleNet(self.mconf)
        self.call_num = 0

        # Network
        self.multiScale = UNet(self.inDims)

    def forward(self, velocity, flags, domain, norm_conf, epoch, batch_idx, str):

        # PHIFLOW MAKE INCOMPRESSIBLE!
        # First store original velocity
        input_velocity = velocity

        # Get divergence and normalize!
        div_in = divergence(input_velocity)
        div_torch = div_in.values._native.transpose(-1, -2).unsqueeze(1).unsqueeze(2)

        #UDiv = torch.cat((velocity.staggered_tensor().tensors[0]._native.transpose(-1, -2).unsqueeze(1),
        #                velocity.staggered_tensor().tensors[1]._native.transpose(-1, -2).unsqueeze(1)), dim=1)[:,:,:-1,:-1]
        #UDiv = UDiv.unsqueeze(2).contiguous()
        #div_torch = velocityDivergence(UDiv, flags)

        # Generate network input
        #div_torch = div_in.values._native.unsqueeze(1)
        if norm_conf['normalize']:
            std_norm = torch.std(div_torch, (1, 2, 3, 4)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            std_mask_v, std_mask_p = get_std_phiflow(velocity, domain, std_norm)
            div_torch /= (std_norm.unsqueeze(1)*norm_conf['scale_factor'])
            velocity /= (std_mask_v*norm_conf['scale_factor'])
        else:
            std_norm = torch.std(div_torch, (1, 2, 3, 4)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
            std_mask_v, std_mask_p = get_std_phiflow(velocity, domain, std_norm)

            std_mask_p = (std_mask_p *0 )+1
            std_mask_v = (std_mask_v *0 )+1

        #div_torch *= flags

        #div_in = CenteredGrid(math.tensor(div_torch.squeeze(1).squeeze(1).squeeze(1).transpose(-1, -2), names=['batch', 'x', 'y']), domain.bounds)

        # Before entering the network, the torch strucutre should be retaken!
        # Remember torch follows (bsz, channel, nny, nnx), flags should already be OK
        # Phiflow follows (bsz, channel, nnx, nny)
        #div_torch = div_torch.transpose(-1, -2)

        #div_torch = div_torch * flags.squeeze(2)
        #div_torch[:, :, :, :, -2:] =0.0
        network_in = torch.cat((div_torch.squeeze(2), flags.squeeze(2)), dim=1)


        # Network inference
        pressure_torch, time_Unet = self.multiScale(network_in)

        #pressure_torch = pressure_torch* flags.squeeze(2)

        # Pressure in obstacles!
        #pressure_torch *= (flags.squeeze(2))
        pressure = CenteredGrid(math.tensor(pressure_torch.squeeze(1).transpose(-1, -2), names=['batch', 'x', 'y']), domain.bounds)
        gradp = field.spatial_gradient(pressure, type=type(input_velocity))

        # Instead of using Phiflows spatial gradients and velocity update, let's use the old ones!
        _, _, UDiv = convert_phi_to_torch(velocity, pressure, pressure)
        UDiv, pressure_torch = UDiv.unsqueeze(2).contiguous(), pressure_torch.unsqueeze(2).contiguous()

        #velocityUpdate(pressure=pressure_torch, U=UDiv, flags=flags)
        #velocity, _ = load_values_phi(UDiv, flags, domain)

        #pressure_torch = pressure_torch.transpose(-1,-2).contiguous()
        #UDiv = UDiv.transpose(-1,-2).contiguous()
        pressure_torch *= input_velocity.dx[0]._native
        velocityUpdate(pressure=pressure_torch, U=UDiv, flags=flags)

        #velocity, _ = load_values_phi(UDiv, flags, domain)
        velocity, _ = load_values(UDiv, flags, domain)


        div_out_torch = velocityDivergence(UDiv, flags)

        #div_out = CenteredGrid(math.tensor(div_out_torch.squeeze(1).squeeze(1).transpose(-1, -2), names=['batch', 'x', 'y']), domain.bounds)
        div_out = divergence(velocity)

        # Subtract grad pressure
        #velocity = (velocity - gradp).with_(extrapolation=velocity.extrapolation)
        # Calculate final divergence
        #div_out = divergence(velocity)
        #if batch_idx == 0:
        #    self.debug(input_velocity, velocity, pressure, gradp, domain, div_torch.squeeze(2), flags, div_out, div_in, epoch, batch_idx, str, std_mask_v, norm_conf)

        if norm_conf['normalize']:
            pressure *= (std_mask_p*norm_conf['scale_factor'])
            velocity *= (std_mask_v*norm_conf['scale_factor'])

        return pressure, velocity, div_out, div_in, time_Unet

    def debug(self, input_velocity, velocity, pressure, gradp, domain, div_torch, flags, div_out, div_in, epoch, batch_idx, str, std_mask_v, norm_conf):

        with torch.no_grad():

            # Now solve with torch to debug
            obstacles = ()
            active = domain.grid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), extrapolation=domain.boundaries['active_extrapolation'])
            accessible = domain.grid(active, extrapolation=domain.boundaries['accessible_extrapolation'])
            hard_bcs = field.stagger(accessible, math.minimum, domain.boundaries['accessible_extrapolation'], type=type(velocity))
            #velocity = layer_obstacle_velocities(velocity * hard_bcs, obstacles).with_(extrapolation=domain.boundaries['near_vector_extrapolation'])
            div_CG = div_in
            div_CG = divergence(input_velocity/ (std_mask_v*norm_conf['scale_factor']))


            # Solve pressure
            def laplace(p):
                grad = spatial_gradient(p, type(velocity))
                grad = grad.with_(extrapolation=domain.boundaries['near_vector_extrapolation'])
                div = divergence(grad)
                lap = where(active, div, p)
                return lap

            pressure_guess = None
            solve_params= math.LinearSolve(None, 1e-4)
            pressure_guess = pressure_guess if pressure_guess is not None else domain.scalar_grid(0)
            converged, pressure_CG, iterations = field.solve(laplace, y=div_CG, x0=pressure_guess, solve_params=solve_params, constants=[active, hard_bcs])
            if math.all_available(converged) and not math.all(converged):
                raise AssertionError(f"pressure solve did not converge after {iterations} iterations\nResult: {pressure_CG.values}")

            # Subtract grad pressure
            gradp_CG = field.spatial_gradient(pressure_CG, type=type(velocity)) * hard_bcs

            pressure_CG_torch = pressure_CG.values._native.unsqueeze(1).transpose(-1, -2)
            _, _, UDiv_CG = convert_phi_to_torch(input_velocity/(std_mask_v*norm_conf['scale_factor']), pressure_CG, pressure_CG)
            UDiv_CG, pressure_CG_torch = UDiv_CG.unsqueeze(2).contiguous(), pressure_CG_torch.unsqueeze(2).contiguous()

            velocityUpdate(pressure=pressure_CG_torch, U=UDiv_CG, flags=flags)
            #velocity_CG, _ = load_values_phi(UDiv_CG, flags, domain)
            velocity_CG, _ = load_values(UDiv_CG, flags, domain)

            gradp_x_CG_np = gradp_CG.staggered_tensor().tensors[0]._native.transpose(-1, -2).cpu().detach().numpy()[0]
            gradp_y_CG_np = gradp_CG.staggered_tensor().tensors[1]._native.transpose(-1, -2).cpu().detach().numpy()[0]
            #velocity_CG = (input_velocity/(std_mask_v*norm_conf['scale_factor']) - gradp_CG).with_(extrapolation=input_velocity.extrapolation)
            div_out_CG = divergence(velocity_CG)
            div_out_CG_torch = velocityDivergence(UDiv_CG, flags)

        debug_folder = norm_conf['debug_folder']
        if batch_idx == 0 and not os.path.isfile(os.path.join(debug_folder, 'Net_debug_{}_{}.png'.format(str, epoch))):
            p_mean = torch.mean(pressure.values._native.detach().cpu()[0, :, :])
            for i in range(0, 1):
                fig, axs = plt.subplots(1, 6, figsize=(20,3))
                axs[0].set_title('Div in torch')
                im0 = axs[0].imshow((div_torch.detach().cpu().numpy())[i, 0, :, :], vmin= -0.1*np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = 0.1*np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('seismic'))
                fig.colorbar(im0, ax=axs[0])
                axs[1].set_title('Div in CG')
                im1 = axs[1].imshow((div_CG.values._native.transpose(-1, -2)).detach().cpu()[i, :, :], vmin= -0.1*np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = 0.1*np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('seismic'))
                fig.colorbar(im1, ax=axs[1])
                axs[2].set_title('Div out torch')
                im2 = axs[2].imshow((div_out.values._native.transpose(-1, -2)).detach().cpu()[i, :, :] , vmin= -0.1*np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = 0.1*np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('seismic'))
                fig.colorbar(im2, ax=axs[2])
                axs[3].set_title('Div out CG')
                #im3 = axs[3].imshow((div_out_CG_torch).detach().cpu()[i, 0, 0, :, :], vmin= -0.1*np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = 0.1*np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('seismic'))
                im3 = axs[3].imshow((div_out_CG.values._native.transpose(-1, -2)).detach().cpu()[i, :, :], vmin= -0.1*np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = 0.1*np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('seismic'))
                fig.colorbar(im3, ax=axs[3])
                axs[4].set_title('Pressure torch')
                im4 = axs[4].imshow(pressure.values._native.transpose(-1, -2).detach().cpu()[i, :, :] -p_mean, vmin= -np.max(np.abs(pressure_CG.values._native.detach().cpu().numpy())[i, :, :]), vmax = np.max(np.abs(pressure_CG.values._native.detach().cpu().numpy())[i, :, :]), cmap=plt.get_cmap('seismic'))
                fig.colorbar(im4, ax=axs[4])
                axs[5].set_title('Pressure CG')
                im5 = axs[5].imshow(pressure_CG.values._native.transpose(-1, -2).detach().cpu()[i, :, :] , vmin= -np.max(np.abs(pressure_CG.values._native.detach().cpu().numpy())[i, :, :]), vmax = np.max(np.abs(pressure_CG.values._native.detach().cpu().numpy())[i, :, :]), cmap=plt.get_cmap('seismic'))
                fig.colorbar(im5, ax=axs[5])
                fig.tight_layout()
                fig.savefig(os.path.join(debug_folder,'Net_debug_{}_{}.png'.format(str, epoch)))
                plt.close()

            for i in range(0, 1):
                fig, axs = plt.subplots(1, 7, figsize=(20,3))
                im0 = axs[0].imshow(np.abs((div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('Reds'))
                axs[0].set_title('Div in ')
                fig.colorbar(im0, ax=axs[0])
                axs[1].set_title('Div out')
                im1 = axs[1].imshow(np.abs((div_out.values._native.transpose(-1, -2)).detach().cpu()[i, :, :] ), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())[i, 0, :, :]), cmap=plt.get_cmap('Reds'))
                fig.colorbar(im1, ax=axs[1])
                axs[2].set_title('velocity x in')
                im2 = axs[2].imshow(np.abs(input_velocity.staggered_tensor().tensors[0]._native[i].transpose(-1, -2).detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
                fig.colorbar(im2, ax=axs[2])
                axs[3].set_title('velocity y in')
                im3 = axs[3].imshow(np.abs(input_velocity.staggered_tensor().tensors[1]._native[i].transpose(-1, -2).detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
                fig.colorbar(im3, ax=axs[3])
                axs[4].set_title('velocity x out')
                im4 = axs[4].imshow(np.abs(velocity.staggered_tensor().tensors[0]._native[i].transpose(-1, -2).detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
                fig.colorbar(im4, ax=axs[4])
                axs[5].set_title('velocity y out')
                im5 = axs[5].imshow(np.abs(velocity.staggered_tensor().tensors[1]._native[i].transpose(-1, -2).detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
                fig.colorbar(im5, ax=axs[5])
                axs[6].set_title('Flags')
                im6 = axs[6].imshow(flags.detach().cpu().numpy()[i, 0, 0], cmap=plt.get_cmap('Reds'))
                fig.colorbar(im6, ax=axs[6])
                fig.tight_layout()
                fig.savefig(os.path.join(debug_folder,'Net_inside_mask_{}_{}.png'.format(str, epoch)))
                plt.close()

            gradp_x_np = gradp.staggered_tensor().tensors[0]._native.transpose(-1, -2).cpu().detach().numpy()[0]
            gradp_y_np = gradp.staggered_tensor().tensors[1]._native.transpose(-1, -2).cpu().detach().numpy()[0]

            for i in range(0, 1):
                fig, axs = plt.subplots(1, 7, figsize=(20,3))
                im0 = axs[0].imshow(np.abs((div_torch.detach().cpu().numpy())[i, 0, :, :]), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())), cmap=plt.get_cmap('Reds'))
                axs[0].set_title('Div in ')
                fig.colorbar(im0, ax=axs[0])
                axs[1].set_title('Div out')
                im1 = axs[1].imshow(np.abs((div_out.values._native.transpose(-1, -2)).detach().cpu()[i, :, :] ), vmax = np.max(np.abs(div_torch.detach().cpu().numpy())), cmap=plt.get_cmap('Reds'))
                fig.colorbar(im1, ax=axs[1])
                axs[2].set_title('velocity x in')
                im2 = axs[2].imshow(np.abs(input_velocity.staggered_tensor().tensors[0]._native[i].transpose(-1, -2).detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
                fig.colorbar(im2, ax=axs[2])
                axs[3].set_title('velocity y in')
                im3 = axs[3].imshow(np.abs(input_velocity.staggered_tensor().tensors[1]._native[i].transpose(-1, -2).detach().cpu()[:, :]), cmap=plt.get_cmap('Reds'))
                fig.colorbar(im3, ax=axs[3])
                im4 = axs[4].imshow(gradp_x_np, vmin = -np.max(gradp_x_np), vmax= np.max(gradp_x_np),cmap=plt.get_cmap('seismic'))
                axs[4].set_title('Gradp x')
                fig.colorbar(im4, ax=axs[4])
                axs[5].set_title('Gradp y')
                im5 = axs[5].imshow(gradp_y_np, vmin = -np.max(gradp_y_np), vmax= np.max(gradp_y_np),cmap=plt.get_cmap('seismic'))
                fig.colorbar(im5, ax=axs[5])
                axs[6].set_title('Pressure')
                im6 = axs[6].imshow(np.abs(pressure.values._native.transpose(-1, -2).detach().cpu()[i, :, :] ), cmap=plt.get_cmap('Reds'))
                fig.colorbar(im6, ax=axs[6])
                fig.tight_layout()
                fig.savefig(os.path.join(debug_folder, 'Net_grad_mask_{}_{}.png'.format(str, epoch)))
                plt.close()


            for i in range(0, 1):
                fig, axs = plt.subplots(1, 6, figsize=(20,3))
                im0 = axs[0].imshow(pressure.values._native.transpose(-1, -2).detach().cpu()[i, :, :]-p_mean, vmin = -0.1*np.max(np.abs(pressure_CG.values._native.detach().cpu()[i, :, :].numpy())), vmax= 0.1*np.max(np.abs(pressure_CG.values._native.detach().cpu()[i, :, :].numpy())), cmap=plt.get_cmap('seismic'))
                axs[0].set_title('Pressure ')
                fig.colorbar(im0, ax=axs[0])

                axs[1].set_title('Pressure CG')
                im1 = axs[1].imshow(pressure_CG.values._native.transpose(-1, -2).detach().cpu()[i, :, :], vmin = -0.1*np.max(np.abs(pressure_CG.values._native.detach().cpu()[i, :, :].numpy())), vmax= 0.1*np.max(np.abs(pressure_CG.values._native.detach().cpu()[i, :, :].numpy())), cmap=plt.get_cmap('seismic'))
                fig.colorbar(im1, ax=axs[1])

                axs[2].set_title('Gradp x')
                im2 = axs[2].imshow(gradp_x_np, vmin = -np.max(gradp_x_CG_np), vmax= np.max(gradp_x_CG_np),cmap=plt.get_cmap('seismic'))
                fig.colorbar(im2, ax=axs[2])

                axs[3].set_title('Gradp y')
                im3 = axs[3].imshow(gradp_y_np, vmin = -np.max(gradp_y_CG_np), vmax= np.max(gradp_y_CG_np),cmap=plt.get_cmap('seismic'))
                fig.colorbar(im3, ax=axs[3])

                axs[4].set_title('Gradp CG x')
                im4 = axs[4].imshow(gradp_x_CG_np, vmin = -np.max(gradp_x_CG_np), vmax= np.max(gradp_x_CG_np),cmap=plt.get_cmap('seismic'))
                fig.colorbar(im4, ax=axs[4])

                axs[5].set_title('Gradp CG y')
                im5 = axs[5].imshow(gradp_y_CG_np, vmin = -np.max(gradp_y_CG_np), vmax= np.max(gradp_y_CG_np),cmap=plt.get_cmap('seismic'))
                fig.colorbar(im5, ax=axs[5])

                fig.tight_layout()
                fig.savefig(os.path.join(debug_folder,'Net_grad_CG_mask_{}_{}.png'.format(str, epoch)))
                plt.close()
