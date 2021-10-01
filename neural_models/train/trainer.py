import glob
import sys
import argparse
import yaml
import os
import numpy as np
import random
from shutil import copyfile
import importlib.util
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from engines.phi.torch.flow import *
from engines.phi.field._grid import Grid, CenteredGrid, StaggeredGrid, unstack_staggered_tensor
from .util_train import lt_setting, convert_phi_to_torch, load_values
from .plot_train import plot_train
from ..architectures.old_fnet import *
from simulation.Train_sim import *

class Trainer:
    """
    Trainer class.
    Train a model for the given criterion and optimizer.  A scheduler may be used,
    as well as a different validation dataloader than the training dataloader.
    """
    def __init__(self, model, train_data_loader, val_data_loader, optimizer, config_train, config_sim,
                       scheduler=None):

        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.optimizer = optimizer
        self.config = config_sim
        self.config_train = config_train
        self.scheduler = scheduler

        self.output_mode = config_train['printTraining']
        self.print_training = self.output_mode == 'show' or self.output_mode == 'save'
        self.save_or_show = self.output_mode == 'save'
        self.plot_every = config_train['freqToFile']

        self.bsz = config_train['batchSize']
        self.lt_loss = self.config['divLongTermLambda'] > 0
        self.lt_grad = self.config['ltGrad']

        # Simulation initialization
        self.trainsim = TrainSim(self.config, self.model, self.bsz)

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.m_path = config_train['modelDir']


    def compute_loss_st(self, velocity, pressure, div_out, data, target):

        # Convert to torch
        div_out_t, out_p_t, out_U_t = convert_phi_to_torch(velocity, pressure, div_out)

        # Calculate targets
        target_p = target[:,0].unsqueeze(1)
        target_div = torch.zeros_like(div_out_t)

        # Measure loss
        pL2Loss = self.config['pL2Lambda'] * self.mse(out_p_t, target_p[:,0,0])
        divL2Loss = self.config['divL2Lambda'] * self.mse(div_out_t, target_div)
        pL1Loss =  self.config['pL1Lambda'] * self.l1(out_p_t, target_p[:,0,0])
        divL1Loss = self.config['divL1Lambda'] * self.l1(div_out_t, target_div)

        return pL2Loss, divL2Loss, pL1Loss, divL1Loss


    def compute_loss_lt(self, velocity, pressure, div_out):

        # Convert to torch
        div_out_t, out_p_t, out_U_t = convert_phi_to_torch(velocity, pressure, div_out)

        # Calculate targets
        target_div_LT = torch.zeros_like(div_out_t)

        return self.config['divLongTermLambda']*self.mse(div_out_t, target_div_LT)

    def initialize_losses(self):
        self.total_loss = 0
        self.p_l2_total_loss = 0
        self.div_l2_total_loss = 0
        self.p_l1_total_loss = 0
        self.div_l1_total_loss = 0
        self.div_lt_total_loss = 0

    def add_terms(self, pL2Loss, divL2Loss, pL1Loss, divL1Loss, divLTLoss, loss_size):
        self.p_l2_total_loss += pL2Loss.data.item()
        self.div_l2_total_loss += divL2Loss.data.item()
        self.p_l1_total_loss += pL1Loss.data.item()
        self.div_l1_total_loss += divL1Loss.data.item()
        self.div_lt_total_loss += divLTLoss.data.item()
        self.total_loss += loss_size.data.item()

    def mean_terms(self, n_batches):
        self.p_l2_total_loss /= n_batches
        self.div_l2_total_loss /= n_batches
        self.p_l1_total_loss /= n_batches
        self.div_l1_total_loss /= n_batches
        self.div_lt_total_loss /= n_batches
        self.total_loss /= n_batches

    def losslist(self):
        return [self.total_loss, self.p_l2_total_loss,
                self.div_l2_total_loss, self.div_lt_total_loss,
                self.p_l1_total_loss, self.div_l1_total_loss]

    def load_from_mantaflow(self, input_, domain):

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

        is3D = len(input_[0, 0, :, 0, 0]) > 1

        assert is3D == False, 'Input can only be 2D'

        # Declare sizes
        self.nnx = len(input_[0,0,0,0])
        self.nny = len(input_[0,0,0,:,0])

        # Flags and velocity is loaded
        if is3D:
            flags = input_[:,4].unsqueeze(1)
            UDiv = input_[:,1:4].contiguous()
        else:
            flags = input_[:,3].unsqueeze(1).contiguous()
            UDiv = input_[:,1:3].contiguous()

        velocity, vel_mask = load_values(UDiv, flags, domain)

        return velocity, vel_mask



    def _train_epoch(self, epoch, path):
        """
        Training method for the specified epoch.
        Returns a log that contains the average loss and metric in this epoch.
        """
        self.model.train()
        self.initialize_losses()
        n_batches = 0
        #loop through data, sorted into batches
        for batch_idx, (data, target) in enumerate (self.train_data_loader):

            if data.size(0) == self.bsz:

                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                self.optimizer.zero_grad()

                is3D = data.size(1) == 6
                assert (is3D and data.size(1) == 6) or (not is3D and data.size(1) == 5), "Data must have \
                        5 input chan for 2D, 6 input chan for 3D"
                flags = data[:,3].unsqueeze(1).contiguous()

                # Create Velocity and Vel_mask tensors
                self.trainsim.velocity, self.trainsim.vel_mask = self.load_from_mantaflow(data, self.trainsim.domain)

                self.trainsim.pressure, self.trainsim.velocity, div_out, div_in = self.model( self.trainsim.velocity,
                            (2-flags), self.trainsim.domain, self.config['normalization'], epoch, batch_idx, 'train')

                # Compute st loss
                pL2Loss, divL2Loss, pL1Loss, divL1Loss = self.compute_loss_st(self.trainsim.velocity, self.trainsim.pressure,
                                                                div_out, data, target)
                loss = pL2Loss + divL2Loss + pL1Loss + divL1Loss

                if self.lt_loss and not self.lt_grad:
                    with torch.no_grad():
                        # Get configuration and dt after lt randomization
                        self.trainsim.dt, num_future_steps, self.config = lt_setting(self.config, is3D)

                        # Forward the simulation using Phiflow!
                        for i in range(0, num_future_steps):
                            div_out, div_in = self.trainsim.run(i, epoch, flags)

                        self.trainsim.run_star()

                    pressure, velocity, div_out, div_in = self.model( self.trainsim.velocity, (2-flags),
                                    self.trainsim.domain, self.config['normalization'], epoch, batch_idx, 'final_nograd_lt')

                    divLTLoss = self.compute_loss_lt(velocity, pressure, div_out)
                    loss += divLTLoss.item()



                elif self.lt_loss:

                    # Get configuration and dt after lt randomization
                    self.trainsim.dt, num_future_steps, self.config = lt_setting(self.config, is3D)

                    # Forward the simulation using Phiflow!
                    for i in range(0, num_future_steps):
                        div_out, div_in = self.trainsim.run(i, epoch, flags)

                    divLTLoss = self.compute_loss_lt(self.trainsim.velocity, self.trainsim.pressure, div_out)
                    loss += divLTLoss.item()


                # Useful Statistics
                if self.lt_loss:
                    self.add_terms(pL2Loss, divL2Loss, pL1Loss, divL1Loss, divLTLoss, loss)
                else:
                    self.add_terms(pL2Loss, divL2Loss, pL1Loss, divL1Loss, 0*divL2Loss, loss)

                # Print every 20th batch of an epoch
                if batch_idx % 20 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}\t'.format(
                        epoch, batch_idx * len(data), len(self.train_data_loader.dataset),
                        100. * batch_idx * len(data) / len(self.train_data_loader.dataset), loss))

                n_batches +=1

                # Run the backpropagation for all the losses.
                loss.backward()

                # Step the optimizer
                self.optimizer.step()
                self.optimizer.zero_grad()


            else:
                print('Skip due to Size mismatch!')

        # Divide loss by dataset length
        self.mean_terms(n_batches)

        # Print for the whole dataset
        print('\n Training set: Avg total loss: {:.6f} (L2(p): {:.6f}; L2(div): {:.6f}; \
                L1(p): {:.6f}; L1(div): {:.6f}; LTDiv: {:.6f})'.format(\
                        self.total_loss, self.p_l2_total_loss, self.div_l2_total_loss, \
                        self.p_l1_total_loss, self.div_l1_total_loss, self.div_lt_total_loss))

        # Return loss scores
        return self.total_loss, self.p_l2_total_loss, self.div_l2_total_loss, \
                self.p_l1_total_loss, self.div_l1_total_loss, self.div_lt_total_loss



    def _val_epoch(self, epoch, path, list_to_plot):
        """
        Validation method for the specified epoch.
        Returns a log that contains the average loss and metric in this epoch.
        """
        self.model.eval()
        self.initialize_losses()
        n_batches = 0

        with torch.no_grad():
            #loop through data, sorted into batches
            for batch_idx, (data, target) in enumerate (self.val_data_loader):

                if data.size(0) == self.bsz:

                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()

                    self.optimizer.zero_grad()

                    is3D = data.size(1) == 6
                    assert (is3D and data.size(1) == 6) or (not is3D and data.size(1) == 5), "Data must have \
                            5 input chan for 2D, 6 input chan for 3D"
                    flags = data[:,3].unsqueeze(1).contiguous()

                    # Create Velocity and Vel_mask tensors
                    self.trainsim.velocity, self.vel_mask = self.load_from_mantaflow(data, self.trainsim.domain)

                    self.trainsim.pressure, self.trainsim.velocity, div_out, div_in = self.model(
                                                                self.trainsim.velocity, (2-flags), self.trainsim.domain, self.config['normalization'], epoch, batch_idx, 'val')

                    # Compute st loss
                    pL2Loss, divL2Loss, pL1Loss, divL1Loss = self.compute_loss_st(self.trainsim.velocity, self.trainsim.pressure, div_out, data, target)
                    loss = pL2Loss + divL2Loss + pL1Loss + divL1Loss

                    if self.lt_loss:

                        # Get configuration and dt after lt randomization
                        self.trainsim.dt, num_future_steps, self.config = lt_setting(self.config, is3D)

                        # Forward the simulation using Phiflow!
                        for i in range(0, num_future_steps):
                            div_out, div_in = self.trainsim.run(i, epoch, flags)

                        divLTLoss = self.compute_loss_lt(self.trainsim.velocity, self.trainsim.pressure, div_out)
                        loss += divLTLoss


                    # Useful Statistics
                    if self.lt_loss:
                        self.add_terms(pL2Loss, divL2Loss, pL1Loss, divL1Loss, divLTLoss, loss)
                    else:
                        self.add_terms(pL2Loss, divL2Loss, pL1Loss, divL1Loss, 0*divL2Loss, loss)


                    # Print every 20th batch of an epoch
                    if batch_idx % 20 == 0:
                        print('Val Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}\t'.format(
                            epoch, batch_idx * len(data), len(self.val_data_loader.dataset),
                            100. * batch_idx * len(data)/ len(self.val_data_loader.dataset), loss))

                    # Print fields for debug
                    if self.print_training  and (batch_idx*len(data) in list_to_plot) and epoch % self.plot_every == 1:
                        plot_train(list_to_plot, batch_idx, epoch, data, target, flags, self.m_path,
                                self.trainsim.pressure, self.trainsim.velocity, div_out, div_in, self.trainsim.domain, self.config,
                                self.save_or_show, self.losslist())

                    n_batches +=1

                else:
                    print('Skip due to Size mismatch!')

            # Divide loss by dataset length
            self.mean_terms(n_batches)

            # Print for the whole dataset
            print('\n Validation set: Avg total loss: {:.6f} (L2(p): {:.6f}; L2(div): {:.6f}; \
                    L1(p): {:.6f}; L1(div): {:.6f}; LTDiv: {:.6f})'.format(\
                            self.total_loss, self.p_l2_total_loss, self.div_l2_total_loss, \
                            self.p_l1_total_loss, self.div_l1_total_loss, self.div_lt_total_loss))

            # Return loss scores
            return self.total_loss, self.p_l2_total_loss, self.div_l2_total_loss, \
                    self.p_l1_total_loss, self.div_l1_total_loss, self.div_lt_total_loss
