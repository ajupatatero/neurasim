import glob
import sys
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import random
import glob
from shutil import copyfile
import importlib.util
import pdb

from ..dataset_managers.dataset import *
from ..architectures.model import *
from .trainer import *
from .util_train import * 

def launch_train(conf):

    conf['dataDir'] = conf['dataDir']
    conf['modelDir'] = conf['modelDir']
    conf['modelFilename'] = conf['modelFilename']
    conf['modelDirname'] = conf['modelDir'] + '/' + conf['modelFilename']
    output_mode = conf['printTraining']
    shuffle_training = conf['shuffleTraining']
    assert output_mode == 'save' or output_mode == 'show' or output_mode == 'none',\
            'In config.yaml printTraining options are save, show or none.'

    resume = conf['resumeTraining']

    # Preprocessing dataset message (will exit after preproc)
    if (conf['preprocOriginalFluidNetDataOnly']):
        print('Running preprocessing only')
        resume = False

    print('Active CUDA Device: GPU', torch.cuda.current_device())
    cuda0 = torch.device('cuda:0')

    # Define training and test datasets
    tr = FluidNetDataset(conf, 'tr', save_dt=4, resume=resume)
    te = FluidNetDataset(conf, 'te', save_dt=4, resume=resume)

    if (conf['preprocOriginalFluidNetDataOnly']):
        sys.exit()

    # We create two conf dicts, general params and model params.
    conf, mconf = tr.createConfDict()

    # Separate some variables from conf dict. When resuming training, this ones will
    # overwrite saved conf (where model is saved).
    # User can modify them in YAML config file or in command line.
    num_workers = conf['numWorkers']
    batch_size = conf['batchSize']
    max_epochs = conf['maxEpochs']
    print_training = output_mode == 'show' or output_mode == 'save'
    save_or_show = output_mode == 'save'
    lr = mconf['lr']

    # Initialize Phiflow domain in GPU
    if torch.cuda.is_available():
        TORCH_BACKEND.set_default_device('GPU')

    #******************************** Restarting training ***************************

    if resume:
        print()
        print('            RESTARTING TRAINING            ')
        print()
        print('==> loading checkpoint')
        mpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_lastEpoch_best.pth')
        assert glob.os.path.isfile(mpath), mpath  + ' does not exits!'
        state = torch.load(mpath)

        print('==> overwriting conf and file_mconf')
        cpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_conf.pth')
        mcpath = glob.os.path.join(conf['modelDir'], conf['modelFilename'] + '_mconf.pth')
        assert glob.os.path.isfile(mpath), cpath  + ' does not exits!'
        assert glob.os.path.isfile(mpath), mcpath  + ' does not exits!'
        conf.update(torch.load(cpath))
        mconf.update(torch.load(mcpath))

        print('==> copying and loading corresponding model module')
        path = conf['modelDir']
        path_list = path.split(glob.os.sep)
        saved_model_name = glob.os.path.join('/', *path_list, path_list[-1] + '_saved.py')
        #temp_model = glob.os.path.join('lib', path_list[-1] + '_saved_resume.py')
        #copyfile(saved_model_name, temp_model)

        #assert glob.os.path.isfile(temp_model), temp_model  + ' does not exits!'
        #spec = importlib.util.spec_from_file_location('model_saved', temp_model)
        spec = importlib.util.spec_from_file_location('model_saved', saved_model_name)
        model_saved = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_saved)

    print('Data loading: done')

    try:
        # Create train and validation loaders
        print('Number of workers: ' + str(num_workers) )

        train_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size, \
                num_workers=num_workers, shuffle=shuffle_training, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(te, batch_size=batch_size, \
                num_workers=num_workers, shuffle=False, pin_memory=True)

        #********************************** Create the model ***************************
        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.kaiming_uniform_(m.weight)

        print('')
        print('------------------- Model ----------------------------')

        # Create model and print layers and params
        if not resume:
            net = PhiflowNet(mconf, 0, conf['modelDir'])
        else:
            net = model_saved.PhiflowNet(mconf, 0, path)

        if torch.cuda.is_available():
            net = net.cuda()

        # Initialize network weights with Kaiming normal method (a.k.a MSRA)
        net.apply(init_weights)
        #lib.summary(net, (5,1,128,128))

        if resume:
            net.load_state_dict(state['state_dict'])

        #********************** Define the optimizer ***********************

        print('==> defining optimizer')

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        if resume:
            optimizer.load_state_dict(state['optimizer'])

        for param_group in optimizer.param_groups:
            print('lr of optimizer')
            print(param_group['lr'])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',factor = 0.9, patience = 10, verbose = True, threshold = 1e-2, threshold_mode = 'rel', min_lr=1e-5)

        #************************ Training and Validation*******************
        # Test set Scenes idx to plot during validation
        list_to_plot = [batch_size, 40*batch_size, 80*batch_size, 178*batch_size]



        # Create some arrays for recording results
        train_loss_plot = np.empty((0,7))
        val_loss_plot = np.empty((0,7))

        # Save loss to disk
        m_path = conf['modelDir']
        model_save_path = glob.os.path.join(m_path, conf['modelFilename'])

        # Save loss as numpy arrays to disk
        p_path = conf['modelDir']
        file_train = glob.os.path.join(p_path, 'train_loss')
        file_val = glob.os.path.join(p_path, 'val_loss')

        # Save mconf and conf to disk
        file_conf = glob.os.path.join(m_path, conf['modelFilename'] + '_conf.pth')
        file_mconf = glob.os.path.join(m_path, conf['modelFilename'] + '_mconf.pth')


        # raw_input returns the empty string for "enter"
        yes = {'yes','y', 'ye', ''}
        no = {'no','n'}

        if resume:
            start_epoch = state['epoch']
        else:
            start_epoch = 1
            if ((not glob.os.path.exists(p_path)) and (not glob.os.path.exists(m_path))):
                if (p_path == m_path):
                    glob.os.makedirs(p_path)
                else:
                    glob.os.makedirs(m_path)
                    glob.os.makedirs(p_path)

            # Here we are a bit barbaric, and we copy the whole model.py into the saved model
            # folder, so that we don't lose the network architecture.
            # We do that only if resuming training.
            path, last = glob.os.path.split(m_path)
            saved_model_name = glob.os.path.join(path, last, last + '_saved.py')
            copyfile('/tmpdir/ajuriail/neuralsim/fluidnet_3/neural_models/architectures/model.py', saved_model_name)

            # Delete plot file if starting from scratch
            if (glob.os.path.isfile(file_train + '.npy') and glob.os.path.isfile(file_val + '.npy')):
                print('Are you sure you want to delete existing files and start training from scratch. [y/n]')
                #Ekhi Debug 
                glob.os.remove(file_train + '.npy')
                glob.os.remove(file_val + '.npy')              
                #choice = input().lower()

        # Save config dicts
        torch.save(conf, file_conf)
        torch.save(mconf, file_mconf)

        #********************************* Run epochs ****************************************

        trainer = Trainer(net, train_loader, test_loader, optimizer, conf, mconf,
                        scheduler=None)

        n_epochs = max_epochs
        if not resume:
            state = {}
            state['bestPerf'] = float('Inf')

        print('')
        print('==> Beginning simulation')
        for epoch in range(start_epoch, n_epochs+1):

            train_loss, p_l2_tr, div_l2_tr, p_l1_tr, div_l1_tr, div_lt_tr = trainer._train_epoch(epoch, path) 
            val_loss, p_l2_val, div_l2_val, p_l1_val, div_l1_val, div_lt_val = trainer._val_epoch(epoch, path, list_to_plot)

            #Step scheduler, will reduce LR if loss has plateaued
            scheduler.step(train_loss)

            for param_group in optimizer.param_groups:
                print('lr of optimizer')
                print(param_group['lr'])

            # Store training loss function
            train_loss_plot = np.append(train_loss_plot, [[epoch, train_loss, p_l2_tr,
                div_l2_tr, p_l1_tr, div_l1_tr, div_lt_tr]], axis=0)
            val_loss_plot = np.append(val_loss_plot, [[epoch, val_loss, p_l2_val,
                div_l2_val, p_l1_val, div_l1_val, div_lt_val]], axis=0)

            # Check if this is the best model so far and if so save to disk
            is_best = False
            state['epoch'] = epoch +1
            state['state_dict'] = net.state_dict()
            state['optimizer'] = optimizer.state_dict()

            if val_loss < state['bestPerf']:
                is_best = True
                state['bestPerf'] = val_loss
            save_checkpoint(state, is_best, m_path, 'convModel_lastEpoch.pth')

            # Save loss to disk -- TODO: Check if there is a more efficient way, instead
            # of loading the whole file...
            if epoch % conf['freqToFile'] == 0:
                plot_train_file = file_train + '.npy'
                plot_val_file = file_val + '.npy'
                train_old = np.empty((0,7))
                val_old = np.empty((0,7))
                if (glob.os.path.isfile(plot_train_file) and glob.os.path.isfile(plot_val_file)):
                    train_old = np.load(plot_train_file)
                    val_old = np.load(plot_val_file)
                train_loss = np.append(train_old, train_loss_plot, axis=0)
                val_loss = np.append(val_old, val_loss_plot, axis=0)
                np.save(file_val, val_loss)
                np.save(file_train, train_loss)
                # Reset arrays
                train_loss_plot = np.empty((0,7))
                val_loss_plot = np.empty((0,7))

    finally:
        if resume:
            # Delete model_saved_resume.py
            print()
            print('Deleting ' + temp_model)
            glob.os.remove(temp_model)
