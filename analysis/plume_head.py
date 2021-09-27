import numpy as np
import torch
import torch.utils.data
import os
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import *
import argparse
from tqdm import tqdm
import pdb


def plot_bars(array_p, labels, name_save, y_lim=None):

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, array_p[:, 0], width, color='blue', label = '3 Scales')
    rects2 = ax.bar(x , array_p[:, 1], width, color='red', label = '4 Scales')
    rects3 = ax.bar(x + width, array_p[:, 2], width, color='green', label= '5 Scales')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'$\mathcal{E}$')
    ax.set_xticks(x)
    ax.set_ylim(0, 0.025)
    ax.set_xticklabels(labels)
    #ax.legend()
    if y_lim != None:
        ax.set_ylim(y_lim)

    fig.tight_layout()

    fig.savefig(folder_im + name_save)


# Initialize parse
# 3 arguments: Richardson number, cg ref to plot and margin Percent
parser = argparse.ArgumentParser(description='Plot Specified cases (Ri and Geom).')
parser.add_argument('-cg', '--cgmetric',  default='True',
        help='R|CG based metric.', type = str)
parser.add_argument('-P', '--Percen',  default=75,
                help='R|Percentage to identify plume head.', type = int)
parser.add_argument('-Ri', '--richardson',  default='1_0',
        help='R|Richardson number in a string (choose between 0_1, 1_0 and 10_0', type = str)
args = parser.parse_args()

# Addtional parameter to check the Ri number, plot it and margin Percent
margin_perc = args.Percen
cg_analysis = args.cgmetric == 'True' or args.cgmetric == 'true' 
Ti = 99
Ri = args.richardson

if cg_analysis:
    Networks = ['CG', 'nolt_3',
                'lt_grad_1_2', 'lt_grad_2_4', 'lt_grad_2_6', 
                'lt_nograd_2_4', 'lt_nograd_4_8', 'lt_nograd_4_16']  

colors = ['blue', 'red', 'green']


if Ri == '0_1':
    every = 10
elif Ri == '1_0':
    every = 5
elif Ri == '10_0':
    every = 2

res = 128

# Base folders and Networks
folder =  '/tmpdir/ajuriail/neuralsim/cases/18_plume/'
folder_im = folder + 'Images/CG/'

if not os.path.exists(folder_im):
    os.makedirs(folder_im)

# Array to compute integral
pos_array = np.zeros((len(Networks), Ti))
integral_values = np.zeros((len(Networks)-1))
ninety_points = np.zeros((len(Networks)-1))
ref_integral = np.zeros((Ti)) 

# Max density to follow the plume
max_density = 0.01
margin = max_density - ((margin_perc/100)*max_density)
x_range = np.arange(Ti)


for i, network in enumerate(Networks):

    # Basee folder and its subfolders
    folder_load = folder + 'results_{}/Ri_{}/'.format(network, Ri)

    density = np.zeros(Ti)
    Pixels_sum = np.zeros(Ti)
    x_range = np.arange(Ti)

    # Loop through files
    for itt in tqdm(range(Ti)):
        # Load file
        filename = '/Rho_NN_output_{0:05}.npy'.format((itt+1)*every)

        folderfile = folder_load + filename
        rho_loaded = np.load(folderfile)
        
        # Get center line and loop through to get head's position
        Line_rho = rho_loaded[0, res//2, :]

        for j in range(res):
            if Line_rho[j]>margin:
                Line_rho[j] = 1
            else:
                Line_rho[j] = 0

        Pixels_sum[itt] = np.sum(Line_rho)

        print('Pixel sum ', Pixels_sum)

        if Pixels_sum[itt]/res > 0.9 and i==0:
            ref_value = Pixels_sum[itt]/res
            jacobi_tt = itt

    if i ==0:
        ref_integral = Pixels_sum/res

    # For integral and Point Distance
    if cg_analysis and i!=0:
        ninety_points[i-1] = np.abs(Pixels_sum[jacobi_tt]/res - ref_value)
        pos_array[i-1] =  Pixels_sum/res

    plt.plot(x_range[1:-1], Pixels_sum[1:-1]/res, color=colors[i%3], linestyle='dashed', linewidth=1, markersize=8, label = network)

    
savefile_png = folder_im + 'Head_Ri_{}.png'.format(Ri)
savefile_pdf = folder_im + 'Head_Ri_{}.png'.format(Ri)


plt.legend(fontsize=12)
#plt.yscale("log")
#plt.ylim(0,0.025)
plt.locator_params(axis='y', nbins=5)
plt.locator_params(axis='x', nbins=6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$\widetilde{t}$',fontsize=16)
plt.ylabel(r'$\widetilde{h}$', fontsize = 16, rotation = 0)

plt.savefig(savefile_png)
plt.savefig(savefile_pdf)

plt.close('all')
