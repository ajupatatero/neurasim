import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import BoundaryNorm
import matplotlib.image as mpimg

from analysis.mesure import *



def meta_Phase_Diagram(in_dir, out_dir, alpha, Re, Nx, Ny, Nt, D, Re_ref=None):
    density=1
    V0=1
    
    if Re_ref is None:
        Re_ref = Re

    fig, ax = plt.subplots()

    for i, alpha_i in enumerate(alpha):
        try:
            vforce = np.load(f'{in_dir}A_{alpha_i}_RE_{Re}_dx_{Nx}_{Ny}_vforce.npy')
            hforce = np.load(f'{in_dir}A_{alpha_i}_RE_{Re}_dx_{Nx}_{Ny}_hforce.npy')
    
            vforce = (vforce*2)/(density*V0*V0*D)  #Cl
            hforce = (hforce*2)/(density*V0*V0*D)  #Cd

            ax.plot(hforce[int(len(hforce)*(2/4)):], vforce[int(len(hforce)*(2/4)):], 'b-.', label=r'$\alpha$'+f' = {alpha_i}')   # s o -x
        except:
            pass

        #LITERATURE DATA
        try:
            Cl_kang = np.load(f'./reference/A_{alpha_i}_RE_{Re_ref}_Cl_Kang1999.npy')
            Cd_kang = np.load(f'./reference/A_{alpha_i}_RE_{Re_ref}_Cd_Kang1999.npy')

            ax.plot(Cd_kang, Cl_kang, 'k--', label=r'$\alpha$'+f' = {alpha_i} (Kang 1999)')
            #aquest tringles, o cercles --^    o --+
        except:
            pass


    #ax.set_ylim([-6, 1])
    #ax.set_xlim([0.3, 1.4])

    ax.set(xlabel='Cd [·]', ylabel='Cl [·]', title=f'Phase Diagram [ Re={Re}, N=[{Nx}x{Ny}] ]')
    ax.legend()
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
    ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

    fig.savefig(f"{out_dir}RE_{Re}_dx_{Nx}_{Ny}_PhaseDiagram.png")
    plt.close()



#aquest en funcio de alfa, per cada re 

# def meta_Mean_Cl_Cd(in_dir, out_dir, alpha, Re, Nx, Ny, Nt, D):
    
#     t_null_vel = int(Nt*(2/4))
#     density=1
#     V0=1
    

#     fig1, ax1 = plt.subplots()
#     fig2, ax2 = plt.subplots()

#     for i, alpha_i in enumerate(alpha):
#         vforce = np.load(f'{in_dir}A_{alpha_i}_RE_{Re}_dx_{Nx}_{Ny}_vforce.npy')
#         hforce = np.load(f'{in_dir}A_{alpha_i}_RE_{Re}_dx_{Nx}_{Ny}_hforce.npy')
    
#         vforce = (vforce*2)/(density*V0*V0*D) #Cl
#         hforce = (hforce*2)/(density*V0*V0*D) #Cd

#         ax1.plot(, vforce[t_null_vel:], label=r'Re'+f' = {alpha_i}')
#         ax1.plot(, hforce[t_null_vel:], label=r'Re'+f' = {alpha_i}')
    
    
#     #LITERATURE DATA




#     ax.set(xlabel='alpha', ylabel='Cl [·]', title=f'Mean Cl [ Re={Re}, N=[{Nx}x{Ny}] ]')
#     ax.legend()
#     ax.xaxis.set_minor_locator(AutoMinorLocator(4))
#     ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#     ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
#     ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

#     fig.savefig(f"{out_dir}RE_{Re}_dx_{Nx}_{Ny}_MeanCl.png")
#     plt.close()




def meta_Strouhal(config):

    for i, ite in enumerate(config['ITERABLE']):
        if ite[0] == 'Re':
            RE = ite[1]
            RE = [float(x) for x in RE]
        if ite[0] == 'Nx':
            NX = ite[1]
        #if ite[0] == 'Alpha':
        #    A = ite[1]


    St_t=np.zeros(len(RE))
    St_n=np.zeros((len(RE),len(NX)))
    St_nfull=np.zeros((len(RE),len(NX)))
    dF=np.zeros((len(RE),len(NX)))
    dFfull=np.zeros((len(RE),len(NX)))
    

    Lx=config['Lx']
    Ly=config['Ly'] 

    Nt=config['Nt']
        
    D=config['D']
    alpha=0

    out_dir=config['out_dir']
    in_dir=config['in_dir']


    for i, Re in enumerate(RE):
        for j, Nx in enumerate(NX):
            Ny=Nx
            dx=Lx/Nx
            dy=Ly/Ny
            dt=dx

            velocity_probe = np.load(f'{in_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_velocity_probe.npy')

            t, t_null_vel, t_upsampled, envelope, deri = detect_stationary(velocity_probe, Nt, dt)

            N = Nt-t_null_vel #600 # Number of sample points  Nt     vs Sample_rate  frequency is sample points in each second
            T = dt #1.0 / 800.0 # sample spacing  DT=1/sample rate

            t = np.linspace(0.0, N*T, N, endpoint=False)
            sample_rate=1/dt
            dF[i,j]=sample_rate/N #frequency ressolution
            
            yf = fft(velocity_probe[t_null_vel:]) 
            yf = 2.0/N * np.abs(yf[0:N//2])
            
            xf = fftfreq(N, T)[:N//2]  #sample points,duration   . l'utlima part sols per agafar positives

            f=abs(xf[np.argmax(np.abs(yf[0:N//2]))]) #Frequency corresponding with maximum module

            #not chopped
            yffull = fft(velocity_probe)
            yffull = 2.0/Nt * np.abs(yffull[0:Nt//2])

            xffull = fftfreq(Nt, T)[:Nt//2] 

            ffull=abs(xf[np.argmax(np.abs(yffull[0:Nt//2]))]) 
            dFfull[i,j]=sample_rate/Nt


            St_n[i,j]=f*D
            St_nfull[i,j]=ffull*D
            St_t[i]=0.198*(1-(19.7/Re))


    #PLOT ERROR STROUHAL
    fig, ax = plt.subplots()

    ax.plot(RE, St_t, 'C0', label='St theoric')
    ax.plot(RE, St_t, 'C0x')
    for j, Nx in enumerate(NX):
        Ny=Nx
        dx=Lx/Nx
        dy=Ly/Ny
        dt=dx

        ax.plot(RE, St_n[:,j], f'C{j+1}', label=f'$St_n$ [{Nx}x{Ny}] Stationary')
        ax.plot(RE, St_n[:,j], f'C{j+1}x')

        ax.plot(RE, St_nfull[:,j], 'kx', label=f'$St_n$ [{Nx}x{Ny}] Full')

        #ERROR INTERVAL STATIONARY
        # end points of errors
        xp = RE 
        yp = St_n[:,j] - dF[:,j]*D
        yp = np.squeeze(yp)
        xn = RE 
        yn = St_n[:,j] + dF[:,j]*D
        yn = np.squeeze(yn)

        vertices = np.array([ np.concatenate([np.concatenate([xp, xn[::-1]], 0), [xp[0]]],0 ),
                            np.concatenate([np.concatenate([yp, yn[::-1]], 0), [yp[0]]],0) ]).T

        codes = Path.LINETO * np.ones(len(vertices), dtype=Path.code_type)
        codes[0] = codes[len(xp)+len(xn)] = Path.MOVETO

        path = Path(vertices, codes)
        patch = PathPatch(path, facecolor=f'C{j+1}', edgecolor='none', alpha=0.3)
        ax.add_patch(patch)


        #ERROR INTERVAL FULL
        # end points of errors
        xp = RE 
        yp = St_nfull[:,j] - dFfull[:,j]*D
        yp = np.squeeze(yp)
        xn = RE 
        yn = St_nfull[:,j] + dFfull[:,j]*D
        yn = np.squeeze(yn)

        vertices = np.array([ np.concatenate([np.concatenate([xp, xn[::-1]], 0), [xp[0]]],0 ),
                            np.concatenate([np.concatenate([yp, yn[::-1]], 0), [yp[0]]],0) ]).T

        codes = Path.LINETO * np.ones(len(vertices), dtype=Path.code_type)
        codes[0] = codes[len(xp)+len(xn)] = Path.MOVETO

        path = Path(vertices, codes)
        patch = PathPatch(path, facecolor='k', edgecolor='none', alpha=0.15)
        ax.add_patch(patch)


    ax.set(xlabel='Reynolds', ylabel='Strouhal', title='Theoretical vs. Numerical')


    ax.xaxis.set_major_locator(MultipleLocator(100))
    #ax.yaxis.set_major_locator(MultipleLocator(0.2))     
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid()
    ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

    ax.legend()
    fig.savefig(f"{out_dir}_error_Strouhal.png")

def meta_forces(config):
    for i, ite in enumerate(config['ITERABLE']):
        if ite[0] == 'Re':
            RE = ite[1]
            RE = [float(x) for x in RE]
        if ite[0] == 'Nx':
            NX = ite[1]
        if ite[0] == 'A':
            A = ite[1]

    Lx=config['Lx']
    Ly=config['Ly'] 

    Nt=config['Nt']
        
    D=config['D']
    alpha=0

    out_dir=config['out_dir']
    in_dir=config['in_dir']


    for i, Re in enumerate(RE):
        for j, Nx in enumerate(NX):

            fig, ax = plt.subplots()

            for z, alpha in enumerate(A):
                pass
                #cal forces
