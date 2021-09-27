from interface.terminal.cmd_info import error

import pdb
import inspect
import csv, yaml
import glob, os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from csv import writer

from analysis.mesure import *


#TODO: see if moves to mesure,?? 
def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

def divergence_aux(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])

def precision_indicators(Cd,Cl,save_field_x_ite, noise=True): 
    if noise:
        Cd_max = np.max(Cd[int(len(Cd)*(3/4))::save_field_x_ite])
        Cd_min = np.min(Cd[int(len(Cd)*(3/4))::save_field_x_ite])

        Cl_max = np.max(Cl[int(len(Cl)*(3/4))::save_field_x_ite])
        Cl_min = np.min(Cl[int(len(Cl)*(3/4))::save_field_x_ite])
    else:
        Cd_max = np.max(Cd)
        Cd_min = np.min(Cd)

        Cl_max = np.max(Cl)
        Cl_min = np.min(Cl)

    Cl = zero_to_nan(Cl)
    Cd = zero_to_nan(Cd)

    Cl_mean = np.nanmean(Cl[save_field_x_ite:])
    Cd_mean = np.nanmean(Cd[save_field_x_ite:])

    DCd = (Cd_max-Cd_min)
    DCl = (Cl_max-Cl_min)
    center = [Cd_min + DCd/2 , Cl_min + DCl/2]

    return Cd_max, Cd_min, Cd_mean, Cl_max, Cl_min, Cl_mean, center, DCd, DCl

def calculate_coeficients(in_dir, alpha, Re, Nx, Ny, Nt, dx, dy, dt, Lx, Ly, xD, D, distance=0.2):
    
    DOMAIN = Domain(x=Nx, y=Ny, boundaries=OPEN, bounds=Box[0:Lx, 0:Ly])
    CYLINDER_2 = HardGeometryMask(Sphere([xD, Ly/2], radius=D/2 + distance )) >> DOMAIN.scalar_grid()

    p_field = np.load(f'{in_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_pressure_field.npy')
    ite = np.load(f'{in_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_iteration_field.npy')

    vforce=np.zeros(Nt)
    hforce=np.zeros(Nt)
    
    for z, itera in enumerate(ite):
        hforce[itera], vforce[itera] = calculate_forces(p_field[z,0], CYLINDER_2, dx, dy)

    t, t_null_vel, _, _, _ = detect_stationary(vforce, Nt, dt)

    if t_null_vel <= Nt/3:
        t_null_vel = int(Nt/2)

    #PASS FORCE TO COEFICIENT
    density=1
    V0=1
    Cl = (vforce*2)/(density*V0*V0*D)
    Cd = (hforce*2)/(density*V0*V0*D) 

    return t, t_null_vel, Cl, Cd

def calculate_performance_coeficients(in_dir, alpha, Re, Nx, Ny):
    with open(f'{in_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_performance_results.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        poisson_time = 0
        ite_time = 0
        unet_inference_time = 0
        cg_inference_time = 0
        for row in csv_reader:
            if row[1] == "run_init":
                    time_init = float(row[2])
            if row[1] == "run_end":
                    total_time = float(row[2]) - time_init 

            if "init_iteration_" in row[1]:
                    ite_init = float(row[2])
            if "_>init_post" in row[1]:
                    ite_time = ite_time + (float(row[2]) - ite_init)

            if "init_poisson__" in row[1]:
                    poisson_init = float(row[2])
            if "end_poisson__" in row[1]:
                    poisson_time = poisson_time + (float(row[2])-poisson_init)

            if "_>UNET_inference_interval" in row[1]:
                    unet_inference_time += float(row[3])

            if "_>CG_inference_interval" in row[1]:
                    cg_inference_time += float(row[3])
    
    return total_time, ite_time, poisson_time, unet_inference_time, cg_inference_time


####################

#TODO: las fuenciones que dependen de VK, poner dentro de sim no aqui general OSINO, EN ANALYSIS_VK

def divergence_time(in_dir, out_dir, meta_out_dir, ax, fig, last_ite, dir_name, alpha, Re, Nx, Ny, y_lim=[], meta=False, meta_within_case=False, meta_within_alpha=False):
    
    if not meta and not meta_within_case and not meta_within_alpha:
        fig, ax = plt.subplots()

    #1.Import the velocity fields
    u_field = np.load(f'{in_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_velocity_x_field.npy')
    v_field = np.load(f'{in_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_velocity_y_field.npy')
    ite = np.load(f'{in_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_iteration_field.npy')

    #2.Calculate the divergence field
    dive = []
    for j, _ in enumerate(ite):
        dive.append( np.sum( divergence_aux([u_field[j,0],v_field[j,0]]) ) / (Nx*Ny) )

    #3.Plot
    saux = r"$\alpha$="
    newline = "\n"
    if meta:
        mlabel = f'{dir_name} {saux}{alpha}'
    elif meta_within_alpha:
        mlabel = f'{dir_name}'
    elif meta_within_case:
        mlabel = f'{saux}{alpha}'
    else:
        mlabel = []


    ax.plot(ite[1:], dive[1:], '--o', label=mlabel if mlabel else "")

    if last_ite: # or (not meta and not meta_within_case and not meta_within_alpha):
        ax.set(xlabel='iteration', ylabel='Average Divergence', title=f'{f"{dir_name + newline}" if not meta and not meta_within_alpha else ""}' + f'[' + f'{ f"Alpha={alpha}, " if not meta and not meta_within_case else ""}' +f'Re={Re}, N=[{Nx}x{Ny}] ]\n')
        
        if mlabel:
            ax.legend()

        if not y_lim:
            pass
        else:
            ax.set_ylim([y_lim[0], y_lim[1]])

        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
        ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

        fig.savefig(f"{meta_out_dir if (meta or meta_within_alpha) else out_dir}A_{alpha if not meta_within_case and not meta else 'XX'}_RE_{Re}_dx_{Nx}_{Ny}_Divergence_Average_Residual.png")
        plt.close()

def divergence_timestep(in_dir, out_dir, dir_name, alpha, Re, Nx, Ny, Lx, Ly, xD, D, dx, dy, time_step=20, zoom=True):

    zoom_position=[xD-D,xD+D,Ly/2-D,Ly/2+D]

    if zoom:
        [x1,x2,y1,y2] = zoom_position
        x1 = int(x1/dx)
        x2 = int(x2/dx)
        y1 = int(y1/dy)
        y2 = int(y2/dy)
    else:
        x = np.arange(0, Lx +dx/2, dx) 
        y = np.arange(0, Ly +dy/2, dy)

        x1 = 0
        x2 = len(x)
        y1 = 0
        y2 = len(y)

    u_field = np.load(f'{in_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_velocity_x_field.npy')
    v_field = np.load(f'{in_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_velocity_y_field.npy')
   
    fig, ax = plt.subplots(nrows=1, figsize=(6, 5.4))

    im = ax.imshow( divergence_aux([u_field[time_step,0][x1:x2,y1:y2],v_field[time_step,0][x1:x2,y1:y2]]) , vmin= -np.max(np.abs( divergence_aux([u_field[time_step,0][x1:x2,y1:y2],v_field[time_step,0][x1:x2,y1:y2]])  )), vmax= np.max(np.abs( divergence_aux([u_field[time_step,0][x1:x2,y1:y2],v_field[time_step,0][x1:x2,y1:y2]])   )),
    interpolation='bilinear', cmap =plt.get_cmap('seismic'))

    saux = r"$\alpha$"
    newline = "\n"
    ax.set_title(f'{dir_name+newline}Divergence @ iteration={time_step} [{saux}={alpha} Re={Re}, N=[{Nx}x{Ny}] ]')
    cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax)
    cbar.set_label('divergence')

    fig.savefig(f"{out_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_Divergence_snap.png")

def forces_time(in_dir, out_dir, meta_out_dir, ax, fig, last_ite, save_field_x_ite, dir_name, alpha, Re, Nx, Ny, Nt, dx, dy, dt, Lx, Ly, xD, D, y_lim=[], distance=0.2, meta=False, meta_within_case=False, meta_within_alpha=False):
    
    if not meta and not meta_within_case and not meta_within_alpha:
        fig, ax = plt.subplots()

    #1.Calculate the forces coeficients
    t, t_null_vel, Cl, Cd = calculate_coeficients(in_dir, alpha, Re, Nx, Ny, Nt, dx, dy, dt, Lx, Ly, xD, D, distance) 

    #2.1.Plot the coeficients over time
    saux = r"$\alpha$="
    newline = "\n"
    if meta:
        mlabel = f'{dir_name} {saux}{alpha}'
    elif meta_within_alpha:
        mlabel = f'{dir_name}'
    elif meta_within_case:
        mlabel = f'{saux}{alpha}'
    else:
        mlabel = []

    ax.plot(t[::save_field_x_ite], Cl[::save_field_x_ite], label='Cl ' + mlabel if mlabel else "Cl")
    ax.plot(t[::save_field_x_ite], Cd[::save_field_x_ite], label='Cd ' + mlabel if mlabel else "Cd")

    #2.2.Plot additional information
    #ax.plot([t[t_null_vel], t[t_null_vel] ],[np.max([np.max(Cl[t_null_vel:]), np.max(Cd[t_null_vel:])] )*1.1, 
    #             np.min( [ np.min(Cl[t_null_vel:]), np.min(Cd[t_null_vel:]) ] )*1.1],'r--', label="Stationary Regime")

    #2.3.Figure labels and saving
    if last_ite:
        ax.set(xlabel='time [s]', ylabel='Coefficient', title=f'{f"{dir_name + newline}" if not meta and not meta_within_alpha else ""}' + f'Force probe [' + f'{ f"Alpha={alpha}, " if not meta and not meta_within_case else ""}' +f'Re={Re}, N=[{Nx}x{Ny}] ]')
        ax.legend()            
        #ax.set_ylim([ np.min( [ np.min(Cl[t_null_vel::save_field_x_ite]), np.min(Cd[t_null_vel::save_field_x_ite]) ] )*1.1 , np.max([np.max(Cl[t_null_vel::save_field_x_ite]), np.max(Cd[t_null_vel::save_field_x_ite])] )*1.1 ])

        if not y_lim:
            pass
        else:
            ax.set_ylim([y_lim[0], y_lim[1]])

        ax.xaxis.set_major_locator(MultipleLocator((Nt*dt)/10))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
        ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)
      
        fig.savefig(f"{meta_out_dir if (meta or meta_within_alpha) else out_dir}A_{alpha if not meta_within_case and not meta else 'XX'}_RE_{Re}_dx_{Nx}_{Ny}_d_{distance}_Forces_time.png")
        plt.close()
   
def forces_phase_diagram(in_dir, in_ref, out_dir, meta_out_dir, ax, fig, last_ite, save_field_x_ite, dir_name, alpha, Re, Nx, Ny, Nt, dx, dy, dt, Lx, Ly,xD, D, distance=0.2, precision=False, meta=False, meta_within_case=False, meta_within_alpha=False):

    if not meta and not meta_within_case and not meta_within_alpha:
        fig, ax = plt.subplots()

    _, _, Cl, Cd = calculate_coeficients(in_dir, alpha, Re, Nx, Ny, Nt, dx, dy, dt, Lx, Ly, xD, D, distance)                   
    

    #3.Plot
    saux = r"$\alpha$="
    newline = "\n"

    if meta:
        mlabel = f'{dir_name} {saux}{alpha}'
    elif meta_within_alpha:
        mlabel = f'{dir_name}'
    elif meta_within_case:
        mlabel = f'{saux}{alpha}'
    else:
        mlabel = []

    ax.plot(Cd[int(len(Cd)*(3/4))::save_field_x_ite], Cl[int(len(Cl)*(3/4))::save_field_x_ite], 'x-', label=mlabel if mlabel else "")
    
    if precision:
        Cd_max, Cd_min, Cd_mean, Cl_max, Cl_min, Cl_mean, center, DCd, DCl = precision_indicators(Cd,Cl,save_field_x_ite)
        ax.plot([Cd_min, Cd_max, Cd_max, Cd_min, Cd_min],[Cl_min, Cl_min, Cl_max, Cl_max, Cl_min],'--r')
        ax.plot(center[0], center[1], 'ro')
        ax.plot(Cd_mean, Cl_mean, 'go')
        ax.text(center[0]+0.01, center[1]+0.01, f'C=[{np.round(center[0],2)},{np.round(center[1],2)}]\n'+r'$\Delta Cd$ ' + f'={np.round(DCd,2)}\n'+r'$\Delta Cl$'+f'={np.round(DCl,2)}')
                        
    #print(f'{dir_name} -> C={np.round(center,2)} , DCd={np.round(DCd,2)} , DCl={np.round(DCl,2)}')
    
    if in_ref:
        Cl_kang = np.load(f'{in_ref}A_{alpha}_RE_{Re}_Cl_Kang1999.npy')
        Cd_kang = np.load(f'{in_ref}A_{alpha}_RE_{Re}_Cd_Kang1999.npy')

        ax.plot(Cd_kang, Cl_kang, 'k--', label=f'(Kang 1999) {saux}{alpha}')

        if precision:
            Cd_max, Cd_min, Cd_mean, Cl_max, Cl_min, Cl_mean, center, DCd, DCl = precision_indicators(Cd_kang,Cl_kang,1, noise=False)

            ax.plot([Cd_min, Cd_max, Cd_max, Cd_min, Cd_min],[Cl_min, Cl_min, Cl_max, Cl_max, Cl_min],'--k')
            ax.plot(center[0], center[1], 'ko')
            ax.plot(Cd_mean, Cl_mean, 'go')
            ax.text(center[0]+0.01, center[1]+0.01, f'C=[{np.round(center[0],2)},{np.round(center[1],2)}]\n'+r'$\Delta Cd$ ' + f'={np.round(DCd,2)}\n'+r'$\Delta Cl$'+f'={np.round(DCl,2)}')


    if last_ite:
        ax.set(xlabel='Cd [·]', ylabel='Cl [·]', title=f'{f"{dir_name + newline}" if not meta and not meta_within_alpha else ""}' + f'Phase Diagram [' + f'{ f"Alpha={alpha}, " if not meta and not meta_within_case else ""}' +f'Re={Re}, N=[{Nx}x{Ny}] ]')
        
        if mlabel:
            ax.legend()
        

        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
        ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

        if not precision:
            fig.savefig(f"{meta_out_dir if (meta or meta_within_alpha) else out_dir}A_{alpha if not meta_within_case and not meta else 'XX'}_RE_{Re}_dx_{Nx}_{Ny}_d_{distance}_Forces_ss.png")
        else:
            fig.savefig(f"{meta_out_dir if (meta or meta_within_alpha) else out_dir}A_{alpha if not meta_within_case and not meta else 'XX'}_RE_{Re}_dx_{Nx}_{Ny}_d_{distance}_Precision_ss.png")
        
        plt.close()


def evaluate_precision(in_dir, in_ref, out_dir, meta_out_dir, ax, fig, last_ite, save_field_x_ite, dir_name, alpha, Re, Nx, Ny, Nt, dx, dy, dt, Lx, Ly,xD, D, meta=False, meta_within_case=False, meta_within_alpha=False):
    
    if not meta and not meta_within_case and not meta_within_alpha:
        fig, ax = plt.subplots()


    #1.Calculate precision coeficients
    #Reference values
    if 'reference' in in_ref:
        Cl = np.load(f'{in_ref}A_{alpha}_RE_{Re}_Cl_Kang1999.npy')
        Cd = np.load(f'{in_ref}A_{alpha}_RE_{Re}_Cd_Kang1999.npy')
        _, _, _, _, _, _, center_ref, DCd_ref, DCl_ref = precision_indicators(Cd,Cl,save_field_x_ite, noise=False)
    else:
        _, _, Cl, Cd = calculate_coeficients(in_ref, alpha, Re, Nx, Ny, Nt, dx, dy, dt, Lx, Ly,xD, D, distance=0.2)
        _, _, _, _, _, _, center_ref, DCd_ref, DCl_ref = precision_indicators(Cd,Cl,save_field_x_ite, noise=True)
    
    #Compare vlues
    _, _, Cl, Cd = calculate_coeficients(in_dir, alpha, Re, Nx, Ny, Nt, dx, dy, dt, Lx, Ly,xD, D, distance=0.2)
    _, _, _, _, _, _, center, DCd, DCl = precision_indicators(Cd,Cl,save_field_x_ite, noise=True)


    #2.Calculate error with respect reference
    error_center = np.round(0.5*( 100*((center[0] - center_ref[0])/center_ref[0]) + 100*((center[1] - center_ref[1])/center_ref[1]) ),2)
    error_box = np.round(0.5*( 100*((DCl - DCl_ref)/DCl_ref) + 100*((DCd - DCd_ref)/DCd_ref) ),2)


    #3.FILE save
    with open(f"{meta_out_dir if (meta or meta_within_alpha) else out_dir}A_{alpha if not meta_within_case and not meta else 'XX'}_RE_{Re}_dx_{Nx}_{Ny}_Precision_Evaluation.csv", 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow([dir_name, alpha, np.round(center,2), np.round(DCd,2), np.round(DCl,2), np.round(center_ref,2), np.round(DCd_ref,2), np.round(DCl_ref,2),        
        np.round((center[0] - center_ref[0])/center_ref[0],2), np.round((center[1] - center_ref[1])/center_ref[1],2), error_center, np.round((DCl - DCl_ref)/DCl_ref,2), np.round((DCd - DCd_ref)/DCd_ref,2), error_box])
        
        if last_ite:
            writer_object.writerow(['dir_name', 'alpha', 'center', 'DCd', 'DCl', 'center_ref', 'DCd_ref', 'DCl_ref',           
            'e_center_x', 'e_center_y', 'e_center_100%', 'e_box_Cl', 'e_box_Cd', 'e_box_100%'])
        
        f_object.close()



    #3.Plot
    saux = r"$\alpha$="
    saux2 = r"$\epsilon_{center}$"
    saux3 = r"$\epsilon_{box}$"
    newline = "\n"
    if meta:
        mlabel_center = f'{saux2}: {dir_name}'
        mlabel_box = f'{saux3}: {dir_name}'
    else :
        mlabel_center = f'{saux2}'
        mlabel_box = f'{saux3}'

    if last_ite:
        if not meta_within_alpha:
            ax.plot(float(alpha), error_center, 'o', label=mlabel_center)
            ax.plot(float(alpha), error_box, 'x', label=mlabel_box)
        else:
            ax.plot(dir_name, error_center, 'o', label=mlabel_center)
            ax.plot(dir_name, error_box, 'x', label=mlabel_box)
    else:
        if not meta_within_alpha:
            ax.plot(float(alpha), error_center, 'o')
            ax.plot(float(alpha), error_box, 'x')
        else:
            ax.plot(dir_name, error_center, 'o')
            ax.plot(dir_name, error_box, 'x')

    if last_ite: 
        ax.set(xlabel='alpha', ylabel='relative error', title=f'{f"{dir_name + newline}" if not meta and not meta_within_alpha else ""}' + f'[' + f'{ f"Alpha={alpha}, " if not meta and not meta_within_case else ""}' +f'Re={Re}, N=[{Nx}x{Ny}] ]\n')
        
        if mlabel_center and mlabel_box:
            ax.legend()

        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
        ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

        fig.savefig(f"{meta_out_dir if (meta or meta_within_alpha) else out_dir}A_{alpha if not meta_within_case and not meta else 'XX'}_RE_{Re}_dx_{Nx}_{Ny}_Precision_Evaluation.png")
        plt.close()

def evaluate_performance(in_dir, in_ref, out_dir, meta_out_dir, last_ite, dir_name, alpha, Re, Nx, Ny, Nt, meta=False, meta_within_case=False, meta_within_alpha=False):

    total_time, ite_time, poisson_time, unet_inference_time, cg_inference_time = calculate_performance_coeficients(in_dir, alpha, Re, Nx, Ny)
    total_time_ref, ite_time_ref, poisson_time_ref, unet_inference_time_ref, cg_inference_time_ref = calculate_performance_coeficients(in_ref, alpha, Re, Nx, Ny)

    with open(f"{meta_out_dir if (meta or meta_within_alpha) else out_dir}A_{alpha if not meta_within_case and not meta else 'XX'}_RE_{Re}_dx_{Nx}_{Ny}_Performance_Evaluation.csv", 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow([dir_name, alpha, np.round(total_time/(1000*60),2), 
            np.round(ite_time/(1000*60),2), np.round((ite_time/Nt)/(1000),2), np.round((ite_time/total_time)*100,1), 
            np.round(poisson_time/(1000*60),2), np.round((poisson_time/Nt)/(1000),2), np.round((poisson_time/ite_time)*100,1),
            np.round(unet_inference_time/(1000*60),2), np.round((unet_inference_time/Nt)/(1000),2), np.round((unet_inference_time/poisson_time)*100,1),
            np.round(cg_inference_time/(1000*60),2), np.round((cg_inference_time/Nt)/(1000),2), np.round((cg_inference_time/poisson_time)*100,1), 
            np.round((((total_time-total_time_ref)/total_time_ref)*100),2), 
            np.round( (unet_inference_time - cg_inference_time_ref)*(100/cg_inference_time_ref) ,2)
            
            ])  
    
        if last_ite:
            writer_object.writerow(['case_name', 'alpha', 'Total Time [min]', 
                'Iteration Time Total [min]', 'Iteration Time [s/ite]', f'Iteration Time % of total sim',
                'Poisson Time Total [min]', 'Poisson Time [s/ite]', f'Poisson Time % of inference',
                'UNET Inference Time Total [min]', 'UNET Inference Time [s/ite]', f'UNET Inference Time % of poisson sim',
                'CG Inference Time Total [min]', 'CG Inference Time [s/ite]', f'CG Inference Time % of poisson sim',
                'Total relative time %', 'Inference relative % time'
                
                ])
        
        f_object.close()

 #vel_probe

#plot_spectrum
#make_gif    #make_step
#CP, pressure, calculate_vorticity, normal_vel

######################################################
def meta_analysis(in_dir: list, function: list, *function_args, in_ref: list = [], config_dir: list = [], out_dir:list = [], meta_out_dir = [], meta=False, meta_within_case=False, meta_within_alpha=False, **function_kwargs):

    #
    if not config_dir:
        config_dir = in_dir
        config_dir_shared = False
    else:
        if not len(config_dir) == len(in_dir):
            if len(config_dir) == 1:
                config_dir_shared = True 
            else:
                print('ERROR: not possible to assign this amount of config_dirs')
                error()
        else:
            config_dir_shared = False
    

    if not out_dir:
        out_dir = in_dir
    else:
        if not len(out_dir) == len(in_dir):
            if len(out_dir) == 1:
                out_dir = out_dir * len(in_dir) 
            else:
                print('ERROR: not possible to assign this amount of out_dirs')
                error()

    if not meta_out_dir:
        meta_out_dir = './'

    #
    if config_dir_shared:
        try:
            with open(config_dir[0]+'config_simulation.yaml','r') as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
        except:
            print('No yaml file founded.')
            error()


    #Create meta figures
    if meta:
        FIG = []
        AX = []
        for _ in range(len(function)):
            fig, ax = plt.subplots() 
            FIG.append(fig)
            AX.append(ax)

    elif meta_within_alpha:
        FIG = [[] for _ in range(len(function))]
        AX = [[] for _ in range(len(function))]
        

        max_num_alphas = 0
        for dir in in_dir:
            max_num_alphas_in_dir = len(glob.glob(f'{dir}A_*_velocity_x_field.npy'))
            if max_num_alphas_in_dir > max_num_alphas:
                max_num_alphas = max_num_alphas_in_dir

        for f in range(len(function)):
            for _ in range(max_num_alphas):
                fig, ax = plt.subplots() 
                FIG[f].append(fig)
                AX[f].append(ax)

    #
    for i, _ in enumerate(in_dir):
        if meta_within_case:
            FIG = []
            AX = []
            for _ in range(len(function)):
                fig, ax = plt.subplots() 
                FIG.append(fig)
                AX.append(ax)

        #Get the name of the subcase (type of unet, particular things, etc)
        dir_name = in_dir[i].split('/')[1] #Because / is the last caracter so it makes and extra ""

        #Get the constant variables (Nt, CFL, etc)
        if not config_dir_shared:
            try:
                with open(config_dir[0]+"config_simulation.yaml",'r') as config_file:
                    config = yaml.load(config_file, Loader=yaml.FullLoader)
            except:
                print('No yaml file founded.')
                error()

        #Extract constant variables
        Lx = config['Lx']
        Ly = config['Ly']
        xD = config['xD']
        D = config['D']
        Nt = config['Nt']
        CFL = config['CFL']

        try:
            save_field_x_ite = int(float(config['save_field_x_ite']))
        except:
            save_field_x_ite = 200

        #Try all possible non constant variables inside the folder
        count_alphas = 0
        for file in glob.glob(f'{in_dir[i]}A_*_velocity_x_field.npy'):  #TODO: what if other files??
            #Detect if last iteration
            count_alphas = count_alphas + 1

            if meta:
                if count_alphas == len(glob.glob(f'{in_dir[i]}A_*_velocity_x_field.npy')) and i == len(in_dir)-1:
                    last_ite = True
                else:
                    last_ite = False
            elif meta_within_alpha:
                if i == len(in_dir)-1:
                    last_ite = True
                else:
                    last_ite = False
            elif meta_within_case:
                if count_alphas == len(glob.glob(f'{in_dir[i]}A_*_velocity_x_field.npy')):
                    last_ite = True
                else:
                    last_ite = False
            else:
                last_ite = True

            #Extract variable parameters
            alpha = float(file.split('/')[-1].split('_')[1])
            Re = float(file.split('/')[-1].split('_')[3])
            Nx = int(file.split('/')[-1].split('_')[5])
            Ny = int(file.split('/')[-1].split('_')[6])         
            #alpha = float(file.split('\\')[1].split('_')[1])
            #Re = float(file.split('\\')[1].split('_')[3])
            #Nx = int(file.split('\\')[1].split('_')[5])
            #Ny = int(file.split('\\')[1].split('_')[6])

            
            #Calculate variable parameters
            dx = Lx/Nx
            dy = Ly/Ny
            dt = dx * CFL

            #function calling
            for f, func in enumerate(function):
                
                print(f'EXECUTING: {func} on dir #{i+1} & alpha #{count_alphas}')

                # "inspect.Signature" instance encapsulating this callable's signature.
                func_sig = inspect.signature(func)

                # Human-readable name of this function for use in exceptions.
                func_name = func.__name__ 

                #Append the func arguments
                args_var = []
                kwargs_var = {}
                for func_arg_index, func_arg in enumerate(func_sig.parameters.values()):
                    
                    #TODO: make ike a pluggin dictionary or something

                    #Positional arguments
                    if str(func_arg) == 'in_dir':
                        args_var.append(in_dir[i])
                    if str(func_arg) == 'in_ref':
                        args_var.append(in_ref)
                    if str(func_arg) == 'out_dir':
                        args_var.append(out_dir[i])
                    if str(func_arg) == 'meta_out_dir':
                        args_var.append(meta_out_dir)
                    if str(func_arg) == 'dir_name':
                        args_var.append(dir_name)

                    if str(func_arg) == 'alpha':
                        args_var.append(alpha)
                    if str(func_arg) == 'Re':
                        args_var.append(Re)

                    if str(func_arg) == 'Nx':
                        args_var.append(Nx)
                    if str(func_arg) == 'Ny':
                        args_var.append(Ny) 
                    if str(func_arg) == 'Nt':
                        args_var.append(Nt)                   
                    if str(func_arg) == 'dx':
                        args_var.append(dx)
                    if str(func_arg) == 'dy':
                        args_var.append(dy)
                    if str(func_arg) == 'dt':
                        args_var.append(dt)

                    if str(func_arg) == 'Lx':
                        args_var.append(Lx)
                    if str(func_arg) == 'Ly':
                        args_var.append(Ly)

                    if str(func_arg) == 'xD':
                        args_var.append(xD)
                    if str(func_arg) == 'D':
                        args_var.append(D)
                    
                    
                    if str(func_arg) == 'ax':
                        args_var.append(AX[f][count_alphas-1] if meta_within_alpha else ( AX[f] if meta or meta_within_case else [] ) ) 
                    if str(func_arg) == 'fig':
                        args_var.append(FIG[f][count_alphas-1] if meta_within_alpha else ( FIG[f] if meta or meta_within_case else [] ) )
                    if str(func_arg) == 'save_field_x_ite':
                        args_var.append(save_field_x_ite)
                    if str(func_arg) == 'last_ite':
                        args_var.append(last_ite)                  

                    #Keyword arguments
                    #TODO: the same than in above, use plugin dict
                    if str(func_arg).split('=')[0] == 'meta'  if '=' in str(func_arg) else False:
                        kwargs_var['meta'] = meta
                    if str(func_arg).split('=')[0] == 'meta_within_case'  if '=' in str(func_arg) else False:
                        kwargs_var['meta_within_case'] = meta_within_case
                    if str(func_arg).split('=')[0] == 'meta_within_alpha'  if '=' in str(func_arg) else False:
                        kwargs_var['meta_within_alpha'] = meta_within_alpha
                
                #print(f'{f=} - {i+1=} - {count_alphas=} - {last_ite=} ')
                #append the rest of args that don't depend on meta_analysis
                for value in function_args:
                    args_var.append(value)
                for key, value in function_kwargs.items():
                    kwargs_var[key] = value
            
                #call the function
                func(*args_var, **kwargs_var)
