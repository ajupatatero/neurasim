import numpy as np

from scipy.signal import hilbert
from scipy import signal

from util.plot import plot_tools
from util.operations.field_operate import *


def detect_stationary(variable,Nt,dt):
    duration = Nt*dt #duration of the signal
    fs = Nt/duration #1/Dt #600.0 #sampling frequency
    t = np.linspace(0, Nt*dt, num=Nt) #
    #t = np.arange(int(fs*duration)) / fs #time base

    #print(f' initial sampling frequency: {fs}')

    #upsampling the data to avoid aliasing
    Ns=1
    variable_upsampled = signal.resample(variable, Nt*Ns)
    t_upsampled = np.linspace(0, Nt*dt, Nt*Ns)

    fs_upsampled = (Nt*Ns)/duration
    #print(f' upsampled sampling frequency: {fs_upsampled}')
        

    z = hilbert(variable_upsampled-np.mean(variable_upsampled)) #form the analytical signal    
    inst_amplitude = np.abs(z) #envelope extraction


    #filter
    b, a = signal.butter(3, 0.005)
    envelope = signal.filtfilt(b, a, inst_amplitude)


    #detect stationary
    deri=(np.diff(envelope)/np.diff(t))
    #deri = ( y[2:] - y[:-2] ) / ( t[2:] - t[:-2] )
    t_null_vel = np.where(abs(deri) <= 0.0000005)
    t_null_vel = np.squeeze(t_null_vel)

    try:
        t_null_vel = t_null_vel[0]
    except:
        t_null_vel = 0

    return t, t_null_vel, t_upsampled, envelope, deri

def compute_spectrum(velocity_probe,Nt,dt):
    
    t, t_null_vel, t_upsampled, envelope, deri = detect_stationary(velocity_probe, Nt, dt)

    N = Nt-t_null_vel #600 # Number of sample points  Nt     vs Sample_rate  frequency is sample points in each second
    T = dt #1.0 / 800.0 # sample spacing  DT=1/sample rate

    t = np.linspace(0.0, N*T, N, endpoint=False)
    sample_rate=1/dt
    dF=sample_rate/N #frequency ressolution
        
    yf = fft(velocity_probe[t_null_vel:]) 
    yf = 2.0/N * np.abs(yf[0:N//2])
        
    xf = fftfreq(N, T)[:N//2]  #sample points,duration   . l'utlima part sols per agafar positives

    f=abs(xf[np.argmax(np.abs(yf[0:N//2]))]) #Frequency corresponding with maximum module

    #not chopped
    yffull = fft(velocity_probe)
    yffull = 2.0/Nt * np.abs(yffull[0:Nt//2])

    xffull = fftfreq(Nt, T)[:Nt//2] 

    ffull=abs(xf[np.argmax(np.abs(yffull[0:Nt//2]))]) 
    dFfull=sample_rate/Nt

    return xf, yf, xffull, yffull, f, dF, ffull, dFfull

def calculate_St(velocity_probe,Nt,dt,D,Re):
    _, _, _, _, f, dF, ffull, dFfull = compute_spectrum(velocity_probe,Nt,dt)

    St_n=f*D
    St_nfull=ffull*D
    St_t=0.198*(1-(19.7/Re))

    return St_t, St_n, St_nfull

def get_reference_pressure(pressure):
    #pressure[x,y]
    pass

def calculate_cp(pressure, pressure_ref=1, rho_ref=1, vel_ref=1):
    return (pressure-pressure_ref)/(0.5*rho_ref*vel_ref*vel_ref)

def calculate_vorticity(Lx,Ly,dx,dy,field):
    #TODO: treure lx, etc del propi field si es grid 
    x = np.arange(0, Lx +dx/2, dx)  #Notice that we include +dx in order to consider the last step in arrange
    y = np.arange(0, Ly +dy/2, dy) 
    u = field.staggered_tensor().tensors[0]._native.cpu().numpy()[0]
    v = field.staggered_tensor().tensors[1]._native.cpu().numpy()[0]

    dFx_dy = np.transpose(np.gradient (np.transpose(u), y, axis=0))   #TODO: en realidad la x y y deberian ser del punto donde hay las velocidades no el centro!!!
    dFy_dx = np.gradient (v, x, axis=0)
    
    return dFy_dx - dFx_dy

def calculate_norm_velocity(field):
    u = field.staggered_tensor().tensors[0]._native.cpu().numpy()[0]
    v = field.staggered_tensor().tensors[1]._native.cpu().numpy()[0]
    return np.sqrt(u**2 + v**2)  #TODO: can be done without converting??

def calculate_forces(pressure, object_mask, dx, dy):
    '''Function to calculate the aerodynamic forces to with the input object is subjected to. 
    
    USAGE:

    NOTICE:
        -It is based on the pressure component only. Therefore, no viscous effect considered.
        -The forces will be given with respect the domain axis. i.e. Fx in x direction, and Fy in y direction. 
         And positive to the same direction as the axis (right and up).
    '''

    #TODO: get dx, dy from pressure if not numpy

    #1.Get the exteriors nodes of the input object
    [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_exterior_edges(object_mask)

    #2.Correct pressure field in case is phiflow type and not array
    if isinstance(pressure,StaggeredGrid):
        #Z0 = field.staggered_tensor().tensors[0]._native.cpu().numpy()
        #Z1 = field.staggered_tensor().tensors[1]._native.cpu().numpy()
        raise NotImplementedError

    elif isinstance(pressure,CenteredGrid):
        #pressure = pressure.values._native.cpu().numpy()
        pressure = pressure.values._native.cpu().numpy()[0]    #TODO: torch 

    #3.Calculate the respective forces
    Force_v = np.sum( pressure[edge_vb_x, edge_vb_y] - pressure[edge_vt_x, edge_vt_y] )*dx
    Force_h = np.sum( pressure[edge_hl_x, edge_hl_y] - pressure[edge_hr_x, edge_hr_y] )*dy

    return Force_h, Force_v

def calculate_forces_with_momentum(pressure, velocity, object_mask, factor=1, rho=1, dx=1, dy=1):
    '''Function to calculate the aerodynamic forces to with the input object is subjected to. 
    
    USAGE:

    NOTICE:
        -It is based on the momentum equation with only pressure and velocity components only. Therefore, no viscous effect considered.
        Nor inertial effects. i.e. quasistationary regime. If not, may have different results.
        -The forces will be given with respect the domain axis. i.e. Fx in x direction, and Fy in y direction. 
         And positive to the same direction as the axis (right and up).
        -The force result is given with the dimension correction so that it correspon to the force generated by the object length.
        Not the square of integration length.
    '''
    
    #TODO: get dx, dy from pressure if not numpy

    #1.Get the interiors nodes of the input object
    [ [edge_hl_x, _], [edge_hr_x, _], [_, edge_vb_y], [_, edge_vt_y] ] = get_exterior_edges(object_mask)
    [ [edge_hl_x, _], [edge_hr_x, _], [_, edge_vb_y], [_, edge_vt_y] ] = exterior_edge_to_interior_edge(edge_hl_x=edge_hl_x,
                                                                     edge_hr_x=edge_hr_x, edge_vb_y=edge_vb_y, edge_vt_y=edge_vt_y)

    #2.Correct pressure field in case is phiflow type and not array
    if isinstance(pressure,CenteredGrid):
        # p = pressure.values._native.cpu().numpy()
        p = pressure.values._native.cpu().numpy()[0]

    #3.Correct velocity field in case is phiflow type and not array
    if isinstance(velocity,StaggeredGrid):
        #u = velocity.staggered_tensor().tensors[0]._native.cpu().numpy()
        #v = velocity.staggered_tensor().tensors[1]._native.cpu().numpy()
        u = velocity.staggered_tensor().tensors[0]._native.cpu().numpy()[0]
        v = velocity.staggered_tensor().tensors[1]._native.cpu().numpy()[0]
    
    #4.Extract Square of integration and Diameter of object
    square = [min(edge_hl_x) - factor, max(edge_hr_x) + factor, min(edge_vb_y) - factor, max(edge_vt_y) + factor] #x1,x2,y1,y2
    
    lx = (square[1]-square[0])*dx
    ly = (square[3]-square[2])*dy

    Dx = lx - 2*factor*dx
    Dy = ly - 2*factor*dy
    
    #5.Calculate Momentum (velocity) component
    # ^ y
    # |
    # |  +--2--------+
    # | 3|  -----   4|
    # |  |  |   |    |ly
    # |  |  -----    |
    # |  +-1--lx-----+
    # .___________________> x

    u4 = u[ square[1], square[2]:square[3] ]
    u3 = u[ square[0], square[2]:square[3] ]
    v2 = v[ square[0]:square[1] , square[3] ]
    v1 = v[ square[0]:square[1] , square[2] ]

    Force_m_x = rho * np.sum( u4*u4 - u3*u3 )*dy
    Force_m_y = rho * np.sum( v2*v2 - v1*v1 )*dx

    #6.Calculate Pressure component
    p1 = p[ square[0]:square[1] , square[2] ]
    p2 = p[ square[0]:square[1] , square[3] ]
    p3 = p[ square[0], square[2]:square[3] ]
    p4 = p[ square[1], square[2]:square[3] ]

    Force_p_x = rho * np.sum( p4 - p3 )*dy
    Force_p_y = rho * np.sum( p2 - p1 )*dx

    #7.Calculate External forces:  
    # Fext + Fgrav + Fvisco - Fpres = Finerti + Fmomentum 
    #-> Fext = Fmoment + Fpres
    Force_h = Force_m_x + Force_p_x
    Force_v = Force_m_y + Force_p_y

    #8.Correct dimensions to acount for difference between lx and Dx
    Force_h = Force_h*(Dy/ly) 
    Force_v = Force_v*(Dx/lx) 

    return Force_h, Force_v

def get_induced_velocity(velocity, object_mask):
    U = to_numpy(velocity)

    # Correct if bsz = 1
    if U[0].shape[0] == 1:
        U[0] = U[0].squeeze()
        U[1] = U[1].squeeze()

    u = U[0]
    v = U[1]

    [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = get_exterior_edges(object_mask)
   
    line_x_sorted, line_y_sorted, angle = get_line_distribution(edges=[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ])


    fig, ax = plt.subplots()
    ax.set(xlabel='theta', ylabel='velocity', title='')
    ax.plot(angle, u[line_x_sorted,line_y_sorted], '--k')
    ax.plot(angle, u[line_x_sorted,line_y_sorted], 'ok', markersize=5)


    print(line_x_sorted)
    print(line_y_sorted)
    print(u[line_x_sorted,line_y_sorted])


    #Limits
    ax.set_ylim(-1.6,1.6) 

    #Grid
    # Change major ticks to show every x.
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))

    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
    ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

    fig.savefig('./u.png')
    plt.close()



    
    fig, ax = plt.subplots()
    ax.set(xlabel='theta', ylabel='velocity', title='')
    ax.plot(angle, v[line_x_sorted,line_y_sorted], '--k')
    ax.plot(angle, v[line_x_sorted,line_y_sorted], 'ok', markersize=5)


    print(line_x_sorted)
    print(line_y_sorted)
    print(u[line_x_sorted,line_y_sorted])


    #Limits
    ax.set_ylim(-1.6,1.6) 

    #Grid
    # Change major ticks to show every x.
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))

    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
    ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

    fig.savefig('./v.png')
    plt.close()



def probe():
    pass

    #fer class? per aixi tenir funcio de geometria per fer plot, stationary, etc
    
    #PLOT VELOCITY PROBE
    fig, ax = plt.subplots()

    duration = Nt*Dt #duration of the signal
    fs = Nt/duration #1/Dt #600.0 #sampling frequency
    t = np.linspace(0, Nt*Dt, num=Nt) #
    #t = np.arange(int(fs*duration)) / fs #time base

    #print(f' initial sampling frequency: {fs}')

    #upsampling the data to avoid aliasing
    N=1
    velocity_probe_upsampled = signal.resample(velocity_probe, Nt*N)
    t_upsampled = np.linspace(0, Nt*Dt, Nt*N)

    fs_upsampled = (Nt*N)/duration
    #print(f' upsampled sampling frequency: {fs_upsampled}')
        

    z = hilbert(velocity_probe_upsampled-np.mean(velocity_probe_upsampled)) #form the analytical signal    
    inst_amplitude = np.abs(z) #envelope extraction


    #filter
    b, a = signal.butter(3, 0.005)
    y = signal.filtfilt(b, a, inst_amplitude)


    #detect stationary
    deri=np.diff(y)/np.diff(t)
    t_null_vel = np.where(abs(deri) <= 0.00005)


    ax.plot(t_upsampled, y,'r', label="Filtered Envelope"); #overlay the extracted envelope
    ax.plot(t, velocity_probe, label="Velocity")
    ax.plot(t[:-1], deri,'g--',label="Envelope Derivate")
    ax.plot([t[t_null_vel[0][0]], t[t_null_vel[0][0]] ],[max(velocity_probe), min(velocity_probe)],'r--', label="Stationary Regime")

    ax.set(xlabel='time [s]', ylabel='vertical velocity', title=f'Velocity Probe [ Re={Re}, N=[{resX}x{resY}] ]')
    ax.legend()

    #GRID
    # Set axis ranges; by default this will put major ticks every 25.
    #ax.set_xlim(0, 200)
    #ax.set_ylim(0, 200)

    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
    ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)


    plt.show()
