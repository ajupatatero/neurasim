import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import BoundaryNorm
import matplotlib.image as mpimg

import imageio

from analysis.mesure import *
from util.operations import field_operate as fo
from engines.phi.field._grid import CenteredGrid, StaggeredGrid

import pdb
import sys


def plot_field(field,
                plot_type=['surface'],
                options=[],
                Lx=None, Ly=None, dx=None, dy=None,
                lx='x', ly='y', lbar='field', ltitle='Plot Field',
                save=False, filename='./field.png',
                fig=None, ax=None):

    '''Main function to plot vectorial and scalar fields (including masking fields).

        USAGE:
        -plots=[ [ 'plot_type', [ ['plot options' , xx], ['plot options' , xx] ]],
                 [ 'plot_type', [ ['plot options' , xx], ['plot options' , xx] ]]
               ]

        plot_type: contourn, mask, surface, streamlines
        general_plot_options:
            -edges -> [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ]
            -square -> [x1,x2,y1,y2]
            -circle -> [x,y,D]
            -aux_contourn -> True/False

            -limits -> [min, max]
            -vector_axis -> 0/1

            -indeces -> True/False
            -grid -> True/False

            -zoom_position -> [x1,x2,y1,y2]
            -full_zoom -> True/False
    '''

    #1.1.Staggered grid == Cells walls
    cx = np.arange(0, Lx +dx/2, dx)
    cy = np.arange(0, Ly +dy/2, dy)
    #NOTICE: we include +dx in order to consider the last step in arrange
    CY, CX = np.meshgrid(cy,cx)

    #1.2.Value grid -> Centered or Staggered in function of value type
    if isinstance(field,StaggeredGrid):
        x = cx
        y = cy
    elif isinstance(field,CenteredGrid):
        x = np.arange(dx/2, Lx-dx/2 +dx/2, dx)
        y = np.arange(dy/2, Ly-dy/2 +dy/2, dy)
    elif isinstance(field,(list, tuple, np.ndarray)) and len(field)==Lx/dx + 1:
        x = cx
        y = cy
    elif isinstance(field,(list, tuple, np.ndarray)):
        x = np.arange(dx/2, Lx-dx/2 +dx/2, dx)
        y = np.arange(dy/2, Ly-dy/2 +dy/2, dy)

    Y, X = np.meshgrid(y,x)


    #3.CORRECT INPUT FIELD TYPE to array
    Z = fo.to_torch(field)

    #4.CREATE FIGURE
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.set(xlabel=lx, ylabel=ly, title=ltitle)

    #5.CREATE DIFFERENT PLOTS
    if 'surface' in plot_type:
        #Get the properties passed
        for op in options:
            globals()[op[0]] = op[1]

        #Levels, Colormap & Normalization
        levels = MaxNLocator(nbins=25).tick_values(limits[0],limits[1])
        cmap = plt.get_cmap('seismic') #'viridis
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        #Region selection to plot
        if full_zoom if 'full_zoom' in globals() else False:
            [x1,x2,y1,y2] = zoom_position
            x1 = int(x1/dx)
            x2 = int(x2/dx)
            y1 = int(y1/dy)
            y2 = int(y2/dy)
        else:
            x1 = 0
            x2 = len(x)
            y1 = 0
            y2 = len(y)

        #sys.setrecursionlimit(5000)
        #Plot surface
        im = plt.pcolormesh(X[x1:x2,y1:y2],Y[x1:x2,y1:y2],Z[x1:x2,y1:y2], cmap=cmap, norm=norm, shading='auto')
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel(lbar)

        #Plot Contourn
        if aux_contourn if 'aux_contourn' in globals() else False:
            cs = ax.contour(X[x1:x2,y1:y2],Y[x1:x2,y1:y2],Z[x1:x2,y1:y2], colors='k', alpha=0.5, linestyles='dashed')
            ax.clabel(cs, colors='k', inline=True, fontsize=10)

        #Plot Grid
        if grid if 'grid' in globals() else False:
            ax.vlines(cx[x1:x2], cy[y1], cy[y2], colors='k', linestyles='solid', alpha=0.5)
            ax.hlines(cy[y1:y2], cx[x1], cx[x2], colors='k', linestyles='solid', alpha=0.5)

        #Plot Nodes Indeces
        if indeces if 'indeces' in globals() else False:
            nx = np.arange(0, Lx/dx, 1, dtype=int)
            ny = np.arange(0, Ly/dy, 1, dtype=int)

            for i in nx[x1:x2:1]:
                for j in ny[y1:y2:1]:
                    ax.text(X[i][j]-dx/4, Y[i][j]-dy/3, f'{i}\n{j}', color='r', fontsize=8)

        #Plot Edges
        if edges is not None if 'edges' in globals() else False:
            [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = edges

            #parets verticals esquerra == hx, hy
            ax.plot(x[edge_hl_x], y[edge_hl_y], 'Xy', markersize=7)
            #parets verticals dreta == hx, hy
            ax.plot(x[edge_hr_x], y[edge_hr_y], 'Xm', markersize=7)

            #parets horizontals baix== vx, vy
            ax.plot(x[edge_vb_x], y[edge_vb_y], 'Pw', markersize=7)
            #parets horizontals dalt== vx, vy
            ax.plot(x[edge_vt_x], y[edge_vt_y], 'Pk', markersize=7)

        #Plot Velocities vectors
        if velocity is not None if 'velocity' in globals() else False:
            U = fo.to_torch(velocity)

            ax.quiver(CX[x1:x2,y1:y2], Y[x1:x2,y1:y2], U[0][x1:x2,y1:y2], np.zeros_like(U[0][x1:x2,y1:y2]), color='g')#, scale=1) #linewidth=1, scale=0.01,
            ax.quiver(X[x1:x2,y1:y2], CY[x1:x2,y1:y2], np.zeros_like(U[1][x1:x2,y1:y2]), U[1][x1:x2,y1:y2], color='y')#, scale=1) #color=['r','b','g'] alpha=.5, edgecolor='k', facecolor='none', linewidth=.5
            #if veloc is 0, then error, needed sclae=1

        #Plot Square
        if square is not None if 'square' in globals() else False:
            ax.plot([square[0], square[1], square[1], square[0], square[0]], [square[2], square[2], square[3], square[3], square[2]] ,'r-')
            ax.plot( square[0] + (square[1]-square[0])/2 , square[2] + (square[3]-square[2])/2 ,'xr')

        #Plot Zoom region
        if (not full_zoom if 'full_zoom' in globals() else False ) and (zoom_position is not None if 'zoom_positon' in globals() else False):
            #inset axes
            axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticklabels('')
            axins.set_yticklabels('')
            ax.indicate_inset_zoom(axins)

            #recurrent call to itself with the ax= axins
            NotImplemented

    #6.SAVE FAIL ERRORS DETECTOR
    ax.plot(X[np.where(np.isnan(Z))],Y[np.where(np.isnan(Z))],'ro')


    #7.POST-TREATMENT
    if save:
        fig.savefig(filename)
    else:
        plt.show()

    plt.close()

    return fig, ax

def plot_distribution(field, edges,
                    plot_type=['full'],
                    options=[],
                    lx='theta', ly='magnitude', ltitle='Plot Line',
                    save=False, filename='./line.png',
                    fig=None, ax=None):

    '''Function to plot line distributions over the given object edges in a clock-wise sorting.

        plot_type: full, up_down
        general_plot_options:
            -limits -> [min, max]
            -vector_axis -> 0/1

    '''


    #1.CORRECT INPUT FIELD TYPE to array
    Z = to_numpy2(field)

    #2.GET THE EDGES
    edge_x, edge_y, angle = get_line_distribution(object_mask=edges)

    #3.CREATE FIGURE
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.set(xlabel=lx, ylabel=ly, title=ltitle)

    #4.CREATE DIFFERENT PLOTS
    if 'full' in plot_type:
        #Get the properties passed
        for op in options:
            globals()[op[0]] = op[1]


        #Plot Line
        ax.plot(angle, Z[edge_x,edge_y], '--k')
        ax.plot(angle, Z[edge_x,edge_y], 'ok', markersize=5)

        #Limits
        if limits is not None if 'limits' in globals() else False:
            ax.set_ylim(limits[0], limits[1])

        #Grid
        # Change major ticks to show every x.
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))

        # Change minor ticks to show every 5. (20/4 = 5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
        ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)


    if 'up_down' in plot_type:
        #Get the properties passed
        for op in options:
            globals()[op[0]] = op[1]

        ax.plot(angle[int(len(edge_x)/2):], Z[edge_x[:int(len(edge_x)/2)],edge_y[:int(len(edge_x)/2)]], '--k')
        ax.plot(angle[int(len(edge_x)/2):], Z[edge_x[:int(len(edge_x)/2)],edge_y[:int(len(edge_x)/2)]], 'ok', markersize=5, label='up_side')

        ax.plot(angle[int(len(edge_x)/2):], Z[edge_x[int(len(edge_x)/2):][::-1],edge_y[int(len(edge_x)/2):][::-1]], '--k')
        ax.plot(angle[int(len(edge_x)/2):], Z[edge_x[int(len(edge_x)/2):][::-1],edge_y[int(len(edge_x)/2):][::-1]], 'xk', markersize=5, label='down-side')

        ax.legend()

        #Limits
        if limits is not None if 'limits' in globals() else False:
            ax.set_ylim(limits[0], limits[1])

        #Grid
        # Change major ticks to show every x.
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))

        # Change minor ticks to show every 5. (20/4 = 5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
        ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

    #5.POST-TREATMENT
    if save:
        fig.savefig(filename)
    else:
        plt.show()

    plt.close()

    return fig, ax

class GIF():
    def __init__(self, gifname='./gif.gif', total_frames=None, last_frames=15):
        self.filenames = []
        self.gifname = gifname
        self.total_frames = total_frames
        self.last_frames = last_frames

    def add_frame(self, iteration, field,
                plot_type=['surface'],
                options=[],
                Lx=None, Ly=None, dx=None, dy=None,
                lx='x', ly='y', lbar='field', ltitle='Plot Field'):

        #Plot frame
        _, _ = plot_field(field,
                    plot_type=plot_type,
                    options=options,
                    Lx=Lx, Ly=Ly, dx=dx, dy=dy,
                    lx=lx, ly=ly, lbar=lbar, ltitle=ltitle,
                    save=True, filename=f'{self.gifname}_Ite_{iteration}.png')

        #Create file name and append it to a list
        filename = f'{self.gifname}_Ite_{iteration}.png'
        self.filenames.append(filename)

        #Repeat last frame
        if (iteration == len(range(self.total_frames))-1 if self.total_frames is not None else False):
            for _ in range(self.last_frames):
                self.filenames.append(filename)

    def add_frame2(self, iteration, field, edges,
                plot_type=['full'],
                options=[],
                lx='angle', ly='magnitude', ltitle='Plot Line'):

        #Plot frame
        plot_distribution(field, edges, plot_type=plot_type, options=options, lx=lx, ly=ly, ltitle=ltitle,
                    save=True, filename=f'{self.gifname}_Ite_{iteration}.png')

        #Create file name and append it to a list
        filename = f'{self.gifname}_Ite_{iteration}.png'
        self.filenames.append(filename)

        #Repeat last frame
        if (iteration == len(range(self.total_frames))-1 if self.total_frames is not None else False):
            for _ in range(self.last_frames):
                self.filenames.append(filename)

    def add_reference(self, iteration):
        #Create file name and append it to a list
        filename = f'{self.gifname}_Ite_{iteration}.png'
        self.filenames.append(filename)

        #Repeat last frame
        if (iteration == len(range(self.total_frames))-1 if self.total_frames is not None else False):
            for _ in range(self.last_frames):
                self.filenames.append(filename)

    def build_gif(self):
        with imageio.get_writer(f'{self.gifname}.gif', mode='I') as writer:
            for filename in self.filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Remove files
        for filename in set(self.filenames):
            os.remove(filename)

def plot_snapshots(ite,field,title, dt, Lx, Ly, dx, dy):

    fig, axs = plt.subplots(2,2)
    fig.suptitle(title)

    plot_field(field[0],
        plot_type=['surface'],
        options=[],
        Lx=Lx, Ly=Ly, dx=dx, dy=dy,
        lx='x', ly='y', lbar='field', ltitle=f'{ite[0]*dt}',
        save=False, fig=fig, ax=axs[0,0])

    fig.savefig('./snaps.png')
    plt.close()



def plot_field_old(Lx, Ly, dx, dy,
    field,
    vector_axis=0, limits=[-1,1], factor=1, edges=[[],[],[],[]], zoom_pos=[0,1,0,1],
    plots=['contourn'],
    lx='x', ly='y', lbar='field', ltitle='Plot Field',
    save=False, filename='./field.png',
    fig=None, ax=None):


    '''Main function to plot vectorial and scalar fields (including masking fields).

    USAGE:
        -plots=[ [ 'plot_type', [ ['plot options' , xx], ['plot options' , xx] ]],
                 [ 'plot_type', [ ['plot options' , xx], ['plot options' , xx] ]]
               ]


        contourn:
        mask:
            -edges -> [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ]
            -square -> [x1,x2,y1,y2]
        surface:
            -limits -> [min, max]
            -contourn -> True/False
            -vector_axis -> 0/1

        debug:
            -indeces -> True/False
            -grid -> True/False
        zoom:
            -position -> [x1,x2,y1,y2]
        full_zoom
            -position -> [x1,x2,y1,y2]



    how to  make that the zoom has the same as the others? And debug, mask inside surface???
    '''

    #Create position vectors and arrays
    if isinstance(field,StaggeredGrid):
        x = np.arange(0, Lx +dx/2, dx)  #NOTICE: we include +dx in order to consider the last step in arrange
        y = np.arange(0, Ly +dy/2, dy)

        cx = np.arange(0, Lx +dx/2, dx)
        cy = np.arange(0, Ly +dy/2, dy)

    elif isinstance(field,CenteredGrid):
        #x,y of the center of the cell
        x = np.arange(dx/2, Lx-dx/2 +dx/2, dx)
        y = np.arange(dy/2, Ly-dy/2 +dy/2, dy)

        #x,y of the control volume faces or cell walls
        cx = np.arange(0, Lx +dx/2, dx)
        cy = np.arange(0, Ly +dy/2, dy)

    elif isinstance(field,(list, tuple, np.ndarray)):
        if len(field)==Lx/dx +1:
            x = np.arange(0, Lx +dx/2, dx)
            y = np.arange(0, Ly +dy/2, dy)

            cx = np.arange(0, Lx +dx/2, dx)
            cy = np.arange(0, Ly +dy/2, dy)
        else:
            #x,y of the center of the cell
            x = np.arange(dx/2, Lx-dx/2 +dx/2, dx)
            y = np.arange(dy/2, Ly-dy/2 +dy/2, dy)

            #x,y of the control volume faces or cell walls
            cx = np.arange(0, Lx +dx/2, dx)
            cy = np.arange(0, Ly +dy/2, dy)

    X, Y = np.meshgrid(x,y)

    #Convert Grid objects into numpy arrays
    if isinstance(field,StaggeredGrid):
        Z0 = field.staggered_tensor().tensors[0]._native.cpu().numpy()
        Z1 = field.staggered_tensor().tensors[1]._native.cpu().numpy()
        #stremlines verificar si rot90
        #ROTATE???
    elif isinstance(field,CenteredGrid):
        Z=field.values._native.cpu().numpy()
        Z=np.rot90(Z)
    else:
        Z=np.rot90(field)

    #FIGURE
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.set(xlabel=lx, ylabel=ly, title=ltitle)

    #Mask plot
    if 'mask' in plots:
        if isinstance(field,StaggeredGrid):
            mask0 = np.ma.masked_array(Z0, Z0 == 0) * 2
            mask1 = np.ma.masked_array(Z1, Z1 == 0) * 2

            mask = mask0
        elif isinstance(field,CenteredGrid):
            mask = np.ma.masked_array(Z, Z == 0) * 2
            mask_inverse = np.ma.masked_array(Z, Z == 1) * 5
        else:
            mask = np.ma.masked_array(Z, Z == 0) * 0.5

        plt.pcolormesh(X,Y,mask, vmin=0, vmax=5, shading='auto')

    #Contorn Plot
    if 'contourn' in plots:
        CS = ax.contour(X, Y, Z, np.arange(-0.5, 0.5, .02), extend='both')
        ax.clabel(CS, inline=True, fontsize=10)
        cbar = plt.colorbar(CS,)
        cbar.ax.set_ylabel(lbar)

    #Vector (quiver) plot
    if 'quiver' in plots:
        q = ax.quiver(X, Y, Z0, Z1)
        ax.quiverkey(q, X=0.3, Y=1.1, U=0.10, label=r'$0.1 \frac{m}{s}$', labelpos='E')

    #Streamlines plot
    if 'streamlines_color' in plots:
        #varying color
        strm = ax.streamplot(X, Y, Z0, Z1, color=Z0, linewidth=1, cmap='viridis', norm=plt.Normalize(-0.5, 0.5))
        fig.colorbar(strm.lines)
    elif 'streamlines_density' in plots:
        #Varying density along a streamline
        ax.streamplot(X, Y, Z0, Z1, density=[0.5, 1], color='k--')
    elif 'streamlines_width' in plots:
        NotImplemented

    #Surface plot
    if 'surface' in plots:
        levels = MaxNLocator(nbins=25).tick_values(limits[0],limits[1])

        # pick the desired colormap, sensible levels, and define a normalization
        # instance which takes data values and translates those into levels.
        cmap = plt.get_cmap('viridis')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        [x1,x2,y1,y2] = zoom_pos
        x1 = int(x1/dx)
        x2 = int(x2/dx)
        y1 = int(y1/dy)
        y2 = int(y2/dy)

        im = plt.pcolormesh(X[y1:y2,x1:x2],Y[y1:y2,x1:x2],Z[y1:y2,x1:x2], cmap=cmap, norm=norm, shading='auto')

        ax.vlines(cx[x1:x2], cy[y1], cy[y2], colors='k', linestyles='solid', alpha=0.5)
        ax.hlines(cy[y1:y2], cx[x1], cx[x2], colors='k', linestyles='solid', alpha=0.5)

        #Edges this maybe better on mask??
        [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = edges

        #parets verticals esquerra == hx, hy
        ax.plot(x[edge_hl_x], y[edge_hl_y], 'Xy', markersize=7)
        #parets verticals dreta == hx, hy
        ax.plot(x[edge_hr_x], y[edge_hr_y], 'Xm', markersize=7)

        #parets horizontals baix== vx, vy
        ax.plot(x[edge_vb_x], y[edge_vb_y], 'Pw', markersize=7)
        #parets horizontals dalt== vx, vy
        ax.plot(x[edge_vt_x], y[edge_vt_y], 'Pk', markersize=7)

        # #to check mesure momentum
        xD=50
        D=10

        # #factor=5
        square=[xD-D/2 - factor*dx, xD+D/2 + factor*dx, Ly/2-D/2 - factor*dy , Ly/2 + D/2 + factor*dy]
        vertices = [int(square[0]/dx), int(square[1]/dx), int(square[2]/dy), int(square[3]/dy)]  # x0,x1,y0,y1


        #Zoom region
        if 'zoom' in plots:
            # inset axes....
            axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])

            [x1,x2,y1,y2] = zoom_pos
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticklabels('')
            axins.set_yticklabels('')

            ax.indicate_inset_zoom(axins)

            ##
            axins.pcolormesh(X,Y,Z, cmap=cmap, norm=norm, shading='auto')

            axins.vlines(cx, cy[0], cy[-1], colors='k', linestyles='solid', alpha=0.5)
            axins.hlines(cy, cx[0], cx[-1], colors='k', linestyles='solid', alpha=0.5)

            nx = np.arange(0, Lx/dx, 1)
            ny = np.arange(0, Ly/dy, 1)
            nx = nx.astype(int)
            ny = ny.astype(int)

            for j in nx[::5]:
                for i in ny[::5]:
                    ax.text(X[i][j], Y[i][j], f'{i}\n{j}', color='r', fontsize=5)


            axins.plot(xD,Ly/2,'xr')

            #to check mesure momentum
            axins.plot([vertices[0], vertices[1], vertices[1], vertices[0], vertices[0]], [vertices[2], vertices[2], vertices[3], vertices[3], vertices[2]] ,'r-')
            axins.plot(vertices[0], vertices[2],'bo')
            axins.plot(vertices[1], vertices[3],'go')


    #Errors Detection
    ax.plot(X[np.where(np.isnan(Z))],Y[np.where(np.isnan(Z))],'ro')


    #POST-TREATMENT
    if save:
        fig.savefig(filename)
    else:
        plt.show()

    plt.close()

    return fig, ax


def plot_spectrum():
    #PLOT FREQUENCY SPECTRUM
    N = Nt-t_null_vel[0][0] #600 # Number of sample points  Nt        vs Sample_rate  frequency is sample points in each second
    T = Dt #1.0 / 800.0 # sample spacing  DT=1/sample rate

    t = np.linspace(0.0, N*T, N, endpoint=False)
    sample_rate=1/Dt
    dF=sample_rate/N #frequency ressolution

    yf = fft(velocity_probe[t_null_vel])
    xf = fftfreq(N, T)[:N//2]  #sample points,duration   . l'utlima part sols per agafar positives

    f=abs(xf[np.argmax(np.abs(yf[0:N//2]))]) #Frequency corresponding with maximum module

    fig, ax = plt.subplots()
    plt.loglog(xf, 2.0/N * np.abs(yf[0:N//2]), label="Stationary")
    plt.loglog(f, 2.0/N *  np.abs(yf[np.argmax(np.abs(yf[0:N//2]))]),'rx',linewidth=2)

    #not chopped
    yf = fft(velocity_probe)
    xf = fftfreq(Nt, T)[:Nt//2]
    plt.loglog(xf, 2.0/Nt * np.abs(yf[0:Nt//2]), label="Full")


    ax.set(xlabel='frequency [Hz]', title=f'Power Spectrum [ Re={Re}, N=[{resX}x{resY}] ] F={round(f,5)} $\pm$ {round(dF,6)} Hz')
    ax.legend()
    ax.grid()
    ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)
    plt.show()

def plot_bar_error():
    pass
