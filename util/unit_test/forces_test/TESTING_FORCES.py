import numpy as np
from engines.phi.torch.flow import *
from util.plot.plot_field import *
from analysis.mesure import *

out_dir='./'
alpha=0
Re=1000
Nx=150
Ny=100
Ly=100
Lx=150

xD=50
D=10

DOMAIN = Domain(x=Nx, y=Ny, boundaries=[OPEN, STICKY], bounds=Box[0:Lx, 0:Ly])

###################
iteration_field = np.load(f'{out_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_iteration_field.npy')
velocity_x_field = np.load(f'{out_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_velocity_x_field.npy')
velocity_y_field = np.load(f'{out_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_velocity_y_field.npy')
pressure_field = np.load(f'{out_dir}A_{alpha}_RE_{Re}_dx_{Nx}_{Ny}_pressure_field.npy')
pfield=pressure_field.squeeze()
      
########################
FORCES_MASK = HardGeometryMask(Sphere([xD, Ly/2], radius=D/2)) >> DOMAIN.scalar_grid()
FORCES_MASK = FORCES_MASK.values._native.cpu().numpy()
sh=FORCES_MASK

##########################3
#funcio per extreure edge exterior

#VERTICAL EDGES or WALLS (HORIZONTAL DIFFERENCE)
edgesh = np.where(sh[1:,:] != sh[:-1,:])

#Left-Vertical walls
edge_hl_x = edgesh[0][:int(len(edgesh[0])/2)]
edge_hl_y = edgesh[1][:int(len(edgesh[1])/2)]

#Right-Vertical walls
edge_hr_x = edgesh[0][int(len(edgesh[0])/2):] + 1
edge_hr_y = edgesh[1][int(len(edgesh[1])/2):]

     
#HORIZONTAL EDGES or WALLS (VERTICAL DIFFERENCE)
edgesv = np.where( sh[:,1:] != sh[:,:-1])

#Horizontal-Bottom walls
edge_vb_x = edgesv[0][0::2]
edge_vb_y = edgesv[1][0:-1:2]

#Horizontal-Top walls
edge_vt_x = edgesv[0][1::2]
edge_vt_y = edgesv[1][1::2] + 1

EDGES=[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ]


zoom_pos=[xD - D/2 -2, xD + D/2 +2, Ly/2 -D/2 -2, Ly/2 + D/2 +2]
#_,_ =  plot_field(150, 100, 1, 1, sh, plots=['surface'], limits= [-1,2], edges= EDGES, zoom_pos= zoom_pos, lx='x', ly='y', 
#   lbar='mask', ltitle='MASK Testing Field', save=True, filename='./force_testing_edges.png')



#########
#TESTING LINE DISTRIBUTIONS

edge_x_sorted, edge_y_sorted, angle = get_line_distribution(object_mask=FORCES_MASK)


dx=1
dy=1
x = np.arange(dx/2, Lx-dx/2 +dx/2, dx) 
y = np.arange(dy/2, Ly-dy/2 +dy/2, dy)

fig, ax = plt.subplots()
ax.plot(x[edge_x_sorted], y[edge_y_sorted], 'Xy', markersize=7)

for i in range(len(edge_x_sorted)):
    ax.text(x[edge_x_sorted[i]]-dx/4, y[edge_y_sorted[i]]-dy/3, f'{i}\n{angle[i]}', color='r', fontsize=8)
        
plt.savefig("mygraph.png")


#### line

fig, ax = plt.subplots()
line = np.linspace(0, 360, len(edge_x_sorted))
ax.plot(angle, pfield[edge_x_sorted,edge_y_sorted], '--k')
ax.plot(angle, pfield[edge_x_sorted,edge_y_sorted], 'ok', markersize=5)


ax.set(xlabel='angle [º]', ylabel='Pressure')

# Change major ticks to show every x.
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(0.1))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

plt.savefig("mygraph_line_v2.png")



#Distribution 
fig, ax = plt.subplots()

line = np.linspace(0, 180, int(len(edge_x_sorted)/2))

ax.plot(line, pfield[edge_x_sorted[:int(len(edge_x_sorted)/2)],edge_y_sorted[:int(len(edge_x_sorted)/2)]], '--k')
ax.plot(line, pfield[edge_x_sorted[:int(len(edge_x_sorted)/2)],edge_y_sorted[:int(len(edge_x_sorted)/2)]], 'ok', markersize=5, label='up_side')


ax.plot(line, pfield[edge_x_sorted[int(len(edge_x_sorted)/2):][::-1],edge_y_sorted[int(len(edge_x_sorted)/2):][::-1]], '--k')
ax.plot(line, pfield[edge_x_sorted[int(len(edge_x_sorted)/2):][::-1],edge_y_sorted[int(len(edge_x_sorted)/2):][::-1]], 'xk', markersize=5, label='down-side')


ax.set(xlabel='x axis [+/- angle]', ylabel='Pressure')
ax.legend()

# Change major ticks to show every x.
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(0.1))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

plt.savefig("mygraph_distri.png")