from neurasim import *

Lx=150
Ly=150
Nx=150
Ny=150
dx=Lx/Nx
dy=Ly/Ny
D=10


#DOMAIN
DOMAIN = Domain(x=Nx, y=Ny, boundaries=[OPEN, STICKY],bounds=Box[0:Lx, 0:Ly])

#TESTING FIELD 1
# pressure = DOMAIN.scalar_grid(0)
# aux_lower = HardGeometryMask(Box[:, Ly/2:Ly]) >> DOMAIN.scalar_grid() 
# aux_upper = HardGeometryMask(Box[:, 0:Ly/2]) >> DOMAIN.scalar_grid() 
# pressure = pressure + aux_lower*0 + aux_upper*1

#TESTING FIELD 2
# pressure = DOMAIN.scalar_grid(0)
# aux_lower_right = HardGeometryMask(Box[Lx/2:Lx, Ly/2:Ly]) >> DOMAIN.scalar_grid() 
# aux_lower_left = HardGeometryMask(Box[0:Lx/2, Ly/2:Ly]) >> DOMAIN.scalar_grid()

# aux_upper_right = HardGeometryMask(Box[Lx/2:Lx, 0:Ly/2]) >> DOMAIN.scalar_grid() 
# aux_upper_left = HardGeometryMask(Box[0:Lx/2, 0:Ly/2]) >> DOMAIN.scalar_grid() 

# pressure = pressure + aux_lower_right*1 + aux_lower_left*0 + aux_upper_right*0 + aux_upper_left*1


# #TESTING FIELD 3
# pressure = DOMAIN.scalar_grid(0)
# aux_lower_right = HardGeometryMask(Box[25:Lx, Ly/2:Ly]) >> DOMAIN.scalar_grid() 
# aux_lower_left = HardGeometryMask(Box[0:25, Ly/2:Ly]) >> DOMAIN.scalar_grid()

# aux_upper_right = HardGeometryMask(Box[25:Lx, 0:Ly/2]) >> DOMAIN.scalar_grid() 
# aux_upper_left = HardGeometryMask(Box[0:25, 0:Ly/2]) >> DOMAIN.scalar_grid() 

# pressure = pressure + aux_lower_right*1 + aux_lower_left*0 + aux_upper_right*0 + aux_upper_left*1


#TESTING FIELD 4
# pressure = DOMAIN.scalar_grid(0)
# aux_lower_right = HardGeometryMask(Box[Ly/2:Ly, 25:Lx]) >> DOMAIN.scalar_grid() 
# aux_lower_left = HardGeometryMask(Box[Ly/2:Ly, 0:25]) >> DOMAIN.scalar_grid()

# aux_upper_right = HardGeometryMask(Box[0:Ly/2, 25:Lx]) >> DOMAIN.scalar_grid() 
# aux_upper_left = HardGeometryMask(Box[0:Ly/2, 0:25]) >> DOMAIN.scalar_grid() 

# pressure = pressure + aux_lower_right*1 + aux_lower_left*0 + aux_upper_right*0 + aux_upper_left*1


#TESTING FIELD 5
# pressure = DOMAIN.scalar_grid(0)
# aux_lower = HardGeometryMask(Box[0:25:,:]) >> DOMAIN.scalar_grid() 
# aux_upper = HardGeometryMask(Box[25:Lx, :]) >> DOMAIN.scalar_grid() 
# pressure = pressure + aux_lower*0 + aux_upper*1


# #TESTING FIELD 6
# pressure = DOMAIN.scalar_grid(0)
# aux_lower = HardGeometryMask(Box[:, Ly-25:Ly]) >> DOMAIN.scalar_grid() 
# aux_upper = HardGeometryMask(Box[:, 0:25]) >> DOMAIN.scalar_grid() 
# aux_right = HardGeometryMask(Box[Lx-25:Lx, :]) >> DOMAIN.scalar_grid() 
# pressure = pressure + aux_upper*1 + aux_lower*1 + aux_right*1 


#TESTING FIELD 7
pressure = DOMAIN.scalar_grid(0)
aux_lower = HardGeometryMask(Box[Lx/2:Lx, :]) >> DOMAIN.scalar_grid() 
aux_upper = HardGeometryMask(Box[0:Lx/2, :]) >> DOMAIN.scalar_grid() 
pressure = pressure + aux_lower*0 + aux_upper*1


#CYLINDER
obstacle = Obstacle(Sphere([25, Ly/2], radius=D/2), angular_velocity=0.0)           
FORCES_MASK = HardGeometryMask(Sphere([25, Ly/2], radius=D/2)) >> DOMAIN.scalar_grid() 
FORCES_MASK = FORCES_MASK.values._native.cpu().numpy()

#CALCULATE FORCES
dxMASK = np.ones_like(pressure.values._native.cpu().numpy())*dx
vforce, hforce = calculate_force(pressure, FORCES_MASK, dxMASK)

#RESULTS 
_, _ =  plot_field(Lx, Ly, dx, dy, pressure, limits=[0,1], plots=['surface'], lx='x', ly='y', 
    lbar='pressure', ltitle='Pressure Testing Field', save=True, filename='./pressure_testing_field.png')

_,_ =  plot_field(Lx, Ly, dx, dy, FORCES_MASK, plots=['mask'], lx='x', ly='y', 
    lbar='mask', ltitle='MASK Testing Field', save=True, filename='./pressure_testing_mask.png')
   
print(f'Vertical: {vforce} - horizontal: {hforce}')
