from engines.phi.torch.flow import *
from .Simulation import *
from util.plot.plot_tools import *
from util.operations.field_operate import *
from neural_models.train.util_train import *

from analysis.mesure import *

class TrainSim(Simulation):
    '''Training SIMULATION CLASS
    This class, defines the simulations for the training procedure. 
    It solves in a general manner  the incompressible NS equations.
    No particular inflow/outflow conditions are implemented.
    '''
    def __init__(self, config, model, bsz):

        super().__init__(config['sim_phi'])
        self.domain = Domain(x=self.Nx, y=self.Ny, boundaries=CLOSED) 
        
        # Initialize velocity and mask and density variables
        self.velocity = self.domain.staggered_grid(Noise(batch=bsz))
        self.vel_mask = self.domain.staggered_grid(Noise(batch=bsz))
        self.density = CenteredGrid(tensor(torch.zeros((bsz, self.Nx, self.Ny)).cuda(),
                                     names=['batch', 'x', 'y']), self.domain.bounds)

        # Visocisty zero during trainings
        self.viscosity= config['viscosity']
        self.dt = config['sim_phi']['dt']

        # Secondary conf file
        self.mconf = config 
        self.model = model

        # For now jst hardcoded
        self.sim_method = "convnet"

    def update(self, pressure, velocity, vel_mask, flags):
        self.velocity = velocity
        self.pressure = pressure
        self.vel_mask = vel_mask
        self.flags = flags


    #def run(self, velocity, vel_mask, density, flags):
    def run(self, it, epoch, flags):

        # Step 0: Declare advection velocity and gravity and bouyancy values
        velocity_clean = self.velocity

        # Start by applying BC!
        self.velocity *= self.vel_mask

        buoyancyScale = self.mconf['buoyancyScale']
        gravityScale = self.mconf['gravityScale']
        dt_cuda = torch.from_numpy(np.array(self.dt)).cuda()

        # Step 1: If viscosity, add viscosity
        if (self.viscosity > 0):
            self.velocity = diffuse.explicit(self.velocity, self.viscosity, dt_cuda)
            #velocity_clean = self.velocity

        # Step 2: Fluid Density
        density = advect.semi_lagrangian(self.density, velocity_clean, dt_cuda) 

        # Step 3: Advect Velocity
        self.velocity = advect.semi_lagrangian(self.velocity, velocity_clean, dt_cuda)
        #self.velocity = change_nan_zero(self.velocity, self.domain)


        # Step 4: Add External Forces
        # Step 4.1: Add Buoyancy
        if buoyancyScale > 0:
            buoyancy_force = density * buoyancyScale[0] * (self.mconf['gravityVec']['x'], self.mconf['gravityVec']['y']) >> self.velocity 
            self.velocity += buoyancy_force
        # Step 4.2: Add Gravity
        if gravityScale > 0:
            unitary_field = self.domain.scalar_grid(1)
            gravity_force = unitary_field * gravityScale[0] *( self.mconf['gravityVec']['x'], self.mconf['gravityVec']['y']) >> self.velocity 
            self.velocity += gravity_force

        # Step 5: Apply BC
        self.velocity *= self.vel_mask

        # Step 6: Perform Velocity correction
        # Step 6.1: Calculate using NN
        if self.sim_method == 'convnet':

            self.pressure, self.velocity, div_out, div_in = self.model(
                    self.velocity, (2-flags), self.domain, self.mconf['normalization'], epoch, 0, 'lt_sim_{}'.format(it))

        # Step 7: Apply BC
        self.velocity *= self.vel_mask

        return div_out, div_in

    def run_star(self):

        # Step 0: Declare advection velocity and gravity and bouyancy values
        velocity_clean = self.velocity
        
        buoyancyScale = self.mconf['buoyancyScale']
        gravityScale = self.mconf['gravityScale']
        dt_cuda = torch.from_numpy(np.array(self.dt)).cuda()

        # Step 1: If viscosity, add viscosity
        if (self.viscosity > 0):
            self.velocity = diffuse.explicit(self.velocity, self.viscosity, dt_cuda)
            #velocity_clean = self.velocity
        
        # Step 2: Fluid Density
        density = advect.semi_lagrangian(self.density, velocity_clean, dt_cuda) 

        # Step 3: Advect Velocity
        self.velocity = advect.semi_lagrangian(self.velocity, velocity_clean, dt_cuda)
        #self.velocity = change_nan_zero(self.velocity, self.domain)

        # Step 4: Add External Forces
        # Step 4.1: Add Buoyancy
        if buoyancyScale > 0:
            buoyancy_force = density * buoyancyScale[0] * (self.mconf['gravityVec']['x'], self.mconf['gravityVec']['y']) >> self.velocity 
            self.velocity += buoyancy_force
        # Step 4.2: Add Gravity
        if gravityScale > 0:
            unitary_field = self.domain.scalar_grid(1)
            gravity_force = unitary_field * gravityScale[0] *( self.mconf['gravityVec']['x'], self.mconf['gravityVec']['y']) >> self.velocity 
            self.velocity += gravity_force
    
        # Step 5: Apply BC
        self.velocity *= self.vel_mask
