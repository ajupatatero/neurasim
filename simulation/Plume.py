from engines.phi.torch.flow import *
from engines.phi.field import divergence
from engines.phi.math.backend._backend import set_global_precision
from .WindTunnel import *
from.Simulation import *
from util.plot.plot_tools import *
from analysis.mesure import *
from interface.files.files import *
from neural_models.train.util_train import *
from util.performance.timer import *
from progress.bar import *

import pdb
import os


class Plume(Simulation):

    def __init__(self,config):
        super().__init__(config)
        
        self.D=config['D']
        self.xD=config['xD']
        self.BCx = config['BC_domain_x']
        self.cyl = config['cylinder']
        self.input_rad = self.Nx*config['input_rad']
        self.Ri = config['Richardson']
        self.gravity = config['gravity']
        self.g_x = config['gravity_x']
        self.g_y = config['gravity_y']
        self.input_vel = config['input_vel']
        self.input_density =self.Ri*self.input_vel**2.0/(self.input_rad*self.gravity)

        if self.GPU:
            self.flags = torch.zeros(1, 1, 1, self.Ny, self.Nx).cuda()
        else:
            self.flags = torch.zeros(1, 1, 1, self.Ny, self.Nx)

        try:
            self.factor = config['probe_factor']
        except:
            self.factor = 1

        self.time_recorder = Timer(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_')          

    def define_simulation_geometry(self):
        self.time_recorder.record(point_name='init_define_simulation_geometry')

        self.DOMAIN = Domain(x=self.Nx, y=self.Ny, boundaries=[CLOSED], bounds=Box[0:self.Lx, 0:self.Ly])
        min_inflow = np.int(self.Lx/2 - self.input_rad)
        max_inflow = np.int(self.Lx/2 + self.input_rad)
        self.INFLOW_DENSITY = HardGeometryMask(Box[min_inflow:max_inflow+1, :3]) >> self.DOMAIN.scalar_grid()
        self.INFLOW_DENSITY *= 1
        self.INFLOW = HardGeometryMask(Box[min_inflow:max_inflow+1, :3]) >> self.DOMAIN.staggered_grid()    
        # For debug
        #INFLOW = CenteredGrid(Sphere(center=(50, 10), radius=5), extrapolation.BOUNDARY, **self.DOMAIN) * 0.2

        self.time_recorder.record(point_name='end_define_simulation_geometry')

    def define_simulation_fields(self):
        self.time_recorder.record(point_name='init_define_simulation_fields')
        
        #Initialize the fields
        bsz = 1
        
        if self.resume:
            #Import the saved fields
            
            velx = np.load(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity_x_field.npy')[-1,0,:,:] 
            vely = np.load(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity_y_field.npy')[-1,0,:,:] 

            velx = torch.from_numpy(velx).cuda()
            vely = torch.from_numpy(vely).cuda()

            velx = velx[None, None, :, :]
            vely = vely[None, None, :, :]

            velocity_big = torch.cat((velx, vely), dim=1)

            tensor_U = math.wrap(velocity_big.squeeze(2), 'batch,vector,x,y')

            tensor_U_unstack = unstack_staggered_tensor(tensor_U)
            self.velocity =  StaggeredGrid(tensor_U_unstack, self.DOMAIN.bounds)

            #TODO: use the staggered function, verify if ok

            
            try:
                velmaskx = np.load(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_vel_mask_x_field.npy')[-1,0,:,:] 
                velmasky = np.load(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_vel_mask_y_field.npy')[-1,0,:,:] 

                velmaskx = torch.from_numpy(velmaskx).cuda()
                velmasky = torch.from_numpy(velmasky).cuda()

                velmaskx = velmaskx[None, None, :, :]
                velmasky = velmasky[None, None, :, :]

                velmaskbig = torch.cat((velmaskx, velmasky), dim=1)

                tensor_U_mask = math.wrap(velmaskbig.squeeze(2), 'batch,vector,x,y')

                tensor_U_mask_unstack = unstack_staggered_tensor(tensor_U_mask)
                self.vel_mask =  StaggeredGrid(tensor_U_mask_unstack, self.DOMAIN.bounds)
            except:
                self.vel_mask = ((self.DOMAIN.staggered_grid(Noise(batch=bsz)) * 0 )+1) 
                print('the vel mask was not imported in resume')

            pfield = np.load(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_pressure_field.npy')[-1,:,:,:]
            self.pressure = CenteredGrid(tensor(torch.from_numpy(pfield).cuda(), names=['batch', 'x', 'y']), self.DOMAIN.bounds)

            self.density = CenteredGrid(tensor(torch.zeros((bsz, self.Nx, self.Ny)).cuda(), names=['batch', 'x', 'y']), self.DOMAIN.bounds)
     
        else:
            #Create the fields
            self.velocity = (self.DOMAIN.staggered_grid(Noise(batch=bsz)) * 0 )
            self.vel_mask = (self.DOMAIN.staggered_grid(Noise(batch=bsz)) * 0 ) 
            self.pressure = CenteredGrid(tensor(torch.zeros((bsz, self.Nx, self.Ny)), names=['batch', 'x', 'y']), self.DOMAIN.bounds)
            self.density = CenteredGrid(tensor(torch.zeros((bsz, self.Nx, self.Ny)), names=['batch', 'x', 'y']), self.DOMAIN.bounds)

        self.time_recorder.record(point_name='end_define_simulation_fields')

    def initialize_aux_variables(self):
        self.time_recorder.record(point_name='init_initialize_aux_variables')
        
        #Output Variables Initialization
        if self.resume:

            self.velocity_x_field=np.load(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity_x_field.npy').tolist()
            self.velocity_y_field=np.load(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity_y_field.npy').tolist()
            self.pressure_field=np.load(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_pressure_field.npy').tolist()
            self.iteration_field=np.load(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_iteration_field.npy').tolist()

            try:
                self.vel_mask_x_field=np.load(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_vel_mask_x_field.npy').tolist()
                self.vel_mask_y_field=np.load(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_vel_mask_y_field.npy').tolist()
            except:
                self.vel_mask_x_field=[]
                self.vel_mask_y_field=[]

        else:

            self.velocity_x_field=[]
            self.velocity_y_field=[]
            self.pressure_field=[]
            self.vel_mask_x_field=[]
            self.vel_mask_y_field=[]
            self.iteration_field=[]

        if self.plot_field:
            self.gif_pressure = GIF(gifname=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_pressure', total_frames=self.Nt)
            self.gif_vorticity = GIF(gifname=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_vorticity', total_frames=self.Nt)
            self.gif_density = GIF(gifname=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_density', total_frames=self.Nt)
            self.gif_divergence = GIF(gifname=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_divergence', total_frames=self.Nt)
            self.gif_velocity = GIF(gifname=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity', total_frames=self.Nt)
            self.gif_velocity_x = GIF(gifname=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity_x', total_frames=self.Nt)
            self.gif_velocity_y = GIF(gifname=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity_y', total_frames=self.Nt)
            self.gif_distribution = GIF(gifname=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_distribution', total_frames=self.Nt)

        self.bar = IncrementalBar(f' [Ri={self.Ri}, Nx={self.Nx}]', max=self.Nt, suffix= '%(percent)d%% [%(index)d/%(max)d] | %(eta_td)s remaining')

        self.time_recorder.record(point_name='end_initialize_aux_variables')

    def solve_poisson(self):
        self.time_recorder.record(point_name=f'ite_{self.ite}_>init_poisson')

        if self.sim_method == 'CG':
            self.time_recorder.record(point_name=f'ite_{self.ite}_>init_poisson__CG')

            self.velocity, self.pressure, self._iterations, self.div_in, time_CG = fluid.make_incompressible_BC(self.velocity, self.DOMAIN, (), pressure_guess=self.pressure,
            solve_params=math.LinearSolve(absolute_tolerance = self.precision, max_iterations = self.max_iterations ), solver=self.sim_method)
            
            self.time_recorder.add_single_interval(time_CG, interval_name = f'ite_{self.ite}_>CG_inference_interval')
            
            self.div_out = divergence(self.velocity)

            self.time_recorder.record(point_name=f'ite_{self.ite}_>end_poisson__CG')

        elif self.sim_method == 'convnet':
            self.time_recorder.record(point_name=f'ite_{self.ite}_>init_poisson__convnet')

            if self.ite<int(self.ite_transition):
                #self.velocity, self.pressure, self._iterations, self.div_in = fluid.make_incompressible(self.velocity, self.DOMAIN, (), pressure_guess=self.pressure,
                #solve_params=math.LinearSolve(absolute_tolerance = self.precision, max_iterations = self.max_iterations) )

                self.velocity, self.pressure, self._iterations, self.div_in, time_CG = fluid.make_incompressible_BC(self.velocity, self.DOMAIN, (), pressure_guess=self.pressure,
                solve_params=math.LinearSolve(absolute_tolerance = self.precision, max_iterations = self.max_iterations ), solver=self.sim_method)

                self.time_recorder.add_single_interval(time_CG, interval_name = f'ite_{self.ite}_>CG_inference_interval')
                
                self.div_out = divergence(self.velocity)
            else:
                in_density_t = self.density.values._native.transpose(-1, -2)
                in_U_t = torch.cat((self.velocity.staggered_tensor().tensors[0]._native.transpose(-1, -2).unsqueeze(1),
                            self.velocity.staggered_tensor().tensors[1]._native.transpose(-1, -2).unsqueeze(1)), dim=1)

                data = torch.cat((in_density_t.unsqueeze(1).unsqueeze(1), 
                            in_U_t[:,0,:-1,:-1].unsqueeze(1).unsqueeze(1), 
                            in_U_t[:,1,:-1,:-1].unsqueeze(1).unsqueeze(1), 
                            (self.flags+1), 
                            in_density_t.unsqueeze(1).unsqueeze(1)), dim = 1)
                #data = data.transpose(-1, -2)

                with torch.no_grad():
                    if self.new_train:
                        # Apply input/output BC
                        _, _, UDiv_CG = convert_phi_to_torch(self.velocity, self.pressure, self.pressure)
                        UDiv_CG = UDiv_CG.unsqueeze(2)
                        self.velocity, _ = load_values(UDiv_CG, 1-self.flags, self.DOMAIN)

                        self.pressure, self.velocity, self.div_out, self.div_in, time_Unet = self.model(self.velocity, 1-self.flags, 
                                    self.DOMAIN, self.config_norm, self.ite, 0, 'vk_inside')

                        time_Unet = float(time_Unet[0]) #to pick the total the rest are steps                        
                        self.time_recorder.add_single_interval(time_Unet, interval_name = f'ite_{self.ite}_>UNET_inference_interval')

                    else:
                        p, U_torch, self.time = self.model(data, self.ite, self.out_dir)
                        self.pressure, self.velocity, self.vel_mask, self.div_out, self.div_in = convert_torch_to_phi(p, U_torch, in_U_t, self.flags, self.DOMAIN)

                    #Net scale prediction correction
                    self.pressure = self.pressure *self.dx

                    #Center aproximation to account for pressure zero
                    self.pressure = self.pressure-2

            self.time_recorder.record(point_name=f'ite_{self.ite}_>end_poisson__convnet')

        #Correct pseudo-pressure to pressure
        self.pressure = self.pressure / self.dt

        self.time_recorder.record(point_name=f'ite_{self.ite}_>end_poisson')

    def plot_pressure(self, zoom_pos = []):

        if self.plot_field_gif:
            self.gif_pressure.add_frame(self.ite, self.pressure,
                plot_type=['surface'],
                options=[ ['limits', [-0.5, 0.5]],
                ['full_zoom', True],
                ['zoom_position', zoom_pos],
                #['edges',edges],
                #['square', [x1,x2,x3,x4]],
                ['aux_contourn', True],
                ['indeces', False],
                ['grid', False]                                    
                ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='pressure []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]')

        if self.plot_field_steps:
            plot_field(self.pressure,
                plot_type=['surface'],
                options=[ ['limits', [-0.5, 0.5]],
                        ['full_zoom', False],
                        ['zoom_position', zoom_pos],
                        ['aux_contourn', True],                
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='pressure []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]', 
                save=True, filename=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_pressure_timestep_{self.ite}.png')

    def plot_vorticity(self, zoom_pos = []):
        vorticity = calculate_vorticity(self.Lx,self.Ly,self.dx,self.dy,self.velocity)           
        
        if self.plot_field_gif:
            self.gif_vorticity.add_frame(self.ite, vorticity,
                plot_type=['surface'],
                options=[ ['limits', [-0.2, 0.5]],
                        ['full_zoom', False],
                        #['zoom_position', zoom_pos],
                        #['edges',edges],
                        #['square', [x1,x2,x3,x4]],
                        ['aux_contourn', True],
                        ['indeces', False],
                        ['grid', False]                                    
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='vorticity []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]')

        if self.plot_field_steps:
            plot_field(vorticity,
                plot_type=['surface'],
                options=[ ['limits', [-0.2, 0.5]],
                        ['full_zoom', False],
                        ['zoom_position', zoom_pos],
                        ['aux_contourn', True],                
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='vorticity []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]', 
                save=True, filename=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_vorticity_timestep_{self.ite}.png')

    def plot_density(self, zoom_pos = []):      
        
        if self.plot_field_gif:
            self.gif_density.add_frame(self.ite, self.density,
                plot_type=['surface'],
                options=[ ['limits', [-self.input_density, self.input_density]],
                        ['full_zoom', False],
                        ['zoom_position', zoom_pos],
                        #['edges',edges],
                        #['square', [x1,x2,x3,x4]],
                        ['aux_contourn', True],
                        ['indeces', False],
                        ['grid', False]                                    
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='density []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]')

        if self.plot_field_steps:
            plot_field(self.density,
                plot_type=['surface'],
                options=[ ['limits', [-self.input_density, self.input_density]],
                        ['full_zoom', False],
                        ['zoom_position', zoom_pos],
                        ['aux_contourn', True],                
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='density []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]', 
                save=True, filename=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_density_timestep_{self.ite}.png')

    def plot_divergence(self, zoom_pos = []):      
        div = divergence(self.velocity)
        if self.plot_field_gif:
            self.gif_divergence.add_frame(self.ite, div,
                plot_type=['surface'],
                options=[ #['limits', [-0.2, 0.5]],
                        ['full_zoom', False],
                        #['zoom_position', zoom_pos],
                        #['edges',edges],
                        #['square', [x1,x2,x3,x4]],
                        ['aux_contourn', True],
                        ['indeces', False],
                        ['grid', False]                                    
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='divergence []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]')

        if self.plot_field_steps:
            plot_field(div,
                plot_type=['surface'],
                options=[ #['limits', [-0.2, 0.5]],
                        ['full_zoom', False],
                        ['zoom_position', zoom_pos],
                        ['aux_contourn', True],                
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='divergence []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]', 
                save=True, filename=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_divergence_timestep_{self.ite}.png')

    def plot_velocity_norm(self, zoom_pos = []):
        norm_velocity = calculate_norm_velocity(self.velocity)            
        if self.plot_field_gif:
            self.gif_velocity.add_frame(self.ite, norm_velocity,
                plot_type=['surface'],
                options=[ ['limits', [0, 1.2]],
                        ['full_zoom', False],
                        ['zoom_position', zoom_pos],
                        #['edges',edges],
                        #['square', [x1,x2,x3,x4]],
                        ['aux_contourn', True],
                        ['indeces', False],
                        ['grid', False]                                   
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='norm velocity []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]')

        if self.plot_field_steps:
            plot_field(norm_velocity,
                plot_type=['surface'],
                options=[ ['limits', [0, 1.2]],
                        ['full_zoom', False],
                        ['zoom_position', zoom_pos],
                        #['edges',edges],
                        #['square', [x1,x2,x3,x4]],
                        ['aux_contourn', True],
                        ['indeces', False],
                        ['grid', False]                                   
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='norm velocity []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]', 
                save=True, filename=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity_timestep_{self.ite}.png')

    def plot_velocity_x(self, zoom_pos = []):
        vel_x = self.velocity.staggered_tensor().tensors[0]._native[0].cpu().numpy()
        if self.plot_field_gif:
            self.gif_velocity_x.add_frame(self.ite, vel_x,
                plot_type=['surface'],
                options=[ ['limits', [0, 1.2]],
                        ['full_zoom', False],
                        ['zoom_position', zoom_pos],
                        #['edges',edges],
                        #['square', [x1,x2,x3,x4]],
                        ['aux_contourn', True],
                        ['indeces', False],
                        ['grid', False]                                   
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='vel x []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]')

        if self.plot_field_steps:
            plot_field(vel_x,
                plot_type=['surface'],
                options=[ ['limits', [0, 1.2]],
                        ['full_zoom', False],
                        ['zoom_position', zoom_pos],
                        #['edges',edges],
                        #['square', [x1,x2,x3,x4]],
                        ['aux_contourn', True],
                        ['indeces', False],
                        ['grid', False]                                   
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='vel x []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]', 
                save=True, filename=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity_x_timestep_{self.ite}.png')

    def plot_velocity_y(self, zoom_pos = []):
        vel_y = self.velocity.staggered_tensor().tensors[1]._native[0].cpu().numpy()
        if self.plot_field_gif:
            self.gif_velocity_y.add_frame(self.ite, vel_y,
                plot_type=['surface'],
                options=[ ['limits', [0, 1.2]],
                        ['full_zoom', False],
                        ['zoom_position', zoom_pos],
                        #['edges',edges],
                        #['square', [x1,x2,x3,x4]],
                        ['aux_contourn', True],
                        ['indeces', False],
                        ['grid', False]                                   
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='vel y []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]')

        if self.plot_field_steps:
            plot_field(vel_y,
                plot_type=['surface'],
                options=[ ['limits', [0, 1.2]],
                        ['full_zoom', False],
                        ['zoom_position', zoom_pos],
                        #['edges',edges],
                        #['square', [x1,x2,x3,x4]],
                        ['aux_contourn', True],
                        ['indeces', False],
                        ['grid', False]                                   
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='vel y []',
                ltitle=f'Plume @ t={np.round(self.dt*self.ite, decimals=1)} s [ Ri={self.Ri}, N=[{self.Nx}x{self.Ny}] ]', 
                save=True, filename=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity_y_timestep_{self.ite}.png')

    
    def plot_geometry(self, zoom_pos = []):
        xp1 = int(self.xD + self.D*2)
        xp2 = int(self.xD + self.D*2.5)
        yp1 = int(self.Ly/2 - self.D*0.25)
        yp2 = int(self.Ly/2 + self.D*0.25)
        
        plot_field(self.CYLINDER,
            plot_type=['surface'],
            options=[ ['limits', [-1, 1]],
                    ['full_zoom', False],
                    ['zoom_position', zoom_pos],
                    ['aux_contourn', False],  
                    ['square', [xp1,xp2,yp1,yp2]]              
                    ],
            Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
            lx='x', ly='y',lbar='geometry',
            ltitle=f'Plume @ [ N=[{self.Nx}x{self.Ny}] ]', 
            save=True, filename=f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_geometry.png')

    def save_variables(self):

        #3.2.SAVE FIELDS
        if self.save_field and ( ( self.ite%self.save_field_x_ite == 0 if self.DEBUG else False ) or self.ite == self.Nt-1 ):
            np.save(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity_x_field.npy', self.velocity_x_field)
            np.save(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_velocity_y_field.npy', self.velocity_y_field)
            np.save(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_pressure_field.npy', self.pressure_field)
            np.save(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_vel_mask_x_field.npy', self.vel_mask_x_field)
            np.save(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_vel_mask_y_field.npy', self.vel_mask_y_field)
            np.save(f'{self.out_dir}Ri_{self.Ri}_dx_{self.Nx}_{self.Ny}_iteration_field.npy', self.iteration_field)

            #export_csv('./results',pressure, self.Lx, self.Ly, self.dx, self.dy)

    def run(self):
        _init_sim = torch.cuda.Event(enable_timing=True) #Internal timer, to check simulation time, in order to avoid 24h max
        _now_sim = torch.cuda.Event(enable_timing=True)
        _init_sim.record()


        self.time_recorder.record(point_name='run_init')

        if self.FP64:
            print('FP64 ACTIVE!!!!!!')
            set_global_precision(64) #PhiFlow
            torch.set_default_dtype(torch.float64) #Torch

            '''NOTICE: since both backends (phiflow & torch) are defined as default here. It is not necessary nor recommended
                to define its precision on other parts of the code. Except for very specific purpouses.            
            '''

        if self.GPU == True:
            TORCH_BACKEND.set_default_device('GPU') #PhiFlow
            torch.set_default_tensor_type('torch.cuda.FloatTensor') #Torch

            '''NOTICE: since both backends (phiflow & torch) are defined as default here. It is not necessary nor recommended
                to use .cuda() in other parts of the software. Since then, will probably create internal conflicts when using 
                multiple tensors located in different devices.

                Another thing, is the .cpu() used for instance, in the plots since this doesn't provoke any conflict since always
                will be required to pass it to cpu. And if it was already on cpu it doesn't bring any problem.
            '''

        self.time_recorder.record(point_name='loading_gpu')

        # Initialize network
        if self.sim_method == 'convnet':
            self.load_model()
            self.time_recorder.record(point_name='loading_network')

        #0.PREPARE SIMULATION
        self.define_simulation_geometry()
        self.define_simulation_fields()
        self.initialize_aux_variables()


        #1.COMPUTATIONS ITERATIONs OVER TIME
        if self.resume:
            ite_init = self.iteration_field[-1]
        else:
            ite_init = 0

        self.time_recorder.record(point_name='init_iterations')
        for self.ite in range(ite_init, self.Nt):
            
            #1.0.Check if simulation time exceeded maximum allocation (24h gpu on pando -> 23h)
            _now_sim.record()
            torch.cuda.synchronize()
            if _init_sim.elapsed_time(_now_sim) >= self.MAX_TIME:
                print(f'the simulation took more than {_init_sim.elapsed_time(_now_sim)/(1000*60)} min, so a new job will be launched to proceed.')
                self.prepare_resume()
                exit()

            self.time_recorder.record(point_name=f'init_iteration_{self.ite}')
            if True: #try:

                self.velocity_free = self.velocity
                #1.1.Advect Density

                self.density = advect.mac_cormack(self.density, self.velocity, self.dt)

                self.density = self.density * (1 - self.INFLOW_DENSITY) + self.INFLOW_DENSITY * self.input_density

                #1.2.Advect Velocity
                self.velocity = advect.semi_lagrangian(self.velocity_free, self.velocity, self.dt)

                self.time_recorder.record(point_name=f'ite_{self.ite}_>advect')

                #1.3.Apply Boundary Conditions
                self.velocity = self.velocity * (1 - self.INFLOW) + self.INFLOW * (0, 1)
                #TODO: el inflow, como init dentro de bc functions



                #if self.sim_method == 'CG' or self.sim_method == 'convnet':
                #    self.velocity = apply_boundaries(self.velocity, self.bc_mask, self.bc_value)
                #    self.time_recorder.record(point_name=f'ite_{self.ite}_>apply_bc')

                buoyancy_force = self.density * self.gravity * (self.g_x, self.g_y) >> self.velocity  # resamples smoke to velocity sample points
                self.velocity = advect.semi_lagrangian(self.velocity, self.velocity, self.dt) + buoyancy_force

                #1.4.Solve Poisson Equation
                self.solve_poisson()

                #1.5.Reenforce Boundary Conditions 
                #if self.sim_method == 'CG' or self.sim_method == 'convnet':
                #    self.velocity = apply_boundaries(self.velocity, self.bc_mask, self.bc_value)
                #    self.time_recorder.record(point_name=f'ite_{self.ite}_>reinforce_bc')
                #include init and inflow????

            #except:
                #ERROR OF CONVERGENCE, STOP SIMULATION + SAVE GIF, ETC
                #TODO: log of failure  f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_log.log'
                #break 


            #2.POST-PROCESSING
            self.time_recorder.record(point_name=f'ite_{self.ite}_>init_post')
            if True: #try:

                #2.3.PLOT RESULTS #TODO: multithrad
                if self.plot_field and self.ite%self.plot_x_ite == 0:
                    zoom_pos=[np.int(self.Lx/2 - 1.5*self.input_rad), np.int(self.Lx/2 + 1.5*self.input_rad), 
                            0, 10]

                    self.plot_pressure(zoom_pos = zoom_pos)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>pressure')

                    self.plot_vorticity(zoom_pos = zoom_pos)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>vorticity')

                    self.plot_density(zoom_pos = zoom_pos)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>density')

                    self.plot_divergence(zoom_pos = zoom_pos)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>divergence')

                    self.plot_velocity_norm(zoom_pos = zoom_pos)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>velocity')

                    self.plot_velocity_x(zoom_pos = zoom_pos)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>velocity_x')

                    self.plot_velocity_y(zoom_pos = zoom_pos)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>velocity_y')


                #2.4.FIELD VARIABLES SAVING PREPARATION
                if self.save_field and self.ite%self.save_field_x_ite == 0:
                    self.velocity_x_field.append(self.velocity.staggered_tensor().tensors[0]._native.cpu().numpy())
                    self.velocity_y_field.append(self.velocity.staggered_tensor().tensors[1]._native.cpu().numpy())
                    self.pressure_field.append(self.pressure.values._native.cpu().numpy())
                    self.vel_mask_x_field.append(self.vel_mask.staggered_tensor().tensors[0]._native.cpu().numpy())
                    self.vel_mask_y_field.append(self.vel_mask.staggered_tensor().tensors[1]._native.cpu().numpy())
                    self.iteration_field.append(self.ite)
                
                if self.ite%self.save_post_x_ite == 0 and self.ite> self.min_ite:
                    filename3 = self.out_dir + '/P_output_{0:05}'.format(self.ite)
                    np.save(filename3,self.pressure.values._native.cpu().numpy())
                    filename4 = self.out_dir + '/Div_output_{0:05}'.format(self.ite)
                    div_val = divergence(self.velocity)
                    np.save(filename4, div_val.values._native.cpu().numpy())

                    filename5 = self.out_dir + '/Ux_NN_output_{0:05}'.format(self.ite)
                    np.save(filename5, self.velocity.staggered_tensor().tensors[0]._native.cpu().numpy()[0,:-1,:-1])
                    filename6 = self.out_dir + '/Uy_NN_output_{0:05}'.format(self.ite)
                    np.save(filename6, self.velocity.staggered_tensor().tensors[1]._native.cpu().numpy()[0,:-1,:-1])

                    filename7 = self.out_dir + '/Rho_NN_output_{0:05}'.format(self.ite)
                    np.save(filename7, self.density.values._native.cpu().numpy())



            #except:
            #    pass
            

            #3.SAVE RESULTS 
            self.time_recorder.record(point_name=f'ite_{self.ite}_>init_save_results')
            if True: #try:
                self.save_variables()
            #except:
            #    pass
            self.time_recorder.record(point_name=f'ite_{self.ite}_>end_save_results')


            self.bar.next()
        self.bar.finish()
        self.time_recorder.record(point_name='end_iterations')


        #FINAL POST-PROCES
        try:
            if self.plot_field:
                self.plot_geometry(zoom_pos = zoom_pos)
                
                if self.plot_field_gif:
                    self.gif_pressure.build_gif()
                    self.gif_vorticity.build_gif()
                    self.gif_density.build_gif()
                    self.gif_divergence.build_gif()
                    self.gif_velocity.build_gif()
                    self.gif_velocity_x.build_gif()
                    self.gif_velocity_y.build_gif()
                    self.gif_distribution.build_gif()
            
        except:
            pass

        #FINAL SAVINGS
        try:
            self.save_variables()
        except:
            pass
            
        if self.resume:
            #TODO: eliminate temporal files and folders
            pass


        self.time_recorder.record(point_name='run_end')
        self.time_recorder.close(save=True)
        