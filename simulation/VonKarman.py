from engines.phi.torch.flow import *
from engines.phi.field import divergence
from engines.phi.math.backend._backend import set_global_precision
from .WindTunnel import *
from util.plot.plot_tools import *
from analysis.mesure import *
from interface.files.files import *
from neural_models.train.util_train import *
from util.performance.timer import *
from progress.bar import *

import pdb
import os


class VonKarman(WindTunnel):
    '''VON KARMAN SIMULATION CLASS
    This class, defines the simulations of type Von Karman. Which consists of a wind tunnel and within, a cylinder.
    '''
    def __init__(self,config):
        super().__init__(config)

        self.D=config['D']
        self.xD=config['xD']
        #self.yD=config['yD']

        self.Re=config['Reynolds']
        self.viscosity=self.D/self.Re
        self.alpha=0

        self.BCx = config['BC_domain_x']

        #TODO: limpiar metodo, y ampliar en caso de BC: (OPEN, CLOSED)
        if self.BCx == 'OPEN':
            self.BCx = OPEN
        elif self.BCx == 'CLOSED':
            self.BCx = CLOSED
        elif self.BCx == 'STICKY':
            self.BCx = STICKY
        else:
            print('NO BC IMPLEMENTED')
            exit()

        self.BCy = config['BC_domain_y']
        #TODO: limpiar metodo, y ampliar en caso de BC: (OPEN, CLOSED)
        if self.BCy == 'OPEN':
            self.BCy = OPEN
        elif self.BCy == 'CLOSED':
            self.BCy = CLOSED
        elif self.BCy == 'STICKY':
            self.BCy = STICKY
        else:
            print('NO BC IMPLEMENTED')
            exit()

        if self.GPU:
            self.flags = torch.zeros(1, 1, 1, self.Ny, self.Nx).cuda()
        else:
            self.flags = torch.zeros(1, 1, 1, self.Ny, self.Nx)
        #self.flags[:, :, :, 0, :] = 1
        #self.flags[:, :, :, -1, :] = 1
        #self.flags[:, :, :, :, 0] = 1
        #self.flags[:, :, :, :, -1] = 1

    def run(self):
        if self.GPU == True:
            TORCH_BACKEND.set_default_device('GPU')

        #DOMAIN
        DOMAIN = Domain(x=self.Nx, y=self.Ny, boundaries=[OPEN, STICKY],bounds=Box[0:self.Lx, 0:self.Ly])

        #INLET BC FLOW
        BOUNDARY_MASK = HardGeometryMask(Box[:0.5, :]) >> DOMAIN.staggered_grid()

        #INITIAL DESTABILIZATION
        xi1 = self.xD + 2.5*self.D
        xi2 = xi1 + self.D/2
        yi1 = self.Ly/2 - self.D/2
        yi2 = self.Ly/2 + self.D
        INIT = HardGeometryMask(Box[xi1:xi2,yi1:yi2]) >> DOMAIN.staggered_grid()

        #CYLINDER
        obstacle = Obstacle(Sphere([self.xD, self.Ly/2], radius=self.D/2), angular_velocity=0.0)
        FORCES_MASK = HardGeometryMask(Sphere([self.xD, self.Ly/2], radius=self.D/2)) >> DOMAIN.scalar_grid()
        self.flags += FORCES_MASK.values._native
        #FORCES_MASK = FORCES_MASK.values._native.cpu().numpy()  in theory not necessary

        #INITIALIZE FIELDS
        bsz = 1
        velocity = ((DOMAIN.staggered_grid(Noise(batch=bsz)) * 0 )+1) *(1,0)
        pressure = CenteredGrid(tensor(torch.zeros((bsz, self.Nx, self.Ny)).cuda(),
                                     names=['batch', 'x', 'y']), DOMAIN.bounds)
        density = CenteredGrid(tensor(torch.zeros((bsz, self.Nx, self.Ny)).cuda(),
                                     names=['batch', 'x', 'y']), DOMAIN.bounds)
        vel_mask = create_from_flags(self.flags, velocity)


        #Output Variables Initialization
        velocity_probe=np.zeros(self.Nt)
        vforce=np.zeros(self.Nt)
        hforce=np.zeros(self.Nt)

        velocity_x_field=[]
        velocity_y_field=[]
        pressure_field=[]
        iteration_field=[]

        #TODO: multithread
        gif_pressure = GIF(gifname=f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_pressure', total_frames=self.Nt)
        gif_vorticity = GIF(gifname=f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vorticity', total_frames=self.Nt)
        gif_velocity = GIF(gifname=f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity', total_frames=self.Nt)


        # Initialize network
        if self.sim_method == 'convnet':
            self.load_model()


        #ITERATION OVER TIME
        bar = Bar(f' [RE={self.Re}, Nx={self.Nx}]', max=self.Nt, suffix='%(percent)d%%')
        for ite in range(self.Nt):

            #1.COMPUTATIONS
            try:
                velocity_free = diffuse.explicit(velocity, self.viscosity, self.dt)
                velocity = advect.semi_lagrangian(velocity_free, velocity, self.dt)
                velocity = velocity * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (1, 0) + INIT*(0,0.5) if ite<10 else velocity * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (1, 0)

                if self.sim_method == 'CG':
                    velocity, pressure, _iterations, div_in = fluid.make_incompressible(velocity, DOMAIN, (obstacle,), pressure_guess=pressure,
                    solve_params=math.LinearSolve(absolute_tolerance=1e-3, max_iterations=1e5))
                    div_out = divergence(velocity)
                else:

                    # Apply BC!
                    in_density_t = density.values._native
                    in_U_t = torch.cat((velocity.staggered_tensor().tensors[0]._native.unsqueeze(1),
                                        velocity.staggered_tensor().tensors[1]._native.unsqueeze(1)), dim=1)

                    in_U_t[:,0,:-1,:-1] = in_U_t[:,0,:-1,:-1] * (1-self.flags)
                    in_U_t[:,1,:-1,:-1] = in_U_t[:,1,:-1,:-1] * (1-self.flags)

                    in_U_t[:,0,:2,:] = 1
                    in_U_t[:,0,-2:,:] = 1

                    data = torch.cat((in_density_t.unsqueeze(1).unsqueeze(1),
                                    in_U_t[:,0,:-1,:-1].unsqueeze(1).unsqueeze(1),
                                    in_U_t[:,1,:-1,:-1].unsqueeze(1).unsqueeze(1),
                                    (self.flags+1),
                                    in_density_t.unsqueeze(1).unsqueeze(1)), dim = 1)
                    data = data.transpose(-1, -2)

                    with torch.no_grad():
                        if self.new_train:
                            pressure, velocity, vel_mask, div_out, div_in, time = self.model(data, velocity,
                                                                        vel_mask, DOMAIN, True,  ite, self.out_dir)
                        else:
                            p, U_torch, time = self.model(data, ite, self.out_dir)
                            pressure, velocity, vel_mask, div_out, div_in = convert_torch_to_phi(p, U_torch, in_U_t, self.flags, DOMAIN)

                #Correct pseudo-pressure to pressure
                pressure = pressure / self.dt

            except:
                #ERROR OF CONVERGENCE, STOP SIMULATION + SAVE GIF, ETC
                break


            #2.POST-PROCESSING
            if True:
                #2.1.VELOCITY PROBE
                Dn = self.D/self.dx
                xp1 = int(self.Ny/3-Dn/5)
                xp2 = int(self.Ny/3 + Dn/5)
                yp1 = int(self.Ny/2 - Dn/10)
                yp2 = int(self.Ny/2 + Dn/10)
                #velocity_probe[ite] = np.mean(velocity.staggered_tensor().tensors[1]._native.cpu().numpy()[xp1:xp2,yp1:yp2])
                velocity_probe[ite] = np.mean(velocity.staggered_tensor().tensors[1]._native.cpu().numpy()[0, xp1:xp2,yp1:yp2])

            #2.2.CALCULATE FORCES
                hforce[ite], vforce[ite] = calculate_forces(pressure, FORCES_MASK, self.dx, self.dy)
                #hforce[ite], vforce[ite] = calculate_forces_with_momentum(pressure, velocity, FORCES_MASK, factor=1, rho=1, dx=self.dx, dy=self.dy)

            #2.3.PLOT RESULTS
            zoom_pos=[self.xD - self.D, self.xD + self.D,
                      self.Ly/2 -self.D, self.Ly/2 + self.D]
            #edges = [ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ]

            if ite%100== 0:
                gif_pressure.add_frame(ite, pressure, plot_type=['surface'],
                        options=[ ['limits', [torch.min(pressure.values._native.cpu()), torch.max(pressure.values._native.cpu())]],
                                ['zoom_position',zoom_pos] ],
                        lx='x', ly='y', lbar='pressure []', ltitle=f'VK @ t={np.round(self.dt*ite, decimals=2)} s [ Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]')


                vorticity = calculate_vorticity(self.Lx,self.Ly,self.dx,self.dy,velocity)
                gif_vorticity.add_frame(ite, vorticity, plot_type=['surface'],
                        options=[ ['limits', [-0.5, 0.5]],
                                ['zoom_position',zoom_pos] ],
                        lx='x', ly='y', lbar='vorticity []', ltitle=f'VK @ t={np.round(self.dt*ite, decimals=2)} s [ Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]')


                norm_velocity = calculate_norm_velocity(velocity)
                gif_velocity.add_frame(ite, norm_velocity, plot_type=['surface'],
                        options=[ ['limits', [0, 0.8]],
                                ['zoom_position',zoom_pos] ],
                        lx='x', ly='y', lbar='norm velocity []', ltitle=f'VK @ t={np.round(self.dt*ite, decimals=2)} s [ Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]')

                # DEBUG Ekhi
                vel_x = velocity.staggered_tensor().tensors[0]._native.cpu().numpy()[0]
                vel_y = velocity.staggered_tensor().tensors[1]._native.cpu().numpy()[0]

                plot_field(vel_x, plot_type=['surface'], options=[ ['limits', [-2, 2]], ['zoom_position',zoom_pos]],
                    lx='x', ly='y', lbar=' velocity x', ltitle=f'VK @ t={np.round(self.dt*ite, decimals=2)} s [ Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]',
                    save=True, filename='{}/velocity_x_field_{}.png'.format(self.out_dir, ite))
                plot_field(vel_y, plot_type=['surface'], options=[ ['limits', [-2, 2]], ['zoom_position',zoom_pos]],
                    lx='x', ly='y', lbar=' velocity y', ltitle=f'VK @ t={np.round(self.dt*ite, decimals=2)} s [ Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]',
                    save=True, filename='{}/velocity_y_field_{}.png'.format(self.out_dir, ite))
                plot_field(div_in, plot_type=['surface'], options=[ ['limits', [-torch.max(div_in.values._native.cpu()), torch.max(div_in.values._native.cpu())]], ['zoom_position',zoom_pos]],
                    lx='x', ly='y', lbar=' Divergence in', ltitle=f'VK @ t={np.round(self.dt*ite, decimals=2)} s [ Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]',
                    save=True, filename='{}/div_in_field_{}.png'.format(self.out_dir, ite))
                plot_field(div_out, plot_type=['surface'], options=[ ['limits', [-torch.max(div_in.values._native.cpu()), torch.max(div_in.values._native.cpu())]], ['zoom_position',zoom_pos]],
                    lx='x', ly='y', lbar=' Divergence out', ltitle=f'VK @ t={np.round(self.dt*ite, decimals=2)} s [ Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]',
                    save=True, filename='{}/div_out_field_{}.png'.format(self.out_dir, ite))

            #3.SAVE RESULTS
            try:
                #3.1.SAVE INTERMIDATE RESULTS OF POST-PROCESS
                if ite%100== 0:
                    print('Saving it: ', ite)
                    np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_probe.npy', velocity_probe)
                    np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vforce.npy', vforce)
                    np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_hforce.npy', hforce)

                #3.2.SAVE FIELD
                if self.save_field and ite%(10)== 0: #save 10%
                    velocity_x_field.append(velocity.staggered_tensor().tensors[0]._native.cpu().numpy())
                    velocity_y_field.append(velocity.staggered_tensor().tensors[1]._native.cpu().numpy())
                    pressure_field.append(pressure.values._native.cpu().numpy())
                    iteration_field.append(ite)

                    np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_x_field.npy', velocity_x_field)
                    np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_y_field.npy', velocity_y_field)
                    np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_pressure_field.npy', pressure_field)
                    np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_iteration_field.npy', iteration_field)

                    #export_csv('./results',pressure, self.Lx, self.Ly, self.dx, self.dy)
            except:
                pass

            bar.next()
        bar.finish()

        #FINAL POST-PROCES
        gif_pressure.build_gif()
        gif_vorticity.build_gif()
        gif_velocity.build_gif()

        #FINAL SAVINGS
        np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_probe.npy', velocity_probe)
        np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vforce.npy', vforce)
        np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_hforce.npy', hforce)

        #
        self.plot_forces()
        self.plot_velocity_probe()
        self.plot_power_spectrum()
        self.plot_forces_ss()

    def plot_velocity_probe(self):
        velocity_probe = np.load(f'{self.in_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_probe.npy')

        t, t_null_vel, t_upsampled, envelope, deri = detect_stationary(velocity_probe, self.Nt, self.dt)

        fig, ax = plt.subplots()
        ax.plot(t_upsampled, envelope,'r', label="Filtered Envelope"); #overlay the extracted envelope
        ax.plot(t, velocity_probe, label="Velocity")
        ax.plot(t_upsampled[:len(deri)], deri,'g--',label="Envelope Derivate")

        try:
            ax.plot([t[t_null_vel], t[t_null_vel] ],[max(velocity_probe), min(velocity_probe)],'r--', label="Stationary Regime")
        except:
            pass

        ax.set(xlabel='time [s]', ylabel='vertical velocity', title=f'Velocity Probe [ Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]')
        ax.legend()

        #GRID
        # Change major ticks to show every x.
        ax.xaxis.set_major_locator(MultipleLocator((self.Nt*self.dt)/10))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))

        # Change minor ticks to show every 5. (20/4 = 5)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))

        # Turn grid on for both major and minor ticks and style minor slightly
        # differently.
        ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
        ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

        #plt.show()
        fig.savefig(f"{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_oscil.png")
        plt.close()

    def plot_power_spectrum(self):
        velocity_probe = np.load(f'{self.in_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_probe.npy')
        xf, yf, xffull, yffull, f, dF, ffull, dFfull = compute_spectrum(velocity_probe,self.Nt,self.dt)

        fig, ax = plt.subplots()
        plt.loglog(xf, yf, label="Stationary")
        plt.loglog(f, yf[np.argmax(yf)],'rx',linewidth=2)

        plt.loglog(xffull, yffull, label="Full")

        ax.set(xlabel='frequency [Hz]', title=f'Power Spectrum [ Re={self.Re}, N=[{self.Nx}x{self.Ny}] ] F={round(f,5)} $\pm$ {round(dF,6)} Hz')
        ax.legend()
        ax.grid()
        ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)
        fig.savefig(f"{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_frequency.png")
        plt.close()

    def plot_forces(self):
        vforce = np.load(f'{self.in_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vforce.npy')
        hforce = np.load(f'{self.in_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_hforce.npy')

        t, t_null_vel, _, _, _ = detect_stationary(vforce, self.Nt, self.dt)

        if t_null_vel <= self.Nt/3:
            t_null_vel = int(self.Nt/2)

        #PASS FORCE TO COEFICIENT
        density=1
        V0=1
        Cl = (vforce*2)/(density*V0*V0*self.D)
        Cd = (hforce*2)/(density*V0*V0*self.D)

        fig, ax = plt.subplots()
        ax.plot(t, Cl, label="Cl")
        ax.plot(t, Cd, label="Cd")

        ax.plot([t[t_null_vel], t[t_null_vel] ],[np.max([np.max(Cl[t_null_vel:]), np.max(Cd[t_null_vel:])] )*1.1,
                 np.min( [ np.min(Cl[t_null_vel:]), np.min(Cd[t_null_vel:]) ] )*1.1],'r--', label="Stationary Regime")

        ax.set(xlabel='time [s]', ylabel='Coefficient', title=f'Force probe [ Alpha={self.alpha}, Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]')
        ax.legend()

        ax.set_ylim([ np.min( [ np.min(Cl[t_null_vel:]), np.min(Cd[t_null_vel:]) ] )*1.1 , np.max([np.max(Cl[t_null_vel:]), np.max(Cd[t_null_vel:])] )*1.1 ])

        ax.xaxis.set_major_locator(MultipleLocator((self.Nt*self.dt)/10))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
        ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

        fig.savefig(f"{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_forces_time.png")
        plt.close()

    def print_mean_forces(self):
        vforce = np.load(f'{self.in_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vforce.npy')
        hforce = np.load(f'{self.in_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_hforce.npy')

        _, t_null_vel, _, _, _ = detect_stationary(vforce, self.Nt, self.dt)

        if t_null_vel <= self.Nt/3:
            t_null_vel = int(self.Nt/2)

        #PASS FORCE TO COEFICIENT
        density=1
        V0=1
        Cl = (vforce*2)/(density*V0*V0*self.D)
        Cd = (hforce*2)/(density*V0*V0*self.D)

        Cl_mean = np.mean(Cl[t_null_vel:])
        Cd_mean = np.mean(Cd[t_null_vel:])

        sourceFile = open(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_CL_CD_mean.txt', 'w')
        print(f'MEAN CL: {Cl_mean}, MEAN CD: {Cd_mean}', file = sourceFile)
        sourceFile.close()

    def plot_forces_ss(self):
        vforce = np.load(f'{self.in_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vforce.npy')
        hforce = np.load(f'{self.in_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_hforce.npy')

        _, t_null_vel, _, _, _ = detect_stationary(vforce, self.Nt, self.dt)

        if t_null_vel <= self.Nt/3:
            t_null_vel = int(self.Nt*(3/4))

        #PASS FORCE TO COEFICIENT
        density=1
        V0=1
        vforce = (vforce*2)/(density*V0*V0*self.D) #Cl
        hforce = (hforce*2)/(density*V0*V0*self.D) #Cd

        fig, ax = plt.subplots()
        ax.plot(hforce[t_null_vel:], vforce[t_null_vel:], label="Cl/Cd")

        ax.set(xlabel='Cd [·]', ylabel='Cl [·]', title=f'Phase Diagram [ Alpha={self.alpha}, Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]')
        ax.legend()

        #ax.xaxis.set_major_locator(MultipleLocator((self.Nt*self.dt)/10))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(which='major', color='#CCCCCC', linestyle='-', alpha=1)
        ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.5)

        fig.savefig(f"{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_forces_ss.png")
        plt.close()

    def plot_snaps(self):
        velocity_x_field = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_x_field.npy')
        velocity_y_field = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_y_field.npy')
        pressure_field = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_pressure_field.npy')
        iteration_field = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_iteration_field.npy')

        plot_snapshots(iteration_field, pressure_field, f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_pressure_field',
                self.dt, self.Lx, self.Ly, self.dx, self.dy)

class VonKarman_rotative(VonKarman):

    def __init__(self,config):
        super().__init__(config)

        self.alpha=config['Alpha']
        self.w= (self.alpha)/(self.D/2)

        if self.Re >= 50000:
            self.Re_INF = True
        else:
            self.Re_INF = False

        try:
            self.factor = config['probe_factor']
        except:
            self.factor = 1

        self.time_recorder = Timer(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_')

    def define_simulation_geometry(self):
        self.time_recorder.record(point_name='init_define_simulation_geometry')

        self.DOMAIN = Domain(x=self.Nx, y=self.Ny, boundaries=[self.BCx, self.BCy], bounds=Box[0:self.Lx, 0:self.Ly])

        self.INFLOW = HardGeometryMask(Box[:1, :]) >> self.DOMAIN.staggered_grid()

        self.INFLOW_2 = HardGeometryMask(Box[:3, :]) >> self.DOMAIN.staggered_grid()
        self.OUTFLOW = HardGeometryMask(Box[-3:, :]) >> self.DOMAIN.staggered_grid()
        self.DOWN_WALL = HardGeometryMask(Box[:, :3]) >> self.DOMAIN.staggered_grid()
        self.UP_WALL = HardGeometryMask(Box[:, -3:]) >> self.DOMAIN.staggered_grid()

        self.cylinder = Obstacle(Sphere([self.xD, self.Ly/2], radius=self.D/2), angular_velocity=self.w)
        self.CYLINDER = HardGeometryMask(Sphere([self.xD, self.Ly/2], radius=self.D/2 )) >> self.DOMAIN.scalar_grid()
        #TODO: hacer que obstacle sea input ok de forces() y hacer obs.geom >>
        self.flags += self.CYLINDER.values._native.transpose(-1, -2) #TODO: clean, all with one mask only


        self.CYLINDER_2 = HardGeometryMask(Sphere([self.xD, self.Ly/2], radius=self.D/2 + 2*self.dx )) >> self.DOMAIN.scalar_grid()


        if not self.sim_method == 'PHI':
            self.bc_mask, self.bc_value = get_obstacles_bc([[self.CYLINDER, self.w*(self.D/2), False] ])

        self.time_recorder.record(point_name='end_define_simulation_geometry')

    def define_simulation_fields(self):
        self.time_recorder.record(point_name='init_define_simulation_fields')

        #INITIAL DESTABILIZATION
        xi1 = self.xD + 1*self.D
        xi2 = xi1 + self.D/2
        yi1 = self.Ly/2 - self.D/2
        yi2 = self.Ly/2 + self.D/2
        self.INIT = HardGeometryMask(Box[xi1:xi2,yi1:yi2]) >> self.DOMAIN.staggered_grid()
        self.INIT_transition = 10

        #Initialize the fields
        bsz = 1

        if self.resume:
            #Import the saved fields
            velx = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_x_field.npy')[-1,0,:,:]
            vely = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_y_field.npy')[-1,0,:,:]

            velx = torch.from_numpy(velx).cuda()
            vely = torch.from_numpy(vely).cuda()

            velx = velx[None, None, :, :]
            vely = vely[None, None, :, :]

            velocity_big = torch.cat((velx, vely), dim=1)

            tensor_U = math.wrap(velocity_big.squeeze(2), 'batch,vector,x,y')

            tensor_U_unstack = unstack_staggered_tensor(tensor_U)
            self.velocity =  StaggeredGrid(tensor_U_unstack, self.DOMAIN.bounds)

            try:
                velmaskx = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vel_mask_x_field.npy')[-1,0,:,:]
                velmasky = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vel_mask_y_field.npy')[-1,0,:,:]

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

            pfield = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_pressure_field.npy')[-1,:,:,:]
            self.pressure = CenteredGrid(tensor(torch.from_numpy(pfield).cuda(), names=['batch', 'x', 'y']), self.DOMAIN.bounds)

            self.density = CenteredGrid(tensor(torch.zeros((bsz, self.Nx, self.Ny)).cuda(), names=['batch', 'x', 'y']), self.DOMAIN.bounds)

        else:
            #Create the fields
            self.velocity = ((self.DOMAIN.staggered_grid(Noise(batch=bsz)) * 0 )+1) *(1,0)
            self.vel_mask = ((self.DOMAIN.staggered_grid(Noise(batch=bsz)) * 0 )+1)
            self.pressure = CenteredGrid(tensor(torch.zeros((bsz, self.Nx, self.Ny)), names=['batch', 'x', 'y']), self.DOMAIN.bounds)
            self.density = CenteredGrid(tensor(torch.zeros((bsz, self.Nx, self.Ny)), names=['batch', 'x', 'y']), self.DOMAIN.bounds)

        self.time_recorder.record(point_name='end_define_simulation_fields')

    def initialize_aux_variables(self):
        self.time_recorder.record(point_name='init_initialize_aux_variables')

        #Output Variables Initialization
        if self.resume:
            if self.post_computations:
                self.velocity_probe = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_probe.npy')
                self.vforce = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vforce.npy')
                self.hforce = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_hforce.npy')

            self.velocity_x_field=np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_x_field.npy').tolist()
            self.velocity_y_field=np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_y_field.npy').tolist()
            self.pressure_field=np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_pressure_field.npy').tolist()
            self.iteration_field=np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_iteration_field.npy').tolist()

            try:
                self.vel_mask_x_field=np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vel_mask_x_field.npy').tolist()
                self.vel_mask_y_field=np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vel_mask_y_field.npy').tolist()
            except:
                self.vel_mask_x_field=[]
                self.vel_mask_y_field=[]

        else:
            if self.post_computations:
                self.velocity_probe=np.zeros(self.Nt)
                self.vforce=np.zeros(self.Nt)
                self.hforce=np.zeros(self.Nt)

            self.velocity_x_field=[]
            self.velocity_y_field=[]
            self.pressure_field=[]
            self.vel_mask_x_field=[]
            self.vel_mask_y_field=[]
            self.iteration_field=[]

        if self.plot_field:
            self.gif_pressure = GIF(gifname=f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_pressure', total_frames=self.Nt)
            self.gif_vorticity = GIF(gifname=f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vorticity', total_frames=self.Nt)
            self.gif_velocity = GIF(gifname=f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity', total_frames=self.Nt)
            self.gif_distribution = GIF(gifname=f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_distribution', total_frames=self.Nt)

        self.bar = IncrementalBar(f' [RE={self.Re}, Nx={self.Nx}]', max=self.Nt, suffix= '%(percent)d%% [%(index)d/%(max)d] | %(eta_td)s remaining')

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

        elif self.sim_method == 'PHI':
            self.time_recorder.record(point_name=f'ite_{self.ite}_>init_poisson__PHI')

            self.cylinder = self.cylinder.copied_with(geometry=self.cylinder.geometry.rotated(-self.cylinder.angular_velocity * self.dt))
            self.velocity, self.pressure, self._iterations, self.div_in = fluid.make_incompressible(self.velocity, self.DOMAIN, (self.cylinder, ), pressure_guess=self.pressure,
            solve_params=math.LinearSolve(absolute_tolerance = self.precision, max_iterations = self.max_iterations ))
            self.div_out = divergence(self.velocity)

            self.time_recorder.record(point_name=f'ite_{self.ite}_>end_poisson__PHI')

        elif self.sim_method == 'convnet':
            self.time_recorder.record(point_name=f'ite_{self.ite}_>init_poisson__convnet')

            if self.ite<int(self.ite_transition):

                self.velocity, self.pressure, self._iterations, self.div_in, time_CG = fluid.make_incompressible_BC(self.velocity, self.DOMAIN, (), pressure_guess=self.pressure,
                solve_params=math.LinearSolve(absolute_tolerance = self.precision, max_iterations = self.max_iterations ), solver=self.sim_method)

                self.time_recorder.add_single_interval(time_CG, interval_name = f'ite_{self.ite}_>CG_inference_interval')

                self.div_out = divergence(self.velocity)
            else:
                in_density_t = self.density.values._native.transpose(-1, -2)
                in_U_t = torch.cat((self.velocity.staggered_tensor().tensors[0]._native.transpose(-1, -2).unsqueeze(1),
                            self.velocity.staggered_tensor().tensors[1]._native.transpose(-1, -2).unsqueeze(1)), dim=1)

                in_U_t[:,0, :, :2] = 1
                in_U_t[:,0, :, -2:] = 1

                data = torch.cat((in_density_t.unsqueeze(1).unsqueeze(1),
                            in_U_t[:,0,:-1,:-1].unsqueeze(1).unsqueeze(1),
                            in_U_t[:,1,:-1,:-1].unsqueeze(1).unsqueeze(1),
                            (self.flags+1),
                            in_density_t.unsqueeze(1).unsqueeze(1)), dim = 1)

                with torch.no_grad():
                    if self.new_train:
                        # Apply input/output BC
                        _, _, UDiv_CG = convert_phi_to_torch(self.velocity, self.pressure, self.pressure)
                        UDiv_CG = UDiv_CG.unsqueeze(2)
                        UDiv_CG[:, 0, :, :, -2:] = 1.0
                        UDiv_CG[:, 0, :, :, :2] = 1.0
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
                ['aux_contourn', True],
                ['indeces', False],
                ['grid', False]
                ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='pressure []',
                ltitle=f'VK @ t={np.round(self.dt*self.ite, decimals=1)} s [ A={self.alpha}, Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]')

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
                ltitle=f'VK @ t={np.round(self.dt*self.ite, decimals=1)} s [ A={self.alpha}, Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]',
                save=True, filename=f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_pressure_timestep_{self.ite}.png')

    def plot_vorticity(self, zoom_pos = []):
        vorticity = calculate_vorticity(self.Lx,self.Ly,self.dx,self.dy,self.velocity)

        if self.plot_field_gif:
            self.gif_vorticity.add_frame(self.ite, vorticity,
                plot_type=['surface'],
                options=[ ['limits', [-0.2, 0.5]],
                        ['full_zoom', False],
                        ['aux_contourn', True],
                        ['indeces', False],
                        ['grid', False]
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='vorticity []',
                ltitle=f'VK @ t={np.round(self.dt*self.ite, decimals=1)} s [ A={self.alpha}, Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]')

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
                ltitle=f'VK @ t={np.round(self.dt*self.ite, decimals=1)} s [ A={self.alpha}, Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]',
                save=True, filename=f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vorticity_timestep_{self.ite}.png')

    def plot_velocity_norm(self, zoom_pos = []):
        norm_velocity = calculate_norm_velocity(self.velocity)

        if self.plot_field_gif:
            self.gif_velocity.add_frame(self.ite, norm_velocity,
                plot_type=['surface'],
                options=[ ['limits', [0, 0.8]],
                        ['full_zoom', False],
                        ['aux_contourn', True],
                        ['indeces', False],
                        ['grid', False]
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y',lbar='norm velocity []',
                ltitle=f'VK @ t={np.round(self.dt*self.ite, decimals=1)} s [ A={self.alpha}, Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]')

    def plot_cp(self):
        pressure_ref = np.mean(self.pressure.values._native.cpu().numpy()[0][4,:])

        cp = calculate_cp(self.pressure, pressure_ref=pressure_ref, rho_ref=1, vel_ref=1)

        if self.plot_field_gif:
            self.gif_distribution.add_frame2(self.ite, cp, self.CYLINDER_2, plot_type=['full'],
                                        options=[['limits', [-2, 1.2] ]
                                        ],
                                        lx='angle', ly='pressure coeficient', ltitle=f'Cp @ t={np.round(self.dt*self.ite, decimals=1)} s [ A={self.alpha}, Re={self.Re}, N=[{self.Nx}x{self.Ny}] ]')

    def reconstruct_velocity_probe(self):

        xp1 = int((self.xD + self.D*2)/self.dx)
        xp2 = int((self.xD + self.D*2.5)/self.dx)
        yp1 = int((self.Ly/2 - self.D*0.25)/self.dy)
        yp2 = int((self.Ly/2 + self.D*0.25)/self.dy)


        print('calculate_velocity_probe>> read file')
        print('BE AWARE THAT ONLY THE SAVE ITERATIONS WILL BE CALCULATED-> THE OTHERS 0')
        velocity_y_field = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_y_field.npy')
        iteration_field = np.load(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_iteration_field.npy')

        velocity_probe=np.zeros(self.Nt)
        for i, ite in enumerate(iteration_field):
            velocity_probe[ite] = np.mean(velocity_y_field[i][0][xp1:xp2,yp1:yp2]) #to squeeze
        np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_probe.npy', velocity_probe)

    def calculate_velocity_probe(self):
        xp1 = int((self.xD + self.D*2)/self.dx)
        xp2 = int((self.xD + self.D*2.5)/self.dx)
        yp1 = int((self.Ly/2 - self.D*0.25)/self.dy)
        yp2 = int((self.Ly/2 + self.D*0.25)/self.dy)

        self.velocity_probe[self.ite] = np.mean(self.velocity.staggered_tensor().tensors[1]._native.cpu().squeeze().numpy()[xp1:xp2,yp1:yp2])

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
            ltitle=f'VK @ [ N=[{self.Nx}x{self.Ny}] ]',
            save=True, filename=f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_geometry.png')

    def save_variables(self):
        #3.1.SAVE INTERMIDATE RESULTS OF POST-PROCESS
        if self.post_computations and ( ( self.ite%self.save_post_x_ite == 0 if self.DEBUG else False) or self.ite == self.Nt-1 ):
            np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_probe.npy', self.velocity_probe)
            np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vforce.npy', self.vforce)
            np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_hforce.npy', self.hforce)

        #3.2.SAVE FIELDS
        if self.save_field and ( ( self.ite%self.save_field_x_ite == 0 if self.DEBUG else False ) or self.ite == self.Nt-1 ):
            np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_x_field.npy', self.velocity_x_field)
            np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_y_field.npy', self.velocity_y_field)
            np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_pressure_field.npy', self.pressure_field)
            np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vel_mask_x_field.npy', self.vel_mask_x_field)
            np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vel_mask_y_field.npy', self.vel_mask_y_field)
            np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_iteration_field.npy', self.iteration_field)

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
                #1.1.Diffuse Velocity
                if not self.Re_INF:
                    self.velocity_free = diffuse.explicit(self.velocity, self.viscosity, self.dt)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>diffuse')
                else:
                    print('DIFFUSION STEP AVOIDED SINCE RE >= 50000')
                    self.velocity_free = self.velocity

                #1.2.Advect Velocity
                self.velocity = advect.semi_lagrangian(self.velocity_free, self.velocity, self.dt)

                if torch.isnan(self.velocity.staggered_tensor().tensors[0]._native).any():
                    print('Nan in Domain')
                    self.velocity = change_nan_zero(self.velocity, self.DOMAIN)

                self.time_recorder.record(point_name=f'ite_{self.ite}_>advect')

                #1.3.Apply Boundary Conditions
                self.velocity = self.velocity * (1 - self.INFLOW) + self.INFLOW * (1, 0) + self.INIT*(0,0.5) if self.ite<int(self.INIT_transition) else self.velocity * (1 - self.INFLOW) + self.INFLOW * (1, 0)

                if self.sim_method == 'CG' or self.sim_method == 'convnet':
                    self.velocity = apply_boundaries(self.velocity, self.bc_mask, self.bc_value)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>apply_bc')

                #1.4.Solve Poisson Equation
                self.solve_poisson()

                #1.5.Reenforce Boundary Conditions
                if self.sim_method == 'CG' or self.sim_method == 'convnet':
                    self.velocity = apply_boundaries(self.velocity, self.bc_mask, self.bc_value)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>reinforce_bc')

            #2.POST-PROCESSING
            self.time_recorder.record(point_name=f'ite_{self.ite}_>init_post')
            if True: #try:
                if self.post_computations: # and self.ite%self.save_post_x_ite == 0:
                    #2.1.VELOCITY PROBE
                    self.calculate_velocity_probe()
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>probe')

                    #2.2.CALCULATE FORCES
                    self.hforce[self.ite], self.vforce[self.ite] = calculate_forces(self.pressure, self.CYLINDER_2, self.dx, self.dy)
                    #hforce[ite], vforce[ite] = calculate_forces_with_momentum(pressure, velocity, FORCES_MASK, factor=1, rho=1, dx=self.dx, dy=self.dy)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>forces')

                #2.3.PLOT RESULTS
                if self.plot_field and self.ite%self.plot_x_ite == 0:
                    zoom_pos=[self.xD - self.D, self.xD + self.D,
                            self.Ly/2 -self.D, self.Ly/2 + self.D]

                    self.plot_pressure(zoom_pos = zoom_pos)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>pressure')

                    self.plot_vorticity(zoom_pos = zoom_pos)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>vorticity')

                    self.plot_velocity_norm(zoom_pos = zoom_pos)
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>velocity')

                    self.plot_cp()
                    self.time_recorder.record(point_name=f'ite_{self.ite}_>cp')

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


            #3.SAVE RESULTS
            self.time_recorder.record(point_name=f'ite_{self.ite}_>init_save_results')
            if True:
                self.save_variables()

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
                    self.gif_velocity.build_gif()
                    self.gif_distribution.build_gif()

            if self.post_computations:
                self.plot_forces()
                self.plot_forces_ss()
        except:
            pass

        #FINAL SAVINGS
        try:
            self.save_variables()
        except:
            pass

        if self.resume:
            pass


        self.time_recorder.record(point_name='run_end')
        self.time_recorder.close(save=True)
