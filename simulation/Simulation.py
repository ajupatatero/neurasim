from neurasim import *
import importlib.util
import warnings
import os

class Simulation():
    '''Higher hierarchy simulation type class of Neurasim software. All other simulation classes must depend on this one. '''

    def __init__(self, config):
        '''Constructor of the Simulation basic and general class. Within you can find all the options and paramaters
        common to any kind of simulation type. Besides, in here, you can define its default values in case not specified in the config.yaml.
        '''

        #############################
        #   INPUT/OUTPUT PROPERTIES #
        #############################
        self.in_dir=config['in_dir']
        self.out_dir=config['out_dir']

        # If doesn't exist, create it!
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)


        #############################
        #   RUNNING OPTIONS         #
        #############################
        #RUNNING MODES
        try:
            self.DEBUG = config['DEBUG']
        except:
            self.DEBUG = False

        #VERBAROSE OPTIONS
        warnings.filterwarnings("ignore") #To disable noisy warnings.

        #MAX TIME OF RUNNING BEFORE RELAUNCHING
        try:
            self.MAX_TIME = config['MAX_TIME']
        except:
            self.MAX_TIME = 23*60*60*1000

        #EXTENDED MAX-TIME SIMULATION
        self.config_copy = config
        try:
            self.resume = config['resume']
        except:
            self.resume = False

        #GPU RUNNING vs CPU:
        try:
            self.GPU = config['GPU']
        except:
            self.GPU = False

        #FLOATING POINTS PRECISION
        try:
            self.FP64 = config['FP64']
        except:
            self.FP64 = False

        #Min iteration to start saving
        try:
            self.min_ite = config['min_ite']
        except:
            self.min_ite = 0

        #SAVE_PERFORMANCE: Option to calculate and save the performance
        try:
            self.save_performance = config['save_performance']
        except:
            self.save_performance = False

        #SAVE_FIELD: Option to save the fields
        try:
            self.save_field = config['save_field']
        except:
            self.save_field = False

        #save the fields each x iterations
        try:
            self.save_field_x_ite = int(float(config['save_field_x_ite']))
        except:
            self.save_field_x_ite = 200

        #PLOT_FIELD: Option to plot fields during execution of run
        try:
            self.plot_field = config['plot_field']
        except:
            self.plot_field = True

        #plot and save the plots each x iterations
        try:
            self.plot_x_ite = int(float(config['plot_x_ite']))
        except:
            self.plot_x_ite = 10

        if self.plot_field:
            #If plot activated, PLOT GIF?
            try:
                self.plot_field_gif = config['plot_field_gif']
            except:
                self.plot_field_gif = True

            #If plot activated, plot timesteps?
            try:
                self.plot_field_steps = config['plot_field_steps']
            except:
                self.plot_field_steps = True

        #POST_COMPUTATIONS: Option to perform the post calculations regarding forces, and probes, etc
        try:
            self.post_computations = config['post_computations']
        except:
            self.post_computations = True

        #Save the post computation calculations variables each x iterations
        try:
            self.save_post_x_ite = int(float(config['save_post_x_ite']))
        except:
            self.save_post_x_ite = 100


        #############################
        #   SIMULATION OPTIONS      #
        #############################
        self.sim_method = config['sim_method']

        if self.sim_method == 'convnet':
            self.load_path = config['network_params']['load_path']
            self.network_name = config['network_params']['model_name']
            self.new_train= config['network_params']['new_train'] == 'new'

            if self.new_train:
                self.config_norm = config['normalization']

            try:
                self.ite_transition = config['ite_transition']
            except:
                self.ite_transition=0

        #TOLERANCE OR PRECISION OF THE LAPLACE MATRIX SOLVER (used in all solvers. i.e. in convnet before ite transition!)
        try:
            self.precision = int(float(config['precision']))
        except:
            self.precision = 1e-3

        #MAX NUMBER OF ITERATIONS OF THE CG METHOD TO SOLVE THE LAPLACE EQUATION
        try:
            self.max_iterations = int(float(config['max_iterations']))
        except:
            self.max_iterations = 1e5


        ################################
        #   DISCRETITATION & GEOMETRY  #
        ################################
        self.Lx=config['Lx']
        self.Ly=config['Ly']

        self.Nx=config['Nx']
        self.Ny=config['Ny']
        self.Nt=config['Nt']
        self.CFL = config['CFL']

        self.dx=self.Lx/self.Nx
        self.dy=self.Ly/self.Ny
        self.dt= self.dx * self.CFL

    def run(self):
        print('This general class has no running scheme defined. You need to use a lower level class.')
        error()

    def prepare_resume(self):
        '''Function that prepares all the files and folders to stop the current instance of the simulation run.
        And to start a new one following the overall simulation. Meaning, it saves the basic simulation fields.
        It creates the temporal folders and slurm launch files. And send a launch on slurm as well.


        NOTICE: in pando supercomputer (ISAE-SUPAERO) has been notice that the folders have to be created manually before.
        '''

        #1.Save simulation fields
        self.velocity_x_field.append(self.velocity.staggered_tensor().tensors[0]._native.cpu().numpy())
        self.velocity_y_field.append(self.velocity.staggered_tensor().tensors[1]._native.cpu().numpy())
        self.pressure_field.append(self.pressure.values._native.cpu().numpy())
        self.vel_mask_x_field.append(self.vel_mask.staggered_tensor().tensors[0]._native.cpu().numpy())
        self.vel_mask_y_field.append(self.vel_mask.staggered_tensor().tensors[1]._native.cpu().numpy())
        self.iteration_field.append(self.ite)

        np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_y_field.npy', self.velocity_y_field)
        np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_pressure_field.npy', self.pressure_field)
        np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vel_mask_x_field.npy', self.vel_mask_x_field)
        np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_vel_mask_y_field.npy', self.vel_mask_y_field)
        np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_velocity_x_field.npy', self.velocity_x_field )
        np.save(f'{self.out_dir}A_{self.alpha}_RE_{self.Re}_dx_{self.Nx}_{self.Ny}_iteration_field.npy', self.iteration_field)

        #2.1.Create temporal folder
        if not os.path.isdir('./temporal_resume/'):
            os.makedirs('./temporal_resume/') #NOTICE: in pando before manually!!!

        #2.2.Create new config file
        with open(f'./temporal_resume/config_simulation.yaml', 'w') as file:
            self.config_copy['resume']=True
            try:
                self.config_copy['resume_count'] = self.config_copy['resume_count'] + 1
            except:
                self.config_copy['resume_count'] = 1

            yaml.dump(self.config_copy, file)

        #2.3.Create new launch file
        with open("./temporal_resume/launcher.sh", "w") as file:
            file.write("#!/bin/bash\n\n\n")

            file.write("#SBATCH --gres=gpu:1\n")
            file.write("#SBATCH --partition=gpu\n")
            file.write("#SBATCH --nodes=1\n")
            file.write("#SBATCH --ntasks-per-node=4\n")
            file.write("#SBATCH --time=24:00:00\n")
            file.write("#SBATCH --begin=now\n")
            file.write(f"#SBATCH --job-name=+{self.config_copy['resume_count'] * 24}h_extension\n")
            file.write("#SBATCH --output=./results/slurm.%j.out\n")
            file.write("#SBATCH --error=./results/slurm.%j.err\n\n\n")

            file.write("source $HOME/fluidnet_env/bin/activate\n")
            file.write("simulate --conf_dir './temporal_resume/'\n")

            file.close()

        #3.Launch new job
        os.system('sbatch ./temporal_resume/launcher.sh')

    def load_model(self):

        mpath = glob.os.path.join(self.load_path, 'convModel_lastEpoch_best.pth')
        assert glob.os.path.isfile(mpath), mpath  + ' does not exits!'
        cpath = glob.os.path.join(self.load_path, 'convModel_conf.pth')
        mcpath = glob.os.path.join(self.load_path, 'convModel_mconf.pth')

        if self.GPU:
            state = torch.load(mpath)
            self.conf = torch.load(cpath)
            self.mconf = torch.load(mcpath)
        else:
            state = torch.load(mpath, map_location=torch.device('cpu'))
            self.conf = torch.load(cpath, map_location=torch.device('cpu'))
            self.mconf = torch.load(mcpath, map_location=torch.device('cpu'))


        temp_model = glob.os.path.join(self.load_path, self.network_name +  '_saved.py')
        assert glob.os.path.isfile(temp_model), temp_model  + ' does not exits!'

        spec = importlib.util.spec_from_file_location('model_saved', temp_model)
        model_saved = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_saved)

        if self.new_train:
            self.model = model_saved.PhiflowNet(self.mconf, 0, self.load_path)
        else:
            self.model = model_saved.FluidNet(self.mconf, 0, self.load_path)
        if self.GPU:
            self.model.cuda()

        self.model.load_state_dict(state['state_dict'])

        self.model.eval()
