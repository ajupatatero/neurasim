import argparse
import yaml
import glob

class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

class InputParser():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class= SmartFormatter, add_help=False)
        self.parser.add_argument("-cd", "--conf_dir", default=glob.os.path.join('./', 'case/'), help="Path of the configuration yaml files directory.")

        #If some parameters are always mandatory, they can be added here, so that there are always passed as default, even when the user doesn't define 
        #them  in the config file or as arg input. At the same time, if you want to use the parser without a yaml file, you have to define the parameters
        #in here, and stablish a default and a TYPE is always a good practice to avoid errors.

        #help, default, type, action (storetrue), choices


        #TODO: hacer que la definicion sea externa a este documento, asi mas generico y no tienes que entrar aqui
        #porque pueden haber ciertos parametros que depenguan solo de la simulacion como el alpha. Sino, hacer un parametro argument
        #addicional que contemple los kwargs no incluidos en parser o directamente como stackoverflow de tratar los argum con key none
        #esto se haria en input proces y asi aqui definir los genericos, y si alguno concreto sim parsear directamente sin hacer nada y coger-lo del config
        #en el fichero sim_main.py


        #SIMULATION
        self.parser.add_argument("-gpu", default=False, type= bool)


        #GEOMETRY
        self.parser.add_argument("-Lx", default=1, type=int)
        self.parser.add_argument("-Ly", default=1, type=int)
        self.parser.add_argument("-Lt", default=1, type=int)

        #FORCES
        self.parser.add_argument("-Re", "--Reynolds", default=1, type=float)
        self.parser.add_argument("-A", "--Alpha", default=0, type=float) #Alpha: 0 #V0/w*r

        self.parser.add_argument("-Nx", default=3, type=int)  #Nx: 100 #[] number of control volumes in x direction
        self.parser.add_argument("-Ny", default=3, type=int) #Ny: 100 #[] number of control volumes in y direction
        self.parser.add_argument("-Nt", default=1, type=int) #Nt: 2 #[] number of time steps to simulate




        self.arguments = None
        self.config = {}

    def parse(self):
        self.arguments = self.parser.parse_args() #(['--hello','test']) to test
        self.load_yaml()
        self.proces_input()
        self.check_input_consistency()
        self.check_input_set()
        return self.config

    def print_input_arguments(self):
        print("Input arguments provided: \n")
        for key,value in vars(self.arguments).items():
            print("arg: " + key + ", value: "+value+"\n")
            
    def load_yaml(self):
        #Load Configuration Parameters
        try:
            with open(self.arguments.conf_dir+"config_simulation.yaml",'r') as config_file:
                config_simulation = yaml.load(config_file, Loader=yaml.FullLoader)
            with open(self.arguments.conf_dir+"config_physics.yaml",'r') as config_file:
                config_physics = yaml.load(config_file, Loader=yaml.FullLoader)
            with open(self.arguments.conf_dir+"config_numerical.yaml",'r') as config_file:
                config_numerical = yaml.load(config_file, Loader=yaml.FullLoader)
            with open(self.arguments.conf_dir+"config_train.yaml",'r') as config_file:
                config_train = yaml.load(config_file, Loader=yaml.FullLoader)
            self.config = config_simulation | config_physics | config_numerical | config_train
        except:
            print('No yaml file founded. Only parsed arguments.')

    def proces_input(self):
        #Priority criteria for selection: 1)input parser  2)yaml  3)parser default
        for key,value in vars(self.arguments).items():
            
            #TODO: simplify grammar with simConf['modelDir'] = arguments.modelDir or simConf['modelDir']

            if value is not None and value is not self.parser.get_default(key):
                #case 1
                self.config[key] = value
            elif ( value is None or value == self.parser.get_default(key) ) and (self.config[key] is not None if key in self.config else False):
                #case 2
                #no need to do any operation
                pass
            elif value is not None: # and (self.config[key] is None if key in self.config else True):    because of elif order of execution
                #case 3
                self.config[key] = self.parser.get_default(key)
            else:
                print("\nFATAL ERROR:\n Input Parser: Input key without any value assigned")
                print(key +" " +value)
                exit()
            
    def check_input_consistency(self):
        '''
        Function to check the coherence and consistency of the inputs. For instance, if the files and folders exist.
        Or if the different paraemeters don't contradict between themselfs.
        '''

        #Notice: This is only for general parameters. If there is some particular coherence check for a subclass. Specify them within it.

        #Check if folders and files exists
        #assert glob.os.path.isfile(restart_config_file), 'YAML config file does not exists for restarting.'
        #assert (glob.os.path.exists(simConf['modelDir'])), 'Directory ' + str(simConf['modelDir']) + ' does not exists'

        #Check if Nx,dx,Lx are coherent

        #etc
        

    def check_input_set(self):
        '''
        Function to check that all the parameters necessary for a given simulation have been passed.
        
        TODO: since all the required parameters should be provided in the parser as default this function will be 
        eventually removed. But for the moment, just in case is conserved.
        
        '''
        pass



class SimulationParser(InputParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], description='Simulation parser.', epilog="---")
        
        #Add extra arguments needed for the simulation

    def check_input_consistency(self):
        super().check_input_consistency()

        #Additional consistency checks for the specific parameters.


class TrainingParser(InputParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], description='Training parser.', epilog="---")
        
        #Add extra arguments needed for the training

    def check_input_consistency(self):
        super().check_input_consistency()

        #Additional consistency checks for the specific parameters.
