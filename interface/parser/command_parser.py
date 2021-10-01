import argparse
import yaml
import glob

class CommandParser():

    def __init__(self):
        self.parser = argparse.ArgumentParser(add_help=False)
        self.parser.add_argument("-cd", "--conf_dir", default='./', help="Path of the configuration yaml files directory.")
        self.parser.add_argument("-od", "--out_dir", default='./results/', help="Path of the output results files directory.")
        self.parser.add_argument("-id", "--in_dir", default='./results/', help="Path of the input files directory.")

        self.arguments = None
        self.config = { }

    def parse(self):
        self.arguments = self.parser.parse_args() #(['--hello','test']) to test
        self.load_yaml()
        self.proces_input()

        return self.config

    def load_yaml(self):
        try:
            if not 'yaml' in self.arguments.conf_dir:
                with open(self.arguments.conf_dir+"config_simulation.yaml",'r') as config_file:
                    self.config = yaml.load(config_file, Loader=yaml.FullLoader)
            else:
                with open(self.arguments.conf_dir,'r') as config_file:
                    self.config = yaml.load(config_file, Loader=yaml.FullLoader)
        except:
            print('No yaml file founded.')

    def proces_input(self):
        #Priority criteria for selection: 1)input parser  2)yaml  3)parser default
        for key,value in vars(self.arguments).items():

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

class SimulateParser(CommandParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], description='Simulate parser.', epilog="---")

        #This are arguments that can be used to pass directly, but specially thought for iteration
        #SIMULATION
        self.parser.add_argument("-gpu", default=False, type= bool)

        #GEOMETRY
        self.parser.add_argument("-Lx", default=1, type=int)
        self.parser.add_argument("-Ly", default=1, type=int)
        self.parser.add_argument("-Lt", default=1, type=int)

        self.parser.add_argument("-Nx", default=3, type=int)
        self.parser.add_argument("-Ny", default=3, type=int)
        self.parser.add_argument("-Nt", default=1, type=int)

        #FORCES
        self.parser.add_argument("-Re", "--Reynolds", default=1, type=float)
        self.parser.add_argument("-A", "--Alpha", default=0, type=float)

class AnalyzeParser(CommandParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], description='This command can be used to execute all kind of post-processing operations. As well as any type of analysis available.', epilog="  Neurasim 2021. All rights reserved.", add_help=True)

        #TYPE OF ANALYSIS
        self.parser.add_argument('-a','--analysis_type', action='append', default=[], help='To specify the post-proc function to call. You can append as many -a as needed.', required=True)

class IterateParser(CommandParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], description='Iterate parser.', epilog="---")
        self.parser.add_argument("-e", "--EXECUTE", default='simulate')


class LaunchParser(CommandParser):
    def __init__(self):
        super().__init__()
        self.parsers =  argparse.ArgumentParser(parents=[self.parser], description='Launch parser.', epilog="---")
        self.parser.add_argument("-c", "--command", default='simulate', help="Command to launch.")

    def parse(self):
        self.arguments = self.parser.parse_args() #(['--hello','test']) to test
        self.load_yaml()

        return self.argument.command, self.config

    def load_yaml(self):
        try:
            with open(self.arguments.conf_dir+"config_pando.yaml",'r') as config_file:
                self.config = yaml.load(config_file, Loader=yaml.FullLoader)
        except:
            print('No yaml file founded.')

class TrainParser(CommandParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], description='Simulation parser.', epilog="---")

    def load_yaml(self):
        try:
            with open(self.arguments.conf_dir+"config_train.yaml",'r') as config_file:
                self.config = yaml.load(config_file, Loader=yaml.FullLoader)
        except:
            print('No yaml file founded.')




class GitParser(CommandParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], description='Simulation parser.', epilog="---")
class UpdateParser(CommandParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], description='Simulation parser.', epilog="---")

class CopycaseParser(CommandParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], description='Simulation parser.', epilog="---")

class NewcaseParser(CommandParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], description='Simulation parser.', epilog="---")
