import re
import traceback

from colorama import init, Fore, Back, Style


CEND      = '\033[0m'
CBOLD     = '\33[1m'
CITALIC   = '\33[3m'
CURL      = '\33[4m'
CBLINK    = '\33[5m'
CBLINK2   = '\33[6m'
CSELECTED = '\33[7m'

CBLACK  = '\33[30m'
CRED    = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\033[34m'
CVIOLET = '\33[35m'
CBEIGE  = '\33[36m'
CWHITE  = '\33[37m'

CBLACKBG  = '\33[40m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG   = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG  = '\33[46m'
CWHITEBG  = '\33[47m'

CGREY    = '\33[90m'
CRED2    = '\33[91m'
CGREEN2  = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2   = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2  = '\33[96m'
CWHITE2  = '\33[97m'

CGREYBG    = '\33[100m'
CREDBG2    = '\33[101m'
CGREENBG2  = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2   = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2  = '\33[106m'
CWHITEBG2  = '\33[107m'



def info(msg:str, config=True):
    if config['INFO'] if isinstance(config,dict) else config:
        print('>>Fluidnet '+ CBLUE + '--INFO: ' + msg + CEND)


def debug(msg: dict or str, config=True):
    if config['DEBUG'] if isinstance(config,dict) else config:
        if isinstance(msg,dict):

            stack = traceback.extract_stack()
            filename, lineno, function_name, code = stack[-2]
            vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]

            print('>>Fluidnet ' + CGREEN + '--DEBUG: [file: ' + filename + ' line: ' + str(lineno) + ']')
            print('      ' + vars_name)
            for key,value in msg.items():
                print("       |-" + key + ": ", end='') 
                print(value)  #in case some bools, or others that cannot be concatenated
            print(CEND)
        elif isinstance(msg,str):
            print('>>Fluidnet ' + CGREEN + '--DEBUG: ' + msg + CEND)
        else:
            NotImplemented 

def warming():
    pass

def error():
    pass