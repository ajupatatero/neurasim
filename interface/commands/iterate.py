from neurasim import *
import multiprocessing

#TODO: consider updating terminal output with curses and pack all function in /terminal/

def call_script(execute,PARSER):
    #Prepare Arguments to Parse
    txt=""
    for i in range(0,len(PARSER)-1,2):
        txt = txt + " -{0} {1}".format(PARSER[i],PARSER[i+1])
                
    print(f'Executing' + "\033[1;34;40m" + f' {str(execute)}' + '\033[0;m' + ' with ' + "\033[1;34;40m" +
          f'{PARSER}' + '\033[0;m' + f' || @ process \033[1;35;40m{os.getpid()}\033[0;m launched by \033[1;35;40m{os.getppid()}\033[0;m')
    os.system(f'{execute}' + txt)


def for_recursive(execute, constant_list, iterable_list, depth_index=0, iteration_index=[]):
    number_of_loops = len(iterable_list) #np.shape(iterable_list)[0] this one depreciated
    
    if depth_index == number_of_loops:  #Base Case
        #create subcase iterable values and properties name parser list
        PARSER = []
        for _, value in enumerate(constant_list):
            PARSER = PARSER + [value[0]] + [value[1]]
        for depth_index, value in enumerate(iterable_list):
            PARSER = PARSER + [value[0]] + [value[1][iteration_index[depth_index]]]

        #parse properties to simulate
        try:
            if callable(execute):
                #pass arguments treated by the parser to the function diretly
                #parser = SimulationParser()
                #args = parser.parse_args()
                #EXECUTE(**vars(args))

                exit()  #now the commands are scripts
            else:
                #TODO: divide iterables into the available gpu or short partition allocated. And for each group, make a launch command that creates 
                # yaml file with the sub set of iterables and call iterate another time. This time, however, with a parameter to indicate not to 
                # divide into multiple partitions. Then what it does, is create multiple threads for each of the sub iterable list.

                #Therefore, the user will call directly iterate with parameter saying = multiplelaunches... and this will create multiples yaml and sbatch files.
                # which will at turn call iterate with that argument as false. 

                #if multi_launch:

                #else:

                #Only in this partition, Notice then, that in panod have to use sbatch not iterate directly!


                #MultiProcessing
                multiprocessing.Process(target=call_script, args=(execute, PARSER)).start()

                #TODO: next step is doing multy thread on the very simulation Maybe not possible? only for secondary task??
                #Threads are very good with input/output operation. better than proces.-> use multi thread for saving files

                #TODO: performance analysys to compare

        except:
            print("\033[1;31"+f'ERROR: case failed with an uncontaplated error in {str(EXECUTE)}' +'\033[0;m')

    else: #Recursive Case
        for iter_index, iter_value in enumerate(iterable_list[depth_index][1]):
            for_recursive(execute=execute, constant_list=constant_list, iterable_list= iterable_list, depth_index = depth_index+1, iteration_index= iteration_index + [iter_index] if depth_index >= 1 else [iter_index])


def main():
    '''Function to run a meta analysis over a set of iterable parameters. Notice ther eis no limitation to 
    the number of parameters to iterate. 

    usage: meta_simulate(CONSTANT,ITERABLE,EXECUTE)

    -CONSTANT = [['Lx', 150],
            ['Lx', 50],
            ['Nt', 10],
            ['gpu', True]    !! The bool arguments are only the -gpu the other part doesn't matter!! 
        ]

    -ITERABLE = [ ['variable name to pass to simulate', [1.5, 2.5, 3.0]],
               ['variable 2', [150, 450]], 
               ['variable 3', [50, 100, 200, 300, 400, 500, 700, 850, 1000]], 
               ['range input', range(0,5,1) ],
               ['array input', np.linspace(0,5,3)]    
            ]
    
    -EXECUTE = function name or script.py with path

    Notice: 
    1) All the variables of the simulation that depend on iterable parameters, include its calculation within 
    the simulate function or script.
    2) The iterable parameters input can be a list, an array, or a range type variable
    
    '''
    
    parser = IterateParser()
    config = parser.parse()

    try:
        CONSTANT = config['CONSTANT']
        ITERABLE = config['ITERABLE']
        EXECUTE = config['EXECUTE']
    except:
        print('No sufficient data in config_simulation.yaml as to execute an iteration.')
        exit()


    print("\033[2J\033[1;1f") # Borrar pantalla y situar cursor
    print("\033[1;35;47m"+"  NEURASIM (meta command launcher) "+'\033[0;m') 
    
    #1.Correct the inputs that are not lists
    for i, ite in enumerate(ITERABLE):
        if ite[1] is not list:
            ITERABLE[i][1] = list(ite[1])

    #2.Launch recursive fors corresponding with each subcase
    for_recursive(execute=EXECUTE, constant_list=CONSTANT, iterable_list = ITERABLE)

if __name__ == '__main__':
    main()