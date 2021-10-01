from neurasim import *

def main():
    parser = AnalyzeParser()
    config = parser.parse()

    class_name = config['simClass']
    param = config

    sim = globals()[class_name](param)


    for i,atype in enumerate(config['analysis_type']):

        #SINGLE
        if atype == 'all':
            sim.plot_velocity_probe()
            sim.plot_power_spectrum()
            sim.plot_forces()
            sim.plot_forces_ss()


        if atype == 'spectrum':
            sim.plot_power_spectrum()
        if atype == 'velocity':
            sim.plot_velocity_probe()
        if atype == 'velocity_probe':
            sim.reconstruct_velocity_probe()
        if atype == 'forces':
            sim.plot_forces()
            sim.plot_forces_ss()
            sim.print_mean_forces()
        if atype == 'geoemtry':
            NotImplemented

        if atype == 'snaps':
            sim.plot_snaps()


if __name__ == '__main__':
    main()