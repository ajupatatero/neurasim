from neurasim import *

def main():
    parser = SimulateParser()
    config = parser.parse()

    class_name = config['simClass']
    param = config

    sim = globals()[class_name](param)
    sim.run()

if __name__ == '__main__':
    main()