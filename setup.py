from setuptools import setup, find_packages

setup(
    name='NeuraSim',
    version='3.2.1',
    description='CNN to accelerate Poisson step in CFD solvers.',
    author='Anonymus',
    license='LICENSE.txt',

    packages=find_packages(include=['NeuraSim']),   
    install_requires=[
        "numpy", "scipy", "matplotlib", "imageio", 
    ],

    #Include shell scripts realted to pando ....
    #scripts=['interface/commands/simulate.py',
    #         'interface/commands/train.py',
    #         'interface/commands/analyze.py',
    #         'interface/commands/mgit.py',
    #         'interface/commands/update.py'
    #],   #only on linux, windows not

    entry_points={
        'console_scripts': [
            'simulate=interface.commands.simulate:main',
            'train=interface.commands.train:main',
            'analyze=interface.commands.analyze:main',
            'meta_analyze=interface.commands.meta_analyze:main',
            'update=interface.commands.update:main',
            'mgit=interface.commands.mgit:main',
            'launch=interface.commands.launch:main',
            'iterate=interface.commands.iterate:main'
        ],
    }
)
