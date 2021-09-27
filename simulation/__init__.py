#from os import scandir


#for arch in scandir('.'):
 #   if arch.is_file() and arch.name is not '__init__.py':
 #       from f'.{arch.name[:-3]}' import *


from .Simulation import *
from .WindTunnel import *
from .VonKarman import *
from .TaylorCouette import *
from .Plume import *