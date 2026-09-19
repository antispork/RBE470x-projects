# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

import numpy

class TestCharacter(CharacterEntity):
    def map_walls(self, wrld):
        wall_map = numpy.zeros((wrld.height(), wrld.width()))
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                wall_map[r, c] = wrld.wall_at(c,r)
        return wall_map



    def do(self, wrld):
        # Your code here
        all_accessible_vars = [
            attr for attr in dir(self)
            if not attr.startswith('__') and not callable(getattr(self, attr))
        ]
        print(all_accessible_vars)

        print(self.map_walls(wrld))
        
        self.move(0,1)

        print(self.dy)
        pass
