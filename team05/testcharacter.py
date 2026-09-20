# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

import numpy
# from collections import deque

class TestCharacter(CharacterEntity):

    def map_walls(self, wrld):
        wall_map = numpy.zeros((wrld.height(), wrld.width()))
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                wall_map[r, c] = - wrld.wall_at(c,r)
        return wall_map

    def map_dist(self, wrld):
        dist_map = self.map_walls(wrld)
        exit_x, exit_y = wrld.exitcell
        print(f'exit_pos={exit_x, exit_y}')
        dist_map[dist_map == 0] = numpy.inf
        dist_map[exit_y, exit_x] = 0

        print(dist_map)

        queue = [(exit_y, exit_x)]
        while queue:
            current = queue.pop(0)
            print(f'cur:{current}')
            value = dist_map[current]
            print(f'val:{value}')
            print(f'neighbors:{self.neighbors_of_8(current, wrld)}')
            for neighbor in self.neighbors_of_8(current, wrld):
                if dist_map[neighbor] > value + 1:
                    dist_map[neighbor] = value + 1
                    queue.append(neighbor)

        return dist_map

    def neighbors_of_8(self, p: tuple[int, int], wrld) -> list[tuple[int, int]]:
        """
        Returns the safe 8-neighbors cells of (x,y) in the occupancy grid.
        :param p       [(int, int)]    The coordinate in the grid.
        :return        [[(int,int)]]   A list of walkable in 4 cardinal directions.
        """
        wall_map = self.map_walls(wrld)
        y, x = p
        # print(y, x)
        height = wrld.height()
        width = wrld.width()
        # print(wrld.height(), wrld.width())
        
        candidates = [
            (y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1),
            (y + 1, x + 1), (y + 1, x - 1), (y - 1, x + 1), (y - 1, x - 1),
        ]
        # print(candidates)

        neighbors = []
        for cy, cx in candidates:
            if 0 <= cy < height and 0 <= cx < width:
                value = wall_map[cy, cx]
                if value != -1:
                    neighbors.append((cy, cx))
        return neighbors



    def do(self, wrld):
        # Your code here
        all_accessible_vars = [
            attr for attr in dir(wrld)
            if not attr.startswith('__') and not callable(getattr(wrld, attr))
        ]
        print(all_accessible_vars)
        self.move(1,1)
        print(self.map_dist(wrld))
        
        pass
