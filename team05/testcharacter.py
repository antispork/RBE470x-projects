# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

import numpy
import math
# from collections import deque

class TestCharacter(CharacterEntity):
    def get_monsters(self, wrld):
        monsters = []
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                if wrld.monsters_at(c,r):
                    monsters.append((c,r))
        return monsters

    def get_bomb(self, wrld):
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                if wrld.bomb_at(c,r):
                    return (c,r)

    def get_explosions(self, wrld):
        explosions = []
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                if wrld.explosion_at(c,r):
                    explosions.append((c,r))
        return explosions

    def map_walls(self, wrld):
        wall_map = numpy.zeros((wrld.height(), wrld.width()))
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                wall_map[r, c] = numpy.nan if wrld.wall_at(c,r) else 0
        return wall_map

    def map_dist(self, wrld, p: tuple[int, int]):
        dist_map = self.map_walls(wrld)
        exit_x, exit_y = p
        # print(f'exit_pos={exit_x, exit_y}')
        dist_map[dist_map == 0] = numpy.inf
        dist_map[exit_y, exit_x] = 0

        # print(dist_map)

        queue = [(exit_y, exit_x)]
        while queue:
            current = queue.pop(0)
            value = dist_map[current]
            # print(f'cur:{current}')
            # print(f'val:{value}')
            # print(f'neighbors:{self.neighbors_of_8(current, wrld)}')
            for neighbor in self.neighbors_of_8(current, wrld):
                if dist_map[neighbor] > value + 1:
                    dist_map[neighbor] = value + 1
                    queue.append(neighbor)

        return dist_map

    def map_dist_exit(self, wrld):
        return self.map_dist(wrld, wrld.exitcell)

    def map_dist_monsters(self, wrld):
        combined = numpy.zeros((wrld.height(), wrld.width()))
        combined[combined == 0] = numpy.inf
        ps = self.get_monsters(wrld)
        for i, p in enumerate(ps):
            monster_dist_map = self.map_dist(wrld, p)
            # print(f'monster {i}: {monster_dist_map}')
            combined = numpy.minimum(combined, monster_dist_map)
        return combined

    def map_dist_bomb(self, wrld):
        return self.map_dist(wrld, self.get_bomb(wrld))

    def map_danger_monsters(self, wrld):
        combined = numpy.zeros((wrld.height(), wrld.width()))
        ps = self.get_monsters(wrld)
        for i, p in enumerate(ps):
            monster_danger_map = self.map_dist(wrld, p)
            # print(f'1:{monster_danger_map}')
            monster_danger_map[monster_danger_map <= 3] = 200
            # print(f'2:{monster_danger_map}')
            monster_danger_map[monster_danger_map < 200] = 0
            # print(f'3:{monster_danger_map}')
            combined = combined + monster_danger_map
            # print(f'4:{combined}')
        return combined

    def map_danger_bomb_explosion(self, wrld):
        combined = numpy.zeros((wrld.height(), wrld.width()))
        p_bomb = self.get_bomb(wrld)
        ps_explosions = self.get_explosions(wrld)
        if p_bomb != None:
            # print(p_bomb)
            center_r, center_c = p_bomb[1], p_bomb[0]
            radius = 4
            combined[p_bomb[1], :] = 100
            combined[:, p_bomb[0]] = 100
        for i, p in enumerate(ps_explosions):
            danger_map = self.map_dist(wrld, p)
            danger_map[danger_map <= 0] = numpy.inf
            danger_map[danger_map < numpy.inf] = 0
            combined = combined + danger_map
        return combined



    def map_square_value(self, wrld):
        exit_map = self.map_dist_exit(wrld)
        monsters_map = self.map_dist_monsters(wrld)
        danger_map = self.map_danger_monsters(wrld) + self.map_danger_bomb_explosion(wrld)
        return exit_map + danger_map + exit_map // numpy.sqrt(monsters_map)

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
                if not math.isnan(value):
                    neighbors.append((cy, cx))
        return neighbors

    def best_move(self, wrld):
        value_map = self.map_square_value(wrld)
        best_move = (self.x, self.y)
        for i, neighbor in enumerate(self.neighbors_of_8((self.y, self.x), wrld)):
            # print(f'curr best: {best_move}')
            # print(f'neighbor: {neighbor}, value: {value_map[neighbor[0], neighbor[1]]}')
            if value_map[neighbor[0], neighbor[1]] < value_map[best_move[1], best_move[0]]:
                best_move = (neighbor[1], neighbor[0])
        # print(f'final best: {best_move}')
        return (best_move[0] - self.x, best_move[1] - self.y)

    def close_monster(self, wrld):
        monster_map = self.map_dist_monsters(wrld)
        if monster_map[self.y, self.x] <= 4:
            return True
        return False



    def do(self, wrld):
        # Your code here
        all_accessible_vars = [
            attr for attr in dir(wrld)
            if not attr.startswith('__') and not callable(getattr(wrld, attr))
        ]
        # print(all_accessible_vars)
        # print(self.map_dist(wrld, wrld.exitcell))
        # print(self.map_dist(wrld, wrld.monster)
        print(f'monster locations: {self.get_monsters(wrld)}')
        print(f'monsters empty? {len(self.get_monsters(wrld)) == 0}')
        print(f'bomb location: {self.get_bomb(wrld)}')
        print(f'bomb empty? {self.get_bomb(wrld) == None}')
        print(f'explosion locations: {self.get_explosions(wrld)}')
        print(f'explosions empty? {len(self.get_explosions(wrld)) == 0}')
        # print(f'i am at {self.x, self.y}!')
        # print(f'neighbors: {self.neighbors_of_8((self.x, self.y), wrld)}')
        # print(self.map_dist_exit(wrld))
        # print(self.map_dist_monsters(wrld))
        print(self.map_square_value(wrld))
        # print(self.best_move(wrld))
        self.dx, self.dy = self.best_move(wrld)
        print(self.dx, self.dy)

        # print(f'm_dangermap:\n {self.map_danger_monsters(wrld)}')
        # print(f'b_dangermap:\n {self.map_danger_bomb_explosion(wrld)}')
        if self.get_bomb(wrld) == None and len(self.get_explosions(wrld)) == 0 and len(self.get_monsters(wrld)) > 0 and (self.close_monster(wrld) or (self.dx, self.dy) == (0,0)):
            self.place_bomb()
        self.move(self.dx, self.dy)

        
        pass
