# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
import heapq 
from entity import CharacterEntity
from colorama import Fore, Back

class TestCharacter(CharacterEntity):

    def do(self, wrld):
        # Locate current position and exit coordinates
        me = wrld.me(self)
        start = (me.x, me.y)
        exit_pos = self.find_exit(wrld)

        # Run A* search to calculate path to exit
        path = self.a_star(wrld, start, exit_pos)

        # path and execute move
        if path and len(path) > 1:

            # Move to the immediate next step in the path
            next_step = path[1]
            dx = next_step[0] - me.x
            dy = next_step[1] - me.y
            self.move(dx, dy)
        else:
            # Stop if no valid path exists or exit is reached
            self.move(0, 0)

    def find_exit(self, wrld):
        """Scans the board to locate the exit cell."""
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.exit_at(x, y):
                    return (x, y)
        return None

    def heuristic(self, a, b):
        """Chebyshev distance for 8-neighborhood movement with step cost = 1."""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def is_traversable(self, wrld, x, y):
        """Checks if a tile is free of boundaries, walls, bombs, explosions, and monsters."""
        if x < 0 or x >= wrld.width() or y < 0 or y >= wrld.height():
            return False
        
        if wrld.wall_at(x, y):
            return False
        if wrld.bomb_at(x, y):
            return False
        if wrld.explosion_at(x, y):
            return False
        if wrld.monsters_at(x, y):
            return False

        return True

    def a_star(self, wrld, start, goal):
        """A* search algorithm returning a list of tuple coordinates from start to goal."""
        # Priority Queue stores: (f_score, g_score, current_node, path_history)
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start, goal), 0, start, [start]))
        
        g_scores = {start: 0}

        while open_set:
            f, g, current, path = heapq.heappop(open_set)

            if current == goal:
                return path

            if g > g_scores.get(current, float('inf')):
                continue

            cx, cy = current

            # Explore 8-neighborhood (diagonal and cardinal)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    neighbor = (cx + dx, cy + dy)

                    if not self.is_traversable(wrld, neighbor[0], neighbor[1]):
                        continue

                    tentative_g = g + 1

                    if tentative_g < g_scores.get(neighbor, float('inf')):
                        g_scores[neighbor] = tentative_g
                        f_score = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor, path + [neighbor]))
        return None