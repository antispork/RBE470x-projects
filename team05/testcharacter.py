
#  The charcter is organised as a STATE MACHINE with four states:

#      SEEK_EXIT   no threat in sight -> walk the A* path to the exit
#      EVADE       a monster / bomb / explosion is close -> expectimax search
#      BOMB        the exit is unreachable (or we are cornered) -> drop a bomb
#      HIDE        one of our bombs is ticking -> stay out of the blast cross

#  The pathing is done by two algorithms:

#      * A* for global navigation
#      * Expectimax over (SensedWorld.next()) for local decisions around monsters and bombs.


import sys
sys.path.insert(0, '../../Bomberman')

import heapq
import random

from entity import CharacterEntity
from sensed_world import SensedWorld
from events import Event
from colorama import Fore, Back



THREAT_RADIUS   = 5      # distance where it switch to expectimax
ADVERSARIAL_R   = 3      # within this distance a monster is treated as hostile
RISK_LAMBDA     = 0.6    # blend: value = L*worst_case + (1-L)*average
DEEP_RADIUS     = 3      # search one ply deeper when a monster is this close
CLEARANCE_BONUS = 200.0  # tie-break bonus for ending clear of a monster's next cell
SEARCH_DEPTH    = 2      # plies of look-ahead (1 ply = we move + monsters move)
BRANCH_LIMIT    = 5      # max monster moves expanded per chance node
MAX_ACTIVE      = 1      # branch at most this many monsters per ply (see note below)

W_EXIT          = 12.0   # weight on "get closer to the exit"
W_MONSTER       = 30.0   # weight on "stay away from monsters"
MONSTER_CAP     = 4      # monster distance is clipped here (far is far enough)
W_ESCAPE        = 40.0   # reward for having a real retreat path from monsters
ESCAPE_DEPTH    = 3      # how many free steps away from danger counts as "safe"
DEATH_SCORE     = -1e6   # dying
WIN_SCORE       =  1e6   # reaching the exit
BLAST_PENALTY   = -450.0 # standing in a cell that a bomb is about to cover
STUCK_LIMIT     = 12     # turns without progress before we consider bombing


# 8-connected move set.  (0,0) is a legal action: waiting is often the single
# best thing to do while a bomb is ticking or a monster is walking past.
MOVES = [(dx, dy)
         for dx in (-1, 0, 1)
         for dy in (-1, 0, 1)]


class MyCharacter(CharacterEntity):
    """Our Bomberman agent."""


    def __init__(self, name, avatar, x, y, depth=None):
        super().__init__(name, avatar, x, y)
        self.depth        = depth if depth is not None else SEARCH_DEPTH      
        self.state        = "SEEK_EXIT"  # current state-machine state
        self.bomb_pos     = None       # where we drop bomb
        self.best_dist    = None       # best exit distance seen so far
        self.stuck        = 0          # turns since best_dist improved
        self.path         = []         # A* queue

    
    #  Called once per turn       
    
    def do(self, wrld):
        me = wrld.me(self)
        if me is None:                 # we are dead / already out; nothing to do
            return

        # 1. Perception 
        self.dist_to_exit = self.exit_distance_map(wrld)   # BFS from the exit
        self.blast        = self.blast_timing(wrld)        # cell -> ticks to fire
        monsters          = self.monster_list(wrld)

        my_d   = self.dist_to_exit.get((me.x, me.y))
        threat = min([self.cheb(me.x, me.y, m.x, m.y) for m in monsters],
                     default=99)

        # 2. Progress bookkeeping (used to detect being stuck) 
        if my_d is not None and (self.best_dist is None or my_d < self.best_dist):
            self.best_dist = my_d
            self.stuck = 0
        else:
            self.stuck += 1

        # 3. State selection 
        # Our own bomb is on the board -> HIDE has priority over everything.
        if self.our_bomb(wrld) is not None or wrld.explosions:
            self.state = "HIDE"
        elif my_d is None or (self.stuck > STUCK_LIMIT and threat > 2):
            # The exit is walled off, or we have made no progress for a long
            # time: blow a wall open.
            self.state = "BOMB"
        elif threat <= THREAT_RADIUS:
            self.state = "EVADE"
        else:
            self.state = "SEEK_EXIT"

        self.paint_debug(wrld)        # colour the terminal view (harmless)

        # 4. Act 
        self.eff_depth = self.depth + 1 if threat <= DEEP_RADIUS else self.depth

        if self.state == "SEEK_EXIT":
            self.act_seek_exit(wrld, me)
        elif self.state == "BOMB":
            self.act_bomb(wrld, me)
        else:                          # EVADE and HIDE both use the search
            self.act_search(wrld, me)

    
    #  STATE: Look for exit                                                  
    
    def act_seek_exit(self, wrld, me):
        """Walk the A* path to the exit. No monster is close, so this is safe."""
        path = self.astar(wrld, (me.x, me.y), wrld.exitcell)
        if len(path) >= 2:
            nx, ny = path[1]
            self.path = path
            self.move(nx - me.x, ny - me.y)
        else:
            # No path (shouldn't happen here) -- fall back to the search.
            self.act_search(wrld, me)

    
    #  STATE: Bomb                                                       
    
    def act_bomb(self, wrld, me):
        """Drop a bomb to open a wall, then immediately start running away."""
        self.place_bomb()
        self.bomb_pos = (me.x, me.y)
        # Step towards the cell that maximises distance from the blast cross
        # while staying on the board.
        best, best_score = (0, 0), -1e9
        for dx, dy in MOVES:
            nx, ny = me.x + dx, me.y + dy
            if not self.walkable(wrld, nx, ny):
                continue
            # Manhattan-off-the-cross score: we want to leave the bomb's row
            # AND column.
            score = min(abs(nx - me.x), 4) + min(abs(ny - me.y), 4)
            if nx != me.x and ny != me.y:
                score += 3                       # diagonal leaves both arms
            if score > best_score:
                best, best_score = (dx, dy), score
        self.move(*best)

    
    #  STATES: EVADE and HIDE. expectimax over the real simulator        
    
    def act_search(self, wrld, me):
        """
        Pick an action in two stages:

          1. HARD SAFETY FILTER. The monsters have already committed their
             moves for this tick, so we know exactly where they will be. 

          2. EXPECTIMAX over what is left, to avoid the slower death of being
             herded into a corner.
        """
        predicted = self.predicted_monster_cells(wrld)

     
        survivors = [(dx, dy) for dx, dy in MOVES
                     if ((dx, dy) == (0, 0) or self.walkable(wrld, me.x + dx, me.y + dy))
                     and not self.dies_immediately(wrld, (dx, dy))]
        if not survivors:
            survivors = [(0, 0)]

        # Stage 2: rank every survivor with the search, then add a *bonus*
        best_val, best = -float('inf'), []
        for a in survivors:
            nx, ny = me.x + a[0], me.y + a[1]
            px, py = (nx, ny) if self.walkable(wrld, nx, ny) else (me.x, me.y)
            clearance_bonus = (CLEARANCE_BONUS
                               if all(self.cheb(px, py, mx, my) >= 2
                                      for (mx, my) in predicted)
                               else 0.0)
            v = self.simulate(wrld, a, False, self.eff_depth, first=True) + clearance_bonus
            if v > best_val + 1e-9:
                best_val, best = v, [a]
            elif abs(v - best_val) <= 1e-9:
                best.append(a)

        # Tie-break towards the exit, then at random so we never lock into a two-cell oscillation.
        best.sort(key=lambda a: self.dist_to_exit.get(
            (me.x + a[0], me.y + a[1]), 999))
        top_d = self.dist_to_exit.get((me.x + best[0][0], me.y + best[0][1]), 999)
        top = [a for a in best
               if self.dist_to_exit.get((me.x + a[0], me.y + a[1]), 999) == top_d]
        self.move(*random.choice(top))

    def predicted_monster_cells(self, wrld):
        """
        Where each monster will be after this tick.

        This is exact, not a guess: RealWorld.next_decisions() calls the
        monsters' do() before the characters', so every monster has already
        chosen its (dx,dy), and SensedWorld copies those fields over.
        """
        cells = []
        for m in self.monster_list(wrld):
            nx = max(0, min(wrld.width() - 1, m.x + m.dx))
            ny = max(0, min(wrld.height() - 1, m.y + m.dy))
            if wrld.wall_at(nx, ny):        # a blocked monster stays put
                nx, ny = m.x, m.y
            cells.append((nx, ny))
            cells.append((m.x, m.y))        # it may also fail to move at all
        return cells

    def dies_immediately(self, wrld, action):
        """One exact tick of simulation: does this action get us killed now?"""
        sim = SensedWorld.from_world(wrld)
        mine = sim.me(self)
        if mine is None:
            return True
        mine.move(*action)
        nxt, events = sim.next()
        for e in events:
            if (e.tpe == Event.CHARACTER_FOUND_EXIT
                    and e.character.name == self.name):
                return False                # winning is not dying
            if self.victim_name(e) == self.name:
                return True
        return nxt.me(self) is None

    
    #  Expectimax                                                        #
    
    def simulate(self, wrld, action, bomb, depth, first=False):
        """Value of taking `action` in `wrld`, backed up `depth` plies."""
        sim = SensedWorld.from_world(wrld)
        me = sim.me(self)
        if me is None:
            return DEATH_SCORE
        me.move(*action)
        if bomb:
            me.place_bomb()

        # FIRST PLY IS EXACT.
        
        if first:
            nxt, events = sim.next()
            return self.backup(nxt, events, depth)

       
        active = sorted(
            (m for m in self.monster_list(sim)
             if self.cheb(m.x, m.y, me.x, me.y) <= THREAT_RADIUS + depth),
            key=lambda m: self.cheb(m.x, m.y, me.x, me.y)
        )[:MAX_ACTIVE]
        return self.monster_node(sim, me, active, 0, depth)

    def monster_node(self, sim, me, active, idx, depth):
        """Assign a move to each active monster, then advance the world."""
        if idx == len(active):
            nxt, events = sim.next()
            return self.backup(nxt, events, depth)

        m = active[idx]
        # A monster that is close enough to lunge is treated as an adversary
        # (MIN node); a distant one is a CHANCE node (it moves at random).
        adversarial = self.cheb(m.x, m.y, me.x, me.y) <= ADVERSARIAL_R
        options = self.monster_moves(sim, m, me)[:BRANCH_LIMIT]

        values = []
        for mv in options:
            m.move(*mv)                       
            values.append(self.monster_node(sim, me, active, idx + 1, depth))

        if not values:
            return DEATH_SCORE
        average = sum(values) / len(values)
        if not adversarial:
            return average                    
        
        return RISK_LAMBDA * min(values) + (1.0 - RISK_LAMBDA) * average

    def backup(self, nxt, events, depth):
        """Terminal test, then evaluate (leaf) or recurse (MAX node)."""
        for e in events:
            if (e.tpe == Event.CHARACTER_FOUND_EXIT
                    and e.character.name == self.name):
                return WIN_SCORE + depth * 100.0      # winning sooner is better
            if self.victim_name(e) == self.name:
                return DEATH_SCORE - depth * 100.0    # dying later is less bad

        me = nxt.me(self)
        if me is None:
            return DEATH_SCORE - depth * 100.0
        if depth <= 1:
            return self.evaluate(nxt, me)

        best = -float('inf')
        for dx, dy in MOVES:
            if (dx, dy) != (0, 0) and not self.walkable(nxt, me.x + dx, me.y + dy):
                continue
            best = max(best, self.simulate(nxt, (dx, dy), False, depth - 1))
        return best if best > -float('inf') else DEATH_SCORE


    #  Evaluation function                                               
   
    def evaluate(self, wrld, me):
        """ value of a non-terminal state"""
        # self.dist_to_exit is recomputed once per turn in do(). walls only change when an explosion clears, so it is safe to reuse it here and
        # it keeps the leaf evaluation cheap.
        d = self.dist_to_exit.get((me.x, me.y))
        if d is None:
            d = self.cheb(me.x, me.y, *wrld.exitcell) + 20   # walled off: penalise
        score = -W_EXIT * d

        # Monster proximity: Adjacent is treated as near-death directly rather than through the linear term,
        # since one tile away a monster simply walks onto us next turn.
        for m in self.monster_list(wrld):
            md = self.cheb(me.x, me.y, m.x, m.y)
            if md <= 1:
                score += DEATH_SCORE / 100.0
            else:
                score += W_MONSTER * min(md, MONSTER_CAP)

        
        score += W_ESCAPE * self.escape_room(wrld, me)

        # Bombs: being inside a blast cross that is about to go off is bad, and the badness grows as the fuse burns down.
        timing = self.blast_timing(wrld)
        t = timing.get((me.x, me.y))
        if t is not None:
            score += BLAST_PENALTY / max(t, 1)

        # Live explosions are lethal to walk into.
        if wrld.explosion_at(me.x, me.y):
            score += BLAST_PENALTY

        return score

    
    #  Navigation: BFS distance field + A*                               
    
    def exit_distance_map(self, wrld):
        """
        BFS from the exit over all non-wall cells.
        Returns {(x,y): number_of_steps}. Recomputed every call because bombs
        destroy walls and therefore change the map.
        """
        goal = wrld.exitcell
        if goal is None:
            return {}
        dist = {goal: 0}
        queue = [goal]
        while queue:
            nxt = []
            for (x, y) in queue:
                for dx, dy in MOVES:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not self.in_bounds(wrld, nx, ny):
                        continue
                    if wrld.wall_at(nx, ny):
                        continue
                    if (nx, ny) in dist:
                        continue
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    nxt.append((nx, ny))
            queue = nxt
        return dist

    def escape_room(self, wrld, me):
        """
        BFS outward from `me`, up to ESCAPE_DEPTH steps, over cells that are
        walkable and are not within distance 1 of any monster. Returns the
        fraction of that BFS frontier that was actually reachable (0..1).

        This is what plain "distance to nearest monster" cannot see: two
        states can have the same distance-1 monster reading, but one has three
        open directions to keep running and the other has the agent's back to
        a wall. The BFS finds that difference directly instead of us having to
        hand-craft a "corner" feature.
        """
        monsters = self.monster_list(wrld)

        def blocked(x, y):
            if not self.walkable(wrld, x, y):
                return True
            return any(self.cheb(x, y, m.x, m.y) <= 1 for m in monsters)

        if blocked(me.x, me.y):
            return 0.0

        seen = {(me.x, me.y)}
        frontier = [(me.x, me.y)]
        reached = 0
        for _ in range(ESCAPE_DEPTH):
            nxt = []
            for (x, y) in frontier:
                for dx, dy in MOVES:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in seen or not self.in_bounds(wrld, nx, ny):
                        continue
                    seen.add((nx, ny))
                    if blocked(nx, ny):
                        continue
                    reached += 1
                    nxt.append((nx, ny))
            frontier = nxt
            if not frontier:
                break
        # Normalise against the best case (8 dirs x ESCAPE_DEPTH steps).
        return reached / (8.0 * ESCAPE_DEPTH)

    def astar(self, wrld, start, goal):
        """
        A* on the 8-connected grid with a heuristic.
        Step cost is 1 plus a soft penalty for cells near a monster or inside
        a bomb's blast cross, so the path naturally keeps its distance.
        """
        if goal is None:
            return []
        monsters = self.monster_list(wrld)
        timing = self.blast_timing(wrld)

        open_heap = [(self.cheb(*start, *goal), 0, start, None)]
        came, gscore = {}, {start: 0}
        while open_heap:
            f, g, cur, parent = heapq.heappop(open_heap)
            if cur in came:
                continue
            came[cur] = parent
            if cur == goal:
                # Reconstruct
                path, node = [], cur
                while node is not None:
                    path.append(node)
                    node = came[node]
                return path[::-1]
            for dx, dy in MOVES:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cur[0] + dx, cur[1] + dy
                if not self.walkable(wrld, nx, ny):
                    continue
                cost = 1.0
                for m in monsters:                    # soft monster clearance
                    md = self.cheb(nx, ny, m.x, m.y)
                    if md <= 2:
                        cost += 40.0
                    elif md <= 4:
                        cost += 6.0
                if (nx, ny) in timing:                # avoid ticking crosses
                    cost += 15.0
                ng = g + cost
                if ng < gscore.get((nx, ny), float('inf')):
                    gscore[(nx, ny)] = ng
                    heapq.heappush(open_heap,
                                   (ng + self.cheb(nx, ny, *goal), ng,
                                    (nx, ny), cur))
        return []

    
    @staticmethod
    def cheb(x1, y1, x2, y2):
        """8-connected distance."""
        return max(abs(x1 - x2), abs(y1 - y2))

    @staticmethod
    def in_bounds(wrld, x, y):
        return 0 <= x < wrld.width() and 0 <= y < wrld.height()

    def walkable(self, wrld, x, y):
        """A cell we may legally and safely step into this turn."""
        return (self.in_bounds(wrld, x, y)
                and not wrld.wall_at(x, y)
                and not wrld.explosion_at(x, y))

    @staticmethod
    def monster_list(wrld):
        out = []
        for _, mlist in wrld.monsters.items():
            out.extend(mlist)
        return out

    def our_bomb(self, wrld):
        """Return our live bomb, or None. We may only ever have one."""
        for _, b in wrld.bombs.items():
            if b.owner is not None and b.owner.name == self.name:
                return b
        return None

    def monster_moves(self, wrld, m, me=None):
        """
        Legal moves for a monster, ordered so that 'towards us' comes first
        (cheap move ordering: the dangerous branch is examined earliest, which
        makes the MIN-node cut-off fire sooner).
        """
        opts = []
        for dx, dy in MOVES:
            nx, ny = m.x + dx, m.y + dy
            if not self.in_bounds(wrld, nx, ny):
                continue
            if wrld.wall_at(nx, ny):
                continue
            opts.append((dx, dy))
        if me is not None:
            opts.sort(key=lambda d: self.cheb(m.x + d[0], m.y + d[1], me.x, me.y))
        return opts

    def blast_timing(self, wrld):
        """
        {cell: ticks_until_it_catches_fire} for every bomb on the board.
        The cross is 4 cells in each cardinal direction and is stopped by the
        first wall it meets (that wall cell is still hit).
        """
        timing = {}

        def mark(cell, t):
            if cell not in timing or t < timing[cell]:
                timing[cell] = t

        for _, b in wrld.bombs.items():
            fuse = b.timer + 1                 # expired() is timer < 0
            mark((b.x, b.y), fuse)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for r in range(1, wrld.expl_range + 1):
                    nx, ny = b.x + dx * r, b.y + dy * r
                    if not self.in_bounds(wrld, nx, ny):
                        break
                    mark((nx, ny), fuse)
                    if wrld.wall_at(nx, ny):
                        break               # the blast stops at the first wall
        return timing

    @staticmethod
    def victim_name(e):
        """Who died in this event?"""
        if e.tpe == Event.BOMB_HIT_CHARACTER:
            return e.other.name if e.other else None
        if e.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
            return e.character.name
        return None

    
    #  Debug        
   
    def paint_debug(self, wrld):
        self.tiles = {}
        for cell in self.path[1:]:
            self.set_cell_color(cell[0], cell[1], Fore.CYAN)
        for cell in self.blast:
            self.set_cell_color(cell[0], cell[1], Fore.RED)
