"""
Approximate Q-learning for the Bomberman agent (replaces the expectimax search
when TestCharacter is created with mode="qlearning").

  Q(s, a) = w . f(s, a)

f(s, a) is a small vector of hand-designed features (distance to the exit,
distance to the nearest monster, how much room there is to run, how dangerous
the blast cross of a ticking bomb is, ...).  The features are measured on the
state that action `a` leads to.  That state comes from one call to
SensedWorld.next(), which is exact because the monsters have already committed
to their move for this tick (RealWorld.next_decisions() runs the monsters' do()
before the characters').  There is no multi-ply search.

The weights `w` are learned with the usual approximate Q-learning update

    delta = r + gamma * max_a' Q(s', a') - Q(s, a)
    w_i  += alpha * delta * f_i(s, a)

Training is done by train_qlearning.py, which runs the game headless and saves
the weights to q_weights.json (next to this file).  When the game is played
normally the character just loads those weights and acts greedily.

A HARD SAFETY FILTER (exact one-tick simulation + an exact "can I still get out
of my own blast cross in time" test) removes actions that are certain death
before the Q-values are consulted.  It can be switched off with `shield=False`
(the trainer has a --no-shield flag).  Use the SAME setting for training and
for playing.

Classes
  ApproxQLearner   the weights, Q(s,a) and the gradient update, save / load
  QPolicy          the per-character glue: features, action choice, bookkeeping
"""

import os
import sys
import json
import random

from sensed_world import SensedWorld
from events import Event

_HERE = os.path.dirname(os.path.abspath(__file__))


#  Constants

MONSTER_CAP     = 4      # monster distance is clipped here (far is far enough)
ESCAPE_DEPTH    = 3      # how many free steps away from danger counts as "safe"
DOOM_SEARCH_MAX = 8      # max BFS depth when checking we can leave our blast cross

ALPHA           = 0.02   # learning rate
GAMMA           = 0.9    # discount factor
DELTA_CLIP      = 20.0   # |TD error| is clipped to this before the update
WEIGHT_CLIP     = 200.0  # |weight| is clipped to this (keeps training finite)
EXIT_DISC       = 0.9    # exit_disc feature = EXIT_DISC ** distance_to_exit

# Rewards (used ONLY while training)
R_EXIT          =  10.0  # reached the exit
R_DEATH         = -10.0  # killed by a monster or by an explosion
R_TIMEOUT       =  -5.0  # trainer stopped the episode with us still alive
R_STEP          =  -0.02 # cost of every Q-controlled step (be quick)
R_PROGRESS      =   0.2  # per BFS step gained towards the exit
R_MON_ADJ       =  -0.5  # ended the step adjacent to a monster
R_MON_D2        =  -0.15 # ended the step two cells from a monster
R_BLAST         =  -0.5  # times 1/fuse while standing in a ticking bomb's cross

# The feature vector. Order is only used for saving / printing.
FEATURES = [
    "bias",         # constant 1
    "progress",     # BFS steps gained towards the exit this move, in [-1, 1]
    "exit_disc",    # EXIT_DISC ** (distance to exit after the move), in (0, 1]
    "mon_adj",      # 1 if a monster is within 1 cell after the move
    "mon_d2",       # 1 if the nearest monster is exactly 2 cells away
    "mon_d3",       # 1 if the nearest monster is exactly 3 cells away
    "mon_dist",     # min(nearest monster distance, MONSTER_CAP) / MONSTER_CAP
    "ahead_adj",    # 1 if a monster, going on in its current direction, ends within 1 cell of us
    "ahead_d2",     # 1 if that predicted monster cell is exactly 2 cells from us
    "escape",       # fraction of cells reachable in ESCAPE_DEPTH steps without touching a monster
    "in_blast",     # 1 if standing inside the cross of a ticking bomb
    "blast_urg",    # 1 / (ticks until that bomb explodes)   (0 if not in the cross)
    "blast_steps",  # steps needed to leave the cross, normalised  (0 if not in the cross)
    "wait",         # 1 if the action is "stand still"
    "dies",         # 1 if the action kills us this tick
    "wins",         # 1 if the action reaches the exit this tick
]

# Starting weights. These are sensible hand-set PRIORS (they roughly mimic the
# old evaluation function: run for the exit, keep ~4 cells from monsters, keep
# room to run, stay out of ticking blast crosses).  They are what the agent
# uses before any training; training then refines them.
#
# They are scaled to the rewards above so the first TD errors are small:
#   * exit_disc = 10 -> Q ~ R_EXIT * 0.9**distance, the discounted value of the exit
#   * bias = -13     -> cancels what mon_dist (+10) and escape (+3) add in open
#                       space with no monster near, so Q ~ value of the exit there
#   * dies / wins    -> the terminal rewards themselves
DEFAULT_WEIGHTS = {
    "bias":        -13.0,
    "progress":      1.0,
    "exit_disc":    10.0,
    "mon_adj":     -12.0,
    "mon_d2":       -3.0,
    "mon_d3":       -1.0,
    "mon_dist":     10.0,
    "ahead_adj":    -6.0,
    "ahead_d2":     -1.5,
    "escape":        3.0,
    "in_blast":     -3.0,
    "blast_urg":    -6.0,
    "blast_steps":  -2.0,
    "wait":          0.0,
    "dies":        -10.0,
    "wins":         10.0,
}
# DEFAULT_WEIGHTS = {
#     "bias":        0.0,
#     "progress":      0.0,
#     "exit_disc":    0.0,
#     "mon_adj":     0.0,
#     "mon_d2":       0.0,
#     "mon_d3":       0.0,
#     "mon_dist":     0.0,
#     "ahead_adj":    0.0,
#     "ahead_d2":     0.0,
#     "escape":        0.0,
#     "in_blast":     0.0,
#     "blast_urg":    0.0,
#     "blast_steps":  0.0,
#     "wait":          0.0,
#     "dies":        0.0,
#     "wins":         0.0,
# }

# Allowed sign of each weight while TRAINING (+1: weight >= 0, -1: weight <= 0,
# missing: free).  Every update that would push a weight to the wrong side of
# zero stops at zero instead (projected gradient).  Danger features may only
# ever cost, "good" features may only ever help; this stops training from
# trading weight between overlapping features (e.g. mon_dist vs mon_d2/mon_d3)
# until a danger feature ends up rewarded.  The hand-set priors obey all of these.
WEIGHT_SIGNS = {
    "progress":    +1,
    "exit_disc":   +1,
    "mon_adj":     -1,
    "mon_d2":      -1,
    "mon_d3":      -1,
    "mon_dist":    +1,
    "ahead_adj":   -1,
    "ahead_d2":    -1,
    "escape":      +1,
    "in_blast":    -1,
    "blast_urg":   -1,
    "blast_steps": -1,
    "dies":        -1,
    "wins":        +1,
}                                  # bias and wait are free

WEIGHTS_FILE = os.path.join(_HERE, "q_weights_v5_mondist.json")

# 8-connected move set.  (0,0) is a legal action: waiting is often the single
# best thing to do while a bomb is ticking or a monster is walking past.
MOVES = [(dx, dy)
         for dx in (-1, 0, 1)
         for dy in (-1, 0, 1)]


class ApproxQLearner:
    """
    The learned part: a weight per feature and the Q-learning update.
    It knows nothing about Bomberman, it only maps {feature: value} -> Q.
    """

    def __init__(self, weights=None, alpha=ALPHA, gamma=GAMMA, epsilon=0.0, constrain=True):
        self.w = dict(DEFAULT_WEIGHTS)
        if weights:
            self.set_weights(weights)
        self.constrain = constrain   # keep weights on their WEIGHT_SIGNS side during update()
        self.alpha    = alpha
        self.gamma    = gamma
        self.epsilon  = epsilon      # exploration rate (only used when training)
        self.episodes = 0            # how many training episodes these weights have seen

    def set_weights(self, weights):
        """Copy over every known feature; unknown keys are ignored."""
        for k in FEATURES:
            if k in weights:
                self.w[k] = float(weights[k])

    def q(self, feats):
        """Q(s,a) = w . f(s,a)"""
        return sum(self.w[k] * v for k, v in feats.items())

    def update(self, feats, target):
        """One approximate-Q gradient step towards `target`. Returns the TD error."""
        delta = target - self.q(feats)
        if delta != delta or delta in (float('inf'), -float('inf')):
            return 0.0                                   # NaN / inf guard
        delta = max(-DELTA_CLIP, min(DELTA_CLIP, delta))
        for k, v in feats.items():
            if v:
                nw = self.w[k] + self.alpha * delta * v
                nw = max(-WEIGHT_CLIP, min(WEIGHT_CLIP, nw))
                if self.constrain:
                    nw = self._on_allowed_side(k, nw)
                self.w[k] = nw
        return delta

    @staticmethod
    def _on_allowed_side(k, value):
        sign = WEIGHT_SIGNS.get(k, 0)
        if sign > 0:
            return max(value, 0.0)
        if sign < 0:
            return min(value, 0.0)
        return value

    def project(self):
        """
        Move every weight that is on the wrong side of zero onto zero.
        Returns {feature: (old, new)} for the weights that changed.
        (Used by the trainer when it resumes from a file saved without constraints.)
        """
        changed = {}
        for k in FEATURES:
            new = self._on_allowed_side(k, self.w[k])
            if new != self.w[k]:
                changed[k] = (self.w[k], new)
                self.w[k] = new
        return changed

    def save(self, path=WEIGHTS_FILE, meta=None):
        data = {"features": FEATURES,
                "weights":  self.w,
                "episodes": self.episodes,
                "alpha":    self.alpha,
                "gamma":    self.gamma}
        if meta:
            data["meta"] = meta                          # free-form info, ignored by load()
        tmp = path + ".tmp"
        with open(tmp, "w") as fd:
            json.dump(data, fd, indent=2)
        os.replace(tmp, path)                            # never leaves a half-written file

    @classmethod
    def load(cls, path=WEIGHTS_FILE, verbose=True, **kwargs):
        """Load saved weights; fall back to DEFAULT_WEIGHTS if there are none."""
        learner = cls(**kwargs)
        if os.path.exists(path):
            try:
                with open(path) as fd:
                    data = json.load(fd)
                learner.set_weights(data.get("weights", data))
                learner.episodes = int(data.get("episodes", 0))
            except Exception as ex:                      # corrupt file: keep the priors
                sys.stderr.write("[Q] could not read %s (%s); using default weights\n"
                                 % (path, ex))
        elif verbose:
            sys.stderr.write("[Q] %s not found; using default prior weights\n" % path)
        return learner


class QPolicy:
    """
    Q-learning decision making for ONE TestCharacter (`agent`).

    The agent supplies the geometry helpers it already has (cheb, in_bounds,
    walkable, monster_list, blast_timing, victim_name) and the per-turn
    `dist_to_exit` BFS field; this class adds the features, the action choice
    and the training bookkeeping.
    """

    def __init__(self, agent, learner=None, training=False, shield=True):
        self.agent     = agent
        self.learner   = learner if learner is not None else ApproxQLearner.load()
        self.training  = training    # True -> epsilon-greedy + weight updates
        self.shield    = shield      # True -> hard safety filter on the actions
        self.pending   = None        # the last chosen (s,a) still waiting for its update
        self.q_updates = 0           # number of weight updates done
        self._cands    = None        # candidates already computed this turn (training)

    #  Turn hooks (called by TestCharacter)

    def begin_turn(self, wrld, me):
        """
        Called once per turn, after the agent has refreshed dist_to_exit.
        The world we see now is the s' of the previous Q-chosen action, so this
        is the moment its update can be completed.
        """
        self._cands = None
        if self.training and self.pending is not None:
            self._cands = self.evaluate_actions(wrld, me)
            self.finish_pending(self.state_value(self._cands))

    def act(self, wrld, me):
        """
        Pick an action in two stages and set it on the agent:

          1. HARD SAFETY FILTER. Actions that kill us now, or that leave us
             inside our own blast cross with no time to get out, are dropped
             (unless that would leave nothing).

          2. Q-VALUES. Among what is left, take the action with the highest
             Q(s,a) = w . f(s,a). While training, with probability epsilon a
             random action from the same pool is taken instead (exploration).
        """
        cands = self._cands if self._cands is not None else self.evaluate_actions(wrld, me)
        self._cands = None
        pool = self.action_pool(cands)

        if self.training and random.random() < self.learner.epsilon:
            chosen = random.choice(pool)
        else:
            chosen = self.best_candidate(pool)

        self.agent.move(*chosen["action"])

        if self.training:
            # Remember (s,a); the update is completed next turn (or in done()).
            self.pending = {"feats":    chosen["feats"],
                            "reward":   chosen["reward"],
                            "terminal": chosen["outcome"] != "alive"}

    def action_pool(self, cands):
        """The candidates the policy is allowed to choose from (the safety filter)."""
        if not self.shield:
            return cands
        alive = [c for c in cands if c["outcome"] != "dead"]
        safe  = [c for c in alive if not c["doomed"]]
        return safe or alive or cands

    def best_candidate(self, pool):
        """Highest Q; ties -> closest to the exit, then at random (never lock into a 2-cell oscillation)."""
        top_q = max(c["q"] for c in pool)
        best  = [c for c in pool if c["q"] >= top_q - 1e-9]
        top_d = min(c["d_new"] for c in best)
        best  = [c for c in best if c["d_new"] == top_d]
        return random.choice(best)

    def state_value(self, cands):
        """V(s) = max_a Q(s,a) over the actions the policy may actually take."""
        return max(c["q"] for c in self.action_pool(cands))

    #  Learning bookkeeping

    def finish_pending(self, next_value):
        """Complete the update for the previous (s,a) now that V(s') is known."""
        p = self.pending
        self.pending = None
        if p is None:
            return
        if p["terminal"]:
            target = p["reward"]
        else:
            target = p["reward"] + self.learner.gamma * next_value
        self.learner.update(p["feats"], target)
        self.q_updates += 1

    def done(self, wrld):
        """
        The game says the character has exited or died.
        There is no next state, so the target is just the terminal reward.
        (May be called more than once for the same death -> guarded by pending.)
        """
        if not self.training or self.pending is None:
            return
        p = self.pending
        self.pending = None
        exited = any(e.tpe == Event.CHARACTER_FOUND_EXIT and e.character.name == self.agent.name
                     for e in wrld.events)
        self.learner.update(p["feats"], R_EXIT if exited else R_DEATH)
        self.q_updates += 1

    def end_episode(self):
        """
        The game stopped (time ran out / the trainer's step limit was hit) while we
        were alive, and the game does not call done() for that.
        """
        if not self.training or self.pending is None:
            return
        p = self.pending
        self.pending = None
        self.learner.update(p["feats"], p["reward"] + R_TIMEOUT)
        self.q_updates += 1

    #  Features: what does each action lead to?

    def evaluate_actions(self, wrld, me):
        """
        For every legal action, simulate ONE tick and describe the outcome.
        Returns a list of dicts: action, outcome ('alive'|'dead'|'win'), feats,
        q, reward, doomed, d_new.
        """
        a = self.agent
        d_old = self.exit_dist(wrld, me.x, me.y)
        cands = []
        for dx, dy in MOVES:
            if (dx, dy) != (0, 0) and not a.walkable(wrld, me.x + dx, me.y + dy):
                continue
            sim = SensedWorld.from_world(wrld)
            mine = sim.me(a)
            if mine is None:
                continue
            mine.move(dx, dy)
            nxt, events = sim.next()
            cands.append(self.describe(wrld, nxt, events, (dx, dy), d_old))
        return cands

    def describe(self, wrld, nxt, events, action, d_old):
        """Turn the simulated successor state into one candidate record."""
        a = self.agent
        outcome = "alive"
        for e in events:
            if e.tpe == Event.CHARACTER_FOUND_EXIT and e.character.name == a.name:
                outcome = "win"
                break
            if a.victim_name(e) == a.name:
                outcome = "dead"
                break
        me2 = nxt.me(a)
        if outcome == "alive" and me2 is None:
            outcome = "dead"

        if outcome == "win":
            feats, reward, doomed, d_new = {"wins": 1.0}, R_EXIT, False, 0
        elif outcome == "dead":
            feats, reward, doomed, d_new = {"dies": 1.0}, R_DEATH, True, 10 ** 6
        else:
            d_new = self.exit_dist(wrld, me2.x, me2.y)
            progress = max(-1, min(1, d_old - d_new))

            md = min([a.cheb(me2.x, me2.y, m.x, m.y) for m in a.monster_list(nxt)],
                     default=99)
            ahead = self.monster_ahead_distance(nxt, me2)
            in_blast, urgency, steps, doomed = self.blast_status(nxt, me2)

            feats = {
                "bias":        1.0,
                "progress":    float(progress),
                "exit_disc":   EXIT_DISC ** d_new,
                "mon_adj":     1.0 if md <= 1 else 0.0,
                "mon_d2":      1.0 if md == 2 else 0.0,
                "mon_d3":      1.0 if md == 3 else 0.0,
                "mon_dist":    min(md, MONSTER_CAP) / float(MONSTER_CAP),
                "ahead_adj":   1.0 if ahead <= 1 else 0.0,
                "ahead_d2":    1.0 if ahead == 2 else 0.0,
                "escape":      self.escape_fraction(nxt, me2),
                "in_blast":    in_blast,
                "blast_urg":   urgency,
                "blast_steps": steps,
                "wait":        1.0 if action == (0, 0) else 0.0,
            }
            reward = R_STEP + R_PROGRESS * progress + R_BLAST * urgency
            if md <= 1:
                reward += R_MON_ADJ
            elif md == 2:
                reward += R_MON_D2

        return {"action":  action,
                "outcome": outcome,
                "feats":   feats,
                "q":       self.learner.q(feats),
                "reward":  reward,
                "doomed":  doomed,
                "d_new":   d_new}

    def monster_ahead_distance(self, wrld, me):
        """
        Distance from `me` to the nearest monster's PREDICTED next cell, assuming each
        monster simply carries on in the direction it is moving now (the monsters go
        straight until they hit an obstacle, and a chasing one keeps closing in).
        A monster whose next cell is blocked is assumed to stay where it is.
        """
        a = self.agent
        best = 99
        for m in a.monster_list(wrld):
            ax, ay = m.x + m.dx, m.y + m.dy
            if not a.in_bounds(wrld, ax, ay) or wrld.wall_at(ax, ay):
                ax, ay = m.x, m.y
            best = min(best, a.cheb(me.x, me.y, ax, ay))
        return best

    def blast_status(self, wrld, me):
        """
        Look at the ticking bombs of `wrld` from the point of view of `me`.
        Returns (in_blast, urgency, steps, doomed):

          in_blast  1.0 if we stand inside a bomb's cross, else 0.0
          urgency   1 / ticks-until-explosion (0 if not in the cross)
          steps     moves needed to leave the cross / (expl_range+1), capped at 1
          doomed    True if we can NOT leave the cross before it fires.

        Timing (see RealWorld.next): in each tick the bombs explode BEFORE the
        characters move.  A bomb with fuse f (= blast_timing value) therefore
        fires in the f-th tick from now and kills whoever is in the cross
        after f-1 more moves.  So we need a path of at most f-1 moves to a cell
        outside every cross.
        """
        timing = self.agent.blast_timing(wrld)
        start = (me.x, me.y)
        fuse = timing.get(start)
        if fuse is None:
            return 0.0, 0.0, 0.0, False

        allowed = max(fuse - 1, 0)
        limit = min(allowed, DOOM_SEARCH_MAX)
        need = self.steps_to_safety(wrld, start, timing, limit)
        norm = float(wrld.expl_range + 1)
        if need is not None:
            return 1.0, 1.0 / fuse, min(need, norm) / norm, False
        # Not found inside the search window. If the window was the whole fuse
        # we are provably doomed; otherwise the fuse is long and we just don't know.
        return 1.0, 1.0 / fuse, 1.0, allowed <= DOOM_SEARCH_MAX

    def steps_to_safety(self, wrld, start, timing, limit):
        """BFS: fewest moves from `start` to a cell that is in no blast cross (<= limit), else None."""
        seen = {start}
        frontier = [start]
        for depth in range(1, limit + 1):
            nxt = []
            for (x, y) in frontier:
                for dx, dy in MOVES:
                    if dx == 0 and dy == 0:
                        continue
                    c = (x + dx, y + dy)
                    if c in seen or not self.agent.walkable(wrld, c[0], c[1]):
                        continue
                    if c not in timing:
                        return depth
                    seen.add(c)
                    nxt.append(c)
            frontier = nxt
            if not frontier:
                break
        return None

    def escape_fraction(self, wrld, me):
        """
        BFS outward from `me`, up to ESCAPE_DEPTH steps, over cells that are
        walkable and are not within distance 1 of any monster. Returns the
        fraction (0..1) of the cells an open field would offer that were
        actually reachable (same idea as TestCharacter.escape_room, normalised).
        """
        a = self.agent
        monsters = a.monster_list(wrld)

        def blocked(x, y):
            if not a.walkable(wrld, x, y):
                return True
            return any(a.cheb(x, y, m.x, m.y) <= 1 for m in monsters)

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
                    if (nx, ny) in seen or not a.in_bounds(wrld, nx, ny):
                        continue
                    seen.add((nx, ny))
                    if blocked(nx, ny):
                        continue
                    reached += 1
                    nxt.append((nx, ny))
            frontier = nxt
            if not frontier:
                break
        # An open field has 8 + 16 + 24 = 4*D*(D+1) cells within D steps.
        return reached / float(4 * ESCAPE_DEPTH * (ESCAPE_DEPTH + 1))

    def exit_dist(self, wrld, x, y):
        """
        Steps from (x,y) to the exit: the BFS distance if there is a path, else
        (walled off) the straight-line 8-distance, so that there is still
        something that pulls the agent towards the wall it has to blow up.
        """
        if wrld.exitcell is None:
            return 0
        d = self.agent.dist_to_exit.get((x, y))
        if d is None:
            d = self.agent.cheb(x, y, *wrld.exitcell)
        return d