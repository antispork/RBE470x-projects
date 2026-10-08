# This is necessary to find the main code
import sys
import os
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from qlearningchararacter import TestCharacter
from qlearning import ApproxQLearner, WEIGHTS_FILE, FEATURES, WEIGHT_SIGNS, ALPHA, GAMMA, _HERE

# Create the game
random.seed(random.randint(1,1000000)) # TODO Change this if you want different random choices
g = Game.fromfile('map.txt')
g.add_monster(StupidMonster("stupid", # name
                            "S",      # avatar
                            3, 5,     # position
))
g.add_monster(SelfPreservingMonster("aggressive", # name
                                    "A",          # avatar
                                    3, 13,        # position
                                    2             # detection range
))

# TODO Add your character
learner = ApproxQLearner.load(os.path.join(_HERE, "q_weights_v5.json"))
g.add_character(TestCharacter("me", # name
                              "C",  # avatar
                              0, 0,  # position
                              mode="qlearning",
                              shield=True,
                              smart_bomb=True,
                              learner=learner,
                              adaptive_threat=True
))

# Run!
g.go(100)
