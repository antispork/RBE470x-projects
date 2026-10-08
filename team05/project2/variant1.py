# This is necessary to find the main code
import sys
import os
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from qlearningchararacter import TestCharacter
from qlearning import ApproxQLearner, WEIGHTS_FILE, FEATURES, WEIGHT_SIGNS, ALPHA, GAMMA, _HERE


# Create the game
g = Game.fromfile('map.txt')

# TODO Add your character
learner = ApproxQLearner.load(os.path.join(_HERE, "q_weights_v1_4.json"))
g.add_character(TestCharacter("me", # name
                              "C",  # avatar
                              0, 0,  # position
                              mode="qlearning",
                              shield=True,
                              smart_bomb=True,
                              learner=learner
))

# Run!
g.go(200)
