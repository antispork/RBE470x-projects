"""
Headless trainer / evaluator for the approximate Q-learning Bomberman agent.

It plays the same scenarios as project1/variantN.py and project2/variantN.py
(same maps, same monsters, same start positions) but without pygame, without
the board printing and without waiting, so it runs as fast as Python allows.

The agent is TestCharacter(..., mode="qlearning"); all Q-learning code is in
qlearning.py.  The learned weights live in  team05/q_weights.json.  To play with
them, create the character in a variantN.py with  mode="qlearning"  (or set
DEFAULT_MODE = "qlearning" in testcharacter.py); it then loads the file and plays
greedily (no learning while playing).

USAGE  (run from anywhere, e.g. from the team05 folder)

    # train on variants 2-5 of both projects, 3000 episodes
    python train_qlearning.py --episodes 3000

    # keep training from the weights saved last time (this is the default)
    python train_qlearning.py --episodes 3000

    # start over from the hand-set prior weights / from zeros
    python train_qlearning.py --episodes 3000 --init priors
    python train_qlearning.py --episodes 3000 --init zeros

    # choose scenarios as  project:variant  (comma separated)
    python train_qlearning.py --scenarios 1:4,1:5,2:5

    # oversample hard scenarios by repeating them (training draws from this list;
    # checkpoints and evaluations still use each distinct scenario once), and decay
    # the learning rate linearly from --alpha to --alpha-end over the run
    python train_qlearning.py --scenarios 2:3,2:4,2:4,2:5,2:5,2:5 --alpha 0.02 --alpha-end 0.002

    # only measure how good the saved weights are (no learning, no saving)
    python train_qlearning.py --eval-only --eval-episodes 20

    # train WITHOUT the hard safety filter (then also play without it!)
    python train_qlearning.py --no-shield

    # train with smart bombing (walk to the blocking wall before bombing; reachable
    # monsters preempt BOMB). Then play with TestCharacter(..., smart_bomb=True) too!
    python train_qlearning.py --smart-bomb

WEIGHT SIGNS
    While training, every weight is kept on its sensible side of zero (danger
    features can only be penalties, good features can only be rewards; see
    WEIGHT_SIGNS in qlearning.py).  Turn it off with --no-sign-constraints.
    A weights file saved without the constraints is projected onto them when
    training resumes from it (a message says which weights moved).

CHECKPOINTS
    Linear Q-learning can drift, so the LAST weights are not always the best.
    Every --checkpoint-every episodes (and at the start and the end) the current
    weights play --checkpoint-games greedy games per scenario on FIXED seeds
    (no learning); the best one so far is kept.
      <weights>.json        the BEST checkpoint  (what the game loads)
      <weights>_last.json   the latest weights   (safety copy)
    Because the starting weights are checkpointed too, the saved result can never
    score worse than where you started on that test set.  The test set is also what
    picks the winner, so a held-out run (--eval-episodes, other seeds) is the fair
    final number.  --checkpoint-every 0 turns this off (last weights go to <weights>.json).

Ctrl-C stops training cleanly and saves the weights.
"""

import argparse
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'Bomberman'))
sys.path.insert(0, _HERE)

from real_world import RealWorld
from events import Event
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

from qlearningchararacter import TestCharacter
from qlearning import ApproxQLearner, WEIGHTS_FILE, FEATURES, WEIGHT_SIGNS, ALPHA, GAMMA

# Variants 2-5 are the ones with monsters; variant 1 is plain A*, nothing to learn.
DEFAULT_SCENARIOS = "1:2,1:3,1:4,1:5,2:2,2:3,2:4,2:5"


#  World construction (mirrors Game.fromfile + variantN.py, minus pygame)

def load_world(map_path):
    with open(map_path, 'r') as fd:
        lines = [ln.rstrip('\r\n') for ln in fd]
    params = {}
    for ln in lines[:4]:
        key, val = ln.split()
        params[key] = int(val)
    width = len(lines[4]) - 2                       # "+" + "-"*width + "+"
    rows = []
    for ln in lines[5:]:
        if not ln.startswith('|'):
            break
        if len(ln) != width + 2:
            raise RuntimeError("Row %d of %s is not %d characters long"
                               % (len(rows) + 1, map_path, width))
        rows.append(ln)
    world = RealWorld.from_params(width, len(rows), params['max_time'],
                                  params['bomb_time'], params['expl_duration'],
                                  params['expl_range'])
    for y, row in enumerate(rows):
        for x in range(width):
            ch = row[x + 1]
            if ch == 'E':
                if world.exitcell:
                    raise RuntimeError("There can be only one exit cell")
                world.add_exit(x, y)
            elif ch == 'W':
                world.add_wall(x, y)
    return world


def make_world(project, variant):
    """Exactly the monsters used by project<project>/variant<variant>.py."""
    world = load_world(os.path.join(_HERE, "project%d" % project, "map.txt"))
    if variant == 2:
        world.add_monster(StupidMonster("stupid", "S", 3, 9))
    elif variant == 3:
        world.add_monster(SelfPreservingMonster("selfpreserving", "S", 3, 9, 1))
    elif variant == 4:
        world.add_monster(SelfPreservingMonster("aggressive", "A", 3, 13, 2))
    elif variant == 5:
        world.add_monster(StupidMonster("stupid", "S", 3, 5))
        # project1/variant5.py uses detection range 1, project2/variant5.py uses 2
        world.add_monster(SelfPreservingMonster("aggressive", "A", 3, 13,
                                                1 if project == 1 else 2))
    elif variant != 1:
        raise ValueError("variant must be 1..5")
    return world


def parse_scenarios(text):
    out = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        p, v = item.split(':')
        p, v = int(p), int(v)
        if p not in (1, 2) or v not in (1, 2, 3, 4, 5):
            raise ValueError("bad scenario '%s' (use project:variant, e.g. 1:4)" % item)
        out.append((p, v))
    if not out:
        raise ValueError("no scenarios given")
    return out


#  One episode

def game_over(world):
    """Same stop conditions as Game.done(), but robust to emptied character lists."""
    if world.time <= 0:
        return True
    return not any(world.characters.values())


def run_episode(project, variant, learner, training, shield, max_steps, seed, smart_bomb=False):
    """Play one game. Returns (outcome, steps, score, q_updates); outcome in exit/dead/timeout."""
    random.seed(seed)
    world = make_world(project, variant)
    me = TestCharacter("me", "C", 0, 0, mode="qlearning",
                       learner=learner, training=training, shield=shield,
                       smart_bomb=smart_bomb)
    world.add_character(me)

    outcome, steps = "timeout", 0
    # Same order as Game.go(): next() first, then everybody decides.
    while not game_over(world) and steps < max_steps:
        world, events = world.next()
        steps += 1
        for e in events:
            if e.tpe == Event.CHARACTER_FOUND_EXIT and e.character.name == "me":
                outcome = "exit"
            elif TestCharacter.victim_name(e) == "me":
                outcome = "dead"
        world.next_decisions()

    if outcome == "timeout":
        me.end_episode()
    return outcome, steps, world.scores.get("me", 0), me.qpolicy.q_updates


#  Training / evaluation drivers

def eval_stats(scenarios, episodes, learner, shield, max_steps, seed, smart_bomb=False):
    """Greedy play, no learning, nothing printed. Returns [(project, variant, [results])]."""
    learner.epsilon = 0.0
    rows = []
    for (p, v) in scenarios:
        rows.append((p, v, [run_episode(p, v, learner, False, shield, max_steps, seed + i, smart_bomb)
                            for i in range(episodes)]))
    return rows


def summarize(rows):
    results = [r for (_, _, rs) in rows for r in rs]
    n = max(len(results), 1)
    return {"n":       len(results),
            "exit":    sum(1 for r in results if r[0] == "exit"),
            "dead":    sum(1 for r in results if r[0] == "dead"),
            "timeout": sum(1 for r in results if r[0] == "timeout"),
            "score":   sum(r[2] for r in results) / float(n)}


def evaluate(scenarios, episodes, learner, shield, max_steps, seed, smart_bomb=False):
    """Greedy play, no learning. Prints one line per scenario."""
    rows = eval_stats(scenarios, episodes, learner, shield, max_steps, seed, smart_bomb)
    for (p, v, results) in rows:
        n_exit = sum(1 for r in results if r[0] == "exit")
        n_dead = sum(1 for r in results if r[0] == "dead")
        avg_steps = sum(r[1] for r in results) / float(len(results))
        avg_score = sum(r[2] for r in results) / float(len(results))
        print("  project%d/variant%d: exit %2d/%d  dead %2d  timeout %2d  "
              "avg steps %6.1f  avg score %8.1f"
              % (p, v, n_exit, episodes, n_dead, episodes - n_exit - n_dead,
                 avg_steps, avg_score))
    st = summarize(rows)
    print("  OVERALL exit rate: %.1f%%" % (100.0 * st["exit"] / max(st["n"], 1)))


def unique(scenarios):
    """The distinct scenarios, in order of first appearance."""
    return list(dict.fromkeys(scenarios))


def last_path(path):
    """q_weights.json -> q_weights_last.json"""
    root, ext = os.path.splitext(path)
    return root + "_last" + (ext or ".json")


def save_best(args, learner, best):
    """Write the best checkpoint's weights to args.weights."""
    snap = ApproxQLearner(weights=best["w"], alpha=learner.alpha, gamma=learner.gamma,
                          constrain=learner.constrain)
    snap.episodes = best["episodes"]
    snap.save(args.weights, meta={
        "kind": "best checkpoint",
        "checkpoint_episode": best["episodes"],
        "exit_rate": best["exit_rate"],
        "mean_score": best["score"],
        "checkpoint_games_per_scenario": args.checkpoint_games,
        "scenarios": args.scenarios,
        "shield": not args.no_shield,
        "smart_bomb": args.smart_bomb,
        "max_steps": args.max_steps})


def train(args, scenarios, learner):
    rng = random.Random(args.seed)
    test_scenarios = unique(scenarios)             # repeats only change how often we TRAIN on one
    window = []                                    # rolling record of recent outcomes
    t0 = time.time()
    shield = not args.no_shield
    use_ckpt = args.checkpoint_every > 0
    best = None                                    # best checkpoint so far (see save_best)
    done_eps = 0
    last_ckpt_ep = -1
    last_saved = 0

    def checkpoint(ep):
        """Evaluate the current weights on the fixed test games; keep them if best so far."""
        nonlocal best, last_ckpt_ep
        st = summarize(eval_stats(test_scenarios, args.checkpoint_games, learner, shield,
                                  args.max_steps, args.seed + 777, args.smart_bomb))
        rate = st["exit"] / float(max(st["n"], 1))
        better = best is None or (rate, st["score"]) > (best["exit_rate"], best["score"])
        prev = "" if best is None else "  (best so far %.1f%%)" % (100 * best["exit_rate"])
        print("checkpoint ep %6d: exit %5.1f%%  dead %5.1f%%  timeout %5.1f%%  "
              "mean score %8.1f  [%d games]  %s%s"
              % (ep, 100 * rate, 100.0 * st["dead"] / max(st["n"], 1),
                 100.0 * st["timeout"] / max(st["n"], 1), st["score"], st["n"],
                 "NEW BEST" if better else "not better", "" if better else prev))
        sys.stdout.flush()
        last_ckpt_ep = ep
        if better:
            best = {"w": dict(learner.w), "episodes": learner.episodes,
                    "exit_rate": rate, "score": st["score"]}
            if ep > 0:                             # the starting weights are not re-written
                save_best(args, learner, best)

    try:
        if use_ckpt:
            checkpoint(0)                          # baseline: where we start from
        for ep in range(1, args.episodes + 1):
            frac = (ep - 1) / float(max(args.episodes - 1, 1))
            learner.epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)
            if args.alpha_end is not None:
                learner.alpha = args.alpha + frac * (args.alpha_end - args.alpha)
            p, v = rng.choice(scenarios)
            outcome, steps, score, upd = run_episode(
                p, v, learner, True, shield, args.max_steps,
                args.seed * 1000003 + ep, args.smart_bomb)
            learner.episodes += 1
            done_eps = ep
            window.append(outcome)
            window = window[-args.log_every:]

            if ep % args.log_every == 0:
                n = float(len(window))
                print("ep %6d  eps %.3f  alpha %.4f  exit %5.1f%%  dead %5.1f%%  timeout %5.1f%%  "
                      "(last scenario %d:%d, %d steps)  %.0fs"
                      % (ep, learner.epsilon, learner.alpha,
                         100 * window.count("exit") / n,
                         100 * window.count("dead") / n,
                         100 * window.count("timeout") / n,
                         p, v, steps, time.time() - t0))
                sys.stdout.flush()
            if ep % args.save_every == 0:
                learner.save(last_path(args.weights) if use_ckpt else args.weights)
                last_saved = ep
            if use_ckpt and ep % args.checkpoint_every == 0:
                checkpoint(ep)
        if use_ckpt and last_ckpt_ep != done_eps:
            checkpoint(done_eps)                   # always judge the final weights too
    except KeyboardInterrupt:
        print("\nInterrupted after %d episodes." % done_eps)

    if not use_ckpt:
        learner.save(args.weights)
        print("Saved weights to %s  (%d episodes total%s)"
              % (args.weights, learner.episodes,
                 "" if last_saved == done_eps else ", final save"))
        return

    learner.save(last_path(args.weights))
    print("Latest weights saved to %s  (%d episodes total)"
          % (last_path(args.weights), learner.episodes))
    if best is None:                               # interrupted before the first checkpoint finished
        learner.save(args.weights)
        print("No checkpoint finished; saved the current weights to %s" % args.weights)
        return
    save_best(args, learner, best)
    print("BEST checkpoint: episode %d, exit %.1f%%, mean score %.1f -> saved to %s"
          % (best["episodes"], 100 * best["exit_rate"], best["score"], args.weights))
    if best["episodes"] != learner.episodes:
        print("(the latest weights were not the best; the weights below and the final "
              "evaluation use the best ones)")
    learner.w = dict(best["w"])                    # print / evaluate the best, not the last


def print_weights(learner):
    print("weights:   (* = held at 0 by its sign constraint)")
    for k in FEATURES:
        held = learner.constrain and k in WEIGHT_SIGNS and learner.w[k] == 0.0
        print("  %-12s %9.4f%s" % (k, learner.w[k], " *" if held else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=2000, help="training episodes (default 2000)")
    ap.add_argument("--scenarios", default=DEFAULT_SCENARIOS,
                    help="comma separated project:variant list (default: %s)" % DEFAULT_SCENARIOS)
    ap.add_argument("--weights", default=WEIGHTS_FILE, help="weights file to read/write")
    ap.add_argument("--init", choices=["file", "priors", "zeros"], default="file",
                    help="start from the saved file (default; priors if missing), "
                         "the hand-set priors, or all zeros")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="learning rate (default %g)" % ALPHA)
    ap.add_argument("--alpha-end", type=float, default=None,
                    help="decay the learning rate linearly from --alpha to this value over the "
                         "run (default: constant)")
    ap.add_argument("--gamma", type=float, default=GAMMA, help="discount (default %g)" % GAMMA)
    ap.add_argument("--eps-start", type=float, default=0.30, help="initial exploration rate")
    ap.add_argument("--eps-end", type=float, default=0.02, help="final exploration rate")
    ap.add_argument("--max-steps", type=int, default=1000,
                    help="stop an episode after this many ticks (default 1000)")
    ap.add_argument("--no-shield", action="store_true",
                    help="disable the hard safety filter (train AND play without it)")
    ap.add_argument("--smart-bomb", action="store_true",
                    help="TestCharacter(smart_bomb=True): walk to the blocking wall before "
                         "bombing, reachable monsters preempt BOMB (train AND play with it)")
    ap.add_argument("--no-sign-constraints", action="store_true",
                    help="let weights take any sign (default: keep them on the side given by "
                         "WEIGHT_SIGNS in qlearning.py)")
    ap.add_argument("--checkpoint-every", type=int, default=250,
                    help="evaluate and keep the best weights every N episodes "
                         "(default 250; 0 = off)")
    ap.add_argument("--checkpoint-games", type=int, default=10,
                    help="greedy games per scenario in each checkpoint (default 10)")
    ap.add_argument("--seed", type=int, default=1, help="random seed")
    ap.add_argument("--log-every", type=int, default=50, help="print stats every N episodes")
    ap.add_argument("--save-every", type=int, default=100, help="save weights every N episodes")
    ap.add_argument("--eval-episodes", type=int, default=0,
                    help="after training, play this many greedy games per scenario")
    ap.add_argument("--eval-only", action="store_true",
                    help="no training: just evaluate the weights (default 10 games per scenario)")
    args = ap.parse_args()

    scenarios = parse_scenarios(args.scenarios)

    constrain = not args.no_sign_constraints
    if args.init == "file":
        learner = ApproxQLearner.load(args.weights, alpha=args.alpha, gamma=args.gamma,
                                      constrain=constrain)
    else:
        learner = ApproxQLearner(alpha=args.alpha, gamma=args.gamma, constrain=constrain)
        if args.init == "zeros":
            for k in FEATURES:
                learner.w[k] = 0.0

    print("scenarios: %s   shield: %s   smart_bomb: %s   sign constraints: %s   alpha %s  gamma %g"
          % (scenarios, "off" if args.no_shield else "on",
             "on" if args.smart_bomb else "off", "on" if constrain else "off",
             ("%g" % args.alpha) if args.alpha_end is None else ("%g -> %g" % (args.alpha, args.alpha_end)),
             args.gamma))

    if args.eval_only:
        n = args.eval_episodes or 10
        print("Evaluating (greedy, no learning, no saving):")
        evaluate(unique(scenarios), n, learner, not args.no_shield, args.max_steps, args.seed + 777,
                 args.smart_bomb)
        return

    if constrain:
        moved = learner.project()
        if moved:
            print("Sign constraints moved these starting weights to 0: %s"
                  % ", ".join("%s (%.3f)" % (k, old) for k, (old, new) in moved.items()))

    train(args, scenarios, learner)
    print_weights(learner)
    if args.eval_episodes:
        # With checkpoints the test games above picked the winner, so judge it on other seeds.
        seed = args.seed + (555555 if args.checkpoint_every > 0 else 777)
        print("Evaluating (greedy, no learning%s):"
              % (", held-out seeds" if args.checkpoint_every > 0 else ""))
        evaluate(unique(scenarios), args.eval_episodes, learner, not args.no_shield,
                 args.max_steps, seed, args.smart_bomb)


if __name__ == "__main__":
    main()