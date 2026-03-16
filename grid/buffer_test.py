import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
from types import SimpleNamespace
from toy_grid_dag import GridEnv
from egfn import EvolutionGFNAgent

def make_args():
    return SimpleNamespace(
        method="fm_egfn",
        population_size=5,
    num_eval_episodes=2,
    num_elites=1,
    tournament_size=2,
    mutation=True,
    mutation_prob=0.05,
    gamma=0.02,
    crossover_prob=0.5,
    crossover=True,
    feedback=False,
    random_policy=False,
    mutation_frac=0.01,
    super_mutation_prob=0.01,
    reset_prob=0.01,
    mutation_strength=0.02,
    weight_limit=10.0,
    # flow net / replay params
    horizon=16,
    ndim=2,
    n_hid=32,
    n_layers=2,
    dev=torch.device("cpu"),
    bootstrap_tau=0.005,
    replay_strategy="uniform",
    replay_capacity=10000,
        replay_buf_size=10000,
        replay_sample_size=8,
        prb=False,
    )

def main():
    args = make_args()

    env = GridEnv(horizon=args.horizon, ndim=args.ndim)
    print("Creating EvolutionGFNAgent...")
    evo = EvolutionGFNAgent(args, [env])
    print("  created")

    replay = evo.agent_star.replay
    print("Replay object:", type(replay))
    print("Replay attributes:", [a for a in dir(replay) if not a.startswith("_")])

    # show current sample (likely empty)
    try:
        sample_before = replay.sample()
    except TypeError:
        sample_before = []
    print("sample() before ->", sample_before)

    # If buffer empty, run one rollout from a population member to populate it
    if hasattr(replay, "buf") and len(replay.buf) == 0:
        print("Buffer empty -> running one rollout to populate it")
        env_list = [env for _ in range(args.num_eval_episodes)]
        try:
            evo.population[0].sample_many(args.num_eval_episodes, env_list)
            print("After rollout, buf length:", len(replay.buf))
        except Exception as e:
            print("rollout failed:", type(e).__name__, e)
            # fallback: perform simple serial rollouts and append minimal tuples into replay.buf
            try:
                print("Filling buffer with manual rollouts (fallback)...")
                for ep in range(args.num_eval_episodes):
                    obs, _, s = env.reset()
                    done = False
                    last_r = 0.0
                    while not done:
                        a = np.random.randint(0, env.ndim + 1)  # include stop action
                        obs2, r, done, s = env.step_dag(a=a, s=s)
                        last_r = r
                    # append a minimal tuple matching expected (obs, action, reward, next_obs, done)
                    # sample_many / evaluate expect the 3rd element to be the reward/objective
                    if hasattr(replay, 'buf'):
                        replay.buf.append((None, None, last_r, None, True))
                print("After manual rollouts, buf length:", len(replay.buf))
            except Exception as e2:
                print("manual population failed:", type(e2).__name__, e2)

    # print some buffer details
    if hasattr(replay, "buf"):
        print("buf len:", len(replay.buf))
    if len(replay.buf) > 0:
        print("first buf entry:", replay.buf[0])

    # call sample() again and print
    try:
        sample_after = replay.sample()
        print("sample() after ->", sample_after)
    except TypeError:
        print("replay.sample has different signature; inspect buf directly")

if __name__ == "__main__":
    main()
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, numpy as np
from types import SimpleNamespace
from toy_grid_dag import MOGridEnv
from toy_grid_dag import GridEnv
from egfn import EvolutionGFNAgent

# minimal args used by your tests
args = SimpleNamespace(method="fm_egfn", population_size=3, num_eval_episodes=2,
                       horizon=16, ndim=2, n_hid=32, n_layers=2, dev=torch.device("cpu"),
                       bootstrap_tau=0.005, replay_strategy="uniform", replay_capacity=10000,
                       replay_buf_size=10000, replay_sample_size=8,
                       mutation=True, mutation_prob=0.05, gamma=0.02,
                       crossover=True, crossover_prob=0.5, num_elites=1,
                       tournament_size=2, feedback=False, random_policy=False)

env = GridEnv(horizon=args.horizon, ndim=args.ndim)
evo = EvolutionGFNAgent(args, [env])

replay = evo.agent_star.replay
print("Replay object:", type(replay))
print("Replay attributes:", [a for a in dir(replay) if not a.startswith("_")])

sample = replay.sample()
print("sample() ->", sample)

# Try common ways to inspect contents
if hasattr(replay, "buffer"):
    print("buffer length:", len(replay.buffer))
    if len(replay.buffer) > 0:
        print("first entry (raw):", replay.buffer[0])
elif hasattr(replay, "sample"):
    try:
        sample = replay.sample()
        print("sample(1) ->", sample)
    except Exception as e:
        print("replay.sample error:", type(e).__name__, e)
else:
    # Generic scan for likely storage attributes
    for name in ("storage", "data", "buf"):
        if hasattr(replay, name):
            col = getattr(replay, name)
            print(f"{name} len:", len(col))
            if len(col): print("first entry:", col[0])
            break