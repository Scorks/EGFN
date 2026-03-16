import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from types import SimpleNamespace
from toy_grid_dag import GridEnv, MOGridEnv
from egfn import EvolutionGFNAgent, MOEvolutionGFNAgent

import warnings
warnings.filterwarnings("ignore")

def make_args():
    # minimal full set used by agents in this repo (tweak if constructor requests more)
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
    )

def snapshot_population(population):
    """Return list[list[np.ndarray]]: per-agent list of parameter arrays."""
    snaps = []
    for agent in population:
        params = [p.detach().cpu().numpy().copy() for p in agent.model.parameters()]
        snaps.append(params)
    return snaps

def compare_snapshots(before, after, change_thr=1e-6):
    """
    Compare two snapshots (same shape). Return per-agent stats dict.
    stats: { 'total_l2': float, 'max_abs': float, 'frac_changed': float }
    """
    stats = []
    for b_agent, a_agent in zip(before, after):
        total_l2 = 0.0
        max_abs = 0.0
        n_changed = 0
        n_total = 0
        for b, a in zip(b_agent, a_agent):
            diff = (a - b).ravel()
            total_l2 += np.linalg.norm(diff)
            max_abs = max(max_abs, np.max(np.abs(diff)))
            n_changed += np.sum(np.abs(diff) > change_thr)
            n_total += diff.size
        stats.append({"total_l2": float(total_l2), "max_abs": float(max_abs), "frac_changed": float(n_changed / max(1, n_total))})
    return stats

def basic_test():
    args = make_args()
    env_single = GridEnv(horizon=args.horizon, ndim=args.ndim)
    env_mo = MOGridEnv(horizon=args.horizon, ndim=args.ndim)

    print("Creating EvolutionGFNAgent (single-objective test)...")
    try:
        evo = EvolutionGFNAgent(args, [env_single])
        print("  EvolutionGFNAgent created successfully")
    except Exception as e:
        print("  EvolutionGFNAgent creation failed:", type(e).__name__, str(e))
        evo = None

    print("Creating MOEvolutionGFNAgent (multi-objective)...")
    try:
        moevo = MOEvolutionGFNAgent(args, [env_mo])
        print("  MOEvolutionGFNAgent created successfully")
        # print initial MO weight vectors if present
        if hasattr(moevo, "weight_vectors"):
            print("  initial weight vectors:", moevo.weight_vectors)
    except Exception as e:
        print("  MOEvolutionGFNAgent creation failed:", type(e).__name__, str(e))
        moevo = None

    # simple helper to safely convert evolve outputs to numpy arrays
    def to_numpy(x, size):
        import torch
        if x is None:
            return np.zeros(size, dtype=float)
        if isinstance(x, (list, tuple, np.ndarray)):
            return np.array(x, dtype=float)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(float)
        return np.zeros(size, dtype=float)

    # run one evolve call for each agent (catch and print errors)
    if evo is not None:
        try:
            before_evo = snapshot_population(evo.population)
            fe = evo.evolve()
            after_evo = snapshot_population(evo.population)
            stats_evo = compare_snapshots(before_evo, after_evo)
            print("EvolutionGFNAgent mutation/crossover stats (per-agent):")
            for i, s in enumerate(stats_evo):
                print(f"  agent {i}: total_l2={s['total_l2']:.6g}, max_abs={s['max_abs']:.6g}, frac_changed={s['frac_changed']:.3f}")

            fe_np = to_numpy(fe, args.population_size)
            print("Collected SO: ", fe_np)
            print("EvolutionGFNAgent evolve -> mean:{:.4f} best:{:.4f} worst:{:.4f}".format(fe_np.mean(), fe_np.max(), fe_np.min()))
        except Exception as e:
            print("EvolutionGFNAgent evolve failed:", type(e).__name__, str(e))

    if moevo is not None:
        try:
            before_mo = snapshot_population(moevo.population)
            fm = moevo.evolve()
            after_mo = snapshot_population(moevo.population)
            stats_mo = compare_snapshots(before_mo, after_mo)
            print("MOEvolutionGFNAgent mutation/crossover stats (per-agent):")
            for i, s in enumerate(stats_mo):
                print(f"  agent {i}: total_l2={s['total_l2']:.6g}, max_abs={s['max_abs']:.6g}, frac_changed={s['frac_changed']:.3f}")
            fm_np = to_numpy(fm, args.population_size)
            print("Collected MO: ", fm_np)
            print("MOEvolutionGFNAgent evolve -> mean:{:.4f} best:{:.4f} worst:{:.4f}".format(fm_np.mean(), fm_np.max(), fm_np.min()))
        except Exception as e:
            print("MOEvolutionGFNAgent evolve failed:", type(e).__name__, str(e))

    # sample one random trajectory (no stop action) and print terminal state and rewards
    try:
        w = np.array([0.5, 0.5])
        obs, _, s = env_mo.reset()
        done = False
        while not done:
            a = np.random.randint(0, env_mo.ndim)  # exclude stop action to force movement
            obs, r, done, s = env_mo.step_dag(a=a, s=s, w=w)
        terminal_obj = env_mo.mo_reward(env_mo.s2x(s))
        print("Sampled terminal state:", s)
        print("Terminal objective vector:", terminal_obj)
        x = env_mo.s2x(s)
        scalar = env_mo.scalarize(x, w)
        print("TESTING X: ", x, "SCALAR: ", scalar)
        try:
            scalar = env_mo.scalarize(terminal_obj, w)
            print("Terminal scalarized reward (w=0.5/0.5):", scalar)
        except Exception:
            print("Scalarization failed or not supported for this env.")
        # show discovered pareto modes for reference
        try:
            pm = env_mo.pareto_modes()
            print("Pareto modes (sample):", pm[:8])
        except Exception:
            pass
    except Exception as e:
        print("Trajectory sampling failed:", type(e).__name__, str(e))

    print("Basic test finished. Next steps:")
    print(" - To iterate, call evolve() repeatedly and inspect population/weight vectors.")
    print(" - If evolve() errors, paste the traceback and we'll adjust the test accordingly.")

if __name__ == "__main__":
    basic_test()