import numpy as np
from toy_grid_dag import MOGridEnv

def run_quick_checks():
    env = MOGridEnv(horizon=40, ndim=2)

    # terminal state (both dims at horizon-1)
    s_term = np.array([env.horizon - 1, env.horizon - 1])
    x_term = env.s2x(s_term)

    # 1) mo_reward output shape
    f = env.mo_reward(x_term)
    assert isinstance(f, np.ndarray) and f.shape == (2,), "mo_reward shape"

    # 2) scalarization returns a scalar and uses z_star update
    w = np.array([0.4, 0.6])
    scalar_val = env.scalarize(x_term, w)
    assert np.isscalar(scalar_val) or np.asarray(scalar_val).shape == (), "scalarize output"

    # 3) sample a trajectory from the start and print terminal reward and state
    obs, _, s = env.reset()  # reset() -> (obs, reward_or_obj, state)
    done = False
    # pick only move actions (0..ndim-1) so we don't stop immediately
    while not done:
        a = np.random.randint(0, env.ndim)  # exclude stop action
        obs, r, done, s = env.step_dag(a=a, s=s, w=w)

    # at terminal state print both objective vector, scalarized reward, and the terminal state
    terminal_obj = env.mo_reward(env.s2x(s))
    print("terminal objective vector:", terminal_obj)
    print("terminal scalarized reward:", env.scalarize(terminal_obj, w))
    print("terminal state:", s)

    # 4) true_density_mo returns consistent lengths
    td, end_states, traj_rewards = env.true_density_mo(w)
    assert len(end_states) == len(traj_rewards), "true_density_mo len mismatch"

    # 5) pareto_modes runs and returns a list of terminal states
    pareto = env.pareto_modes()
    print(pareto)
    assert isinstance(pareto, list), "pareto_modes result type"

    print("MOGridEnv quick checks passed")

if __name__ == "__main__":
    run_quick_checks()