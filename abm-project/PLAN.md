# Green Stock ABM — Implementation Plan

## Research question

How do individual traders' responses to the green energy transition narrative
collectively lead to delayed price increases in green energy infrastructure
stocks? Each agent updates a belief about future stock value from a positive
information signal `I(t)` and a negative information signal `N(t)`, then trades
when its belief crosses personal thresholds. The aggregate of these trades
drives the price.

## Class hierarchy and data flow

```
run_simulation.py
        │  (master seed, scenario factory)
        ▼
GreenStockABM ──── owns ──▶ list[Agent]
        │                       │
        │  step(t):              │  update_belief(I_t, N_t, rng)  -> b_i
        │   I_t, N_t = info_fn,  │  take_action()                 -> B_i
        │   for each agent:      │
        │     update_belief      │
        │     record b_i         │
        │     take_action -> B_i │
        │   P(t+1) = P(t)+ΣB_i   │
        ▼
   histories: beliefs[T+1, n], actions[T+1, n], price[T+1]
        │
        ▼
   plots.py  -> results/*.png      run_simulation.py -> results/*.npz
```

- `Agent` holds the latent state `x`, derived belief `b = tanh(x)`, and the
  personal thresholds `theta_l`, `theta_u`. It exposes `update_belief` and
  `take_action`. It is intentionally thin so that all randomness is owned by
  the model's seeded `numpy.random.Generator`.
- `GreenStockABM` owns the agent list, the shared parameters
  (`gamma, delta, sigma`), the information signals `I(t)`, `N(t)`, the price
  series, and the seeded RNG. It drives the per-step loop and stores full
  history arrays for downstream analysis.
- `plots.py` reads the histories off one or more `GreenStockABM` instances and
  saves PNGs.
- `run_simulation.py` is a thin reproducible driver: it picks a master seed,
  instantiates the three scenarios via factories, runs them, dumps history
  arrays to `.npz`, and writes the required figures.

## File structure under `abm-project/`

- `PLAN.md` (this file).
- `abm/__init__.py` — re-exports `Agent`, `GreenStockABM`, plotting helpers,
  scenario factories.
- `abm/agent.py` — `Agent` class.
- `abm/model.py` — `GreenStockABM` class, default `I(t)` and `N(t)` callables,
  scenario factories (`make_aggressive`, `make_conservative`, `make_mixed`).
- `abm/plots.py` — `plot_price_trajectories`, `plot_belief_trajectory`.
- `run_simulation.py` — runs the three canonical scenarios, saves `.npz`
  histories and PNG figures into `results/`.
- `tests/test_abm.py` — pytest suite: bounds, determinism, price rule, and the
  two reduced-model figure tests.
- `requirements.txt` — `numpy`, `matplotlib`, `pytest`.
- `results/` — created at runtime; holds `.npz` history dumps and PNG figures
  (with a `tests/` subfolder for reduced-model figures).

## Model specification (recap)

For each agent `i` at time `t`:

- Latent update: `x_i(t+1) = gamma * x_i(t) + delta * (I(t) - N(t)) + eps_i(t)`.
- Belief:        `b_i(t+1) = tanh(x_i(t+1))` so `b_i in (-1, 1)`.
- Action:        `B_i(t) = 1` if `b_i(t) > theta_u_i`; `-1` if `b_i(t) <= theta_l_i`;
                 else `0`.

Default signal and parameter values (from the prompt):

- `I(t) = 0.2 + 0.05 * t`.
- `N(t) = (1 + 0.3 * sin(pi * t / 10)) / 2`.
- `eps_i(t) ~ Normal(0, sigma^2)` with `sigma = 0.02`.
- `gamma = 0.9`, `delta = 0.3`.

Aggregate dynamics:

- `B_total(t) = sum_i B_i(t)`.
- `P(t+1) = P(t) + B_total(t)`.

## Simulation scenarios

Canonical baseline: `n = 1000`, `T = 50`, `P(0) = 10000`. The initial price is
set well above the maximum possible negative cumulative behavior
(`-n * T = -50000` is a worst-case lower bound but realistic draws are far
smaller in magnitude) so the price stays positive across the runs that we
report.

- Aggressive buyers: `theta_u ~ U(0.2, 0.4)`, `theta_l ~ U(-0.8, -0.6)`.
- Conservative buyers: `theta_u ~ U(0.6, 0.8)`, `theta_l ~ U(-0.4, -0.2)`.
- Mixed buyers: half the agents drawn from the aggressive profile, the other
  half from the conservative profile, by random assignment using the model's
  RNG.

## Implementation details

### `abm/agent.py` — `Agent`

- Fields: `x: float` (starts at 0), `b: float` (= `tanh(x)`, starts at 0),
  `theta_l: float`, `theta_u: float`, plus shared `gamma`, `delta`, `sigma`.
- `update_belief(I_t, N_t, rng)` follows the spec order exactly:
  1. `eps_t = rng.normal(0.0, sigma)`.
  2. `self.x = gamma * self.x + delta * (I_t - N_t) + eps_t`.
  3. `self.b = math.tanh(self.x)`.
- `take_action() -> int`:
  - `1` if `self.b > theta_u`,
  - `-1` if `self.b <= theta_l`,
  - `0` otherwise.

### `abm/model.py` — `GreenStockABM`

- Constructor parameters (with baseline defaults):
  - `n: int = 1000`, `T: int = 50`, `P0: float = 10000.0`.
  - `gamma: float = 0.9`, `delta: float = 0.3`, `sigma: float = 0.02`.
  - `threshold_sampler: Callable[[Generator, int], tuple[np.ndarray, np.ndarray]]`
    returning `(theta_l, theta_u)` arrays of length `n`. This supports the
    Mixed scenario cleanly.
  - `info_fn: Callable[[int], float] = lambda t: 0.2 + 0.05 * t`.
  - `noise_fn: Callable[[int], float] = lambda t: (1 + 0.3 * sin(pi*t/10))/2`.
  - `seed: int | None = None`.
- State: `self.rng = np.random.default_rng(seed)`, `self.agents: list[Agent]`,
  history arrays preallocated:
  - `beliefs: np.ndarray` shape `(T+1, n)`, `beliefs[0] = 0`.
  - `actions: np.ndarray` shape `(T+1, n)`, `actions[0] = 0`.
  - `price: np.ndarray` shape `(T+1,)`, `price[0] = P0`.
- `_init_agents()`: draw `theta_l, theta_u` arrays via `threshold_sampler` and
  build the `Agent` list with `x = 0`.
- `step(t)`:
  - `I_t, N_t = info_fn(t), noise_fn(t)`.
  - For each agent: `update_belief(I_t, N_t, self.rng)`, record
    `beliefs[t+1, i] = agent.b`, then `actions[t+1, i] = agent.take_action()`.
  - `B_total = actions[t+1].sum()`; `price[t+1] = price[t] + B_total`.
- `run()`: loop `t = 0 .. T-1` calling `step(t)`; return `self`.
- Module-level scenario factories take `(seed, n, T, P0)` and return a
  configured `GreenStockABM`:
  - `make_aggressive`, `make_conservative`, `make_mixed`.

### `abm/plots.py`

- `plot_price_trajectories(models_by_label, out_path)`: one figure, one line
  per scenario, axes/legend/title; saves PNG.
- `plot_belief_trajectory(model, agent_idx, out_path)`: `b_i(t)` vs `t` for
  one agent, with horizontal dashed lines at that agent's `theta_l` and
  `theta_u`; saves PNG.

### `run_simulation.py`

- Master seed: `SEED = 20260610`, defaults `n=1000`, `T=50`, `P0=10000`.
- Builds the three scenarios using derived seeds (`SEED + 0/1/2`).
- For each scenario: `model.run()`, then
  `np.savez(results/<scenario>_history.npz, beliefs=..., actions=..., price=..., theta_l=..., theta_u=...)`.
- Writes `results/price_trajectories.png` and `results/belief_trajectory.png`
  (agent `0` of the Mixed scenario).
- `argparse` exposes optional `--seed`, `--n`, `--T` overrides; defaults
  reproduce the canonical run.

### `tests/test_abm.py` (pytest)

Tests (kept small with `n ~ 50`, `T ~ 30` unless noted):

- `test_beliefs_in_bounds`: `np.all((beliefs > -1) & (beliefs < 1))`.
- `test_actions_in_set`: `set(np.unique(actions)).issubset({-1, 0, 1})`.
- `test_thresholds_in_bounds`: for each scenario,
  `-1 < theta_l < 0 < theta_u < 1` and the sampled values lie within each
  scenario's stated uniform range.
- `test_seed_reproducibility`: two models with the same seed produce
  identical `beliefs`, `actions`, `price` arrays.
- `test_price_update_rule`: `np.allclose(np.diff(price), actions[1:].sum(axis=1))`.
- Reduced models (save figures to `results/tests/`):
  - `test_reduced_no_noise_no_eps`: build a model with
    `noise_fn = lambda t: 0.0`, `sigma = 0.0`. Save belief + price plot to
    `results/tests/reduced_no_N_no_eps.png`. Assert beliefs are deterministic
    given the seed and `beliefs.mean(axis=1)` is monotonically non-decreasing.
  - `test_reduced_constant_info_no_eps`: `info_fn = lambda t: 0.6`,
    `noise_fn = lambda t: 0.5`, `sigma = 0.0`. With `I - N = 0.1` constant and
    `x(0) = 0`, the deterministic recursion `x(t+1) = 0.9 * x(t) + 0.03`
    converges monotonically to `x* = delta*(I-N)/(1-gamma) = 0.3` and
    `b* = tanh(0.3) ≈ 0.2913`. Save figure to
    `results/tests/reduced_constI_constN_no_eps.png`. Assert (a) all agents
    share identical belief trajectories, (b) beliefs are monotonically
    non-decreasing over `t`, and (c) the trajectory matches the closed-form
    `b(t) = tanh(0.3 * (1 - 0.9**t))` within tolerance.

Per the project rules these test files are written once and not modified again.

## Implementation order

1. Write this `PLAN.md`.
2. Implement `abm/agent.py` and `abm/model.py`.
3. Implement `abm/plots.py`.
4. Write `tests/test_abm.py` and run `pytest`; iterate on model/plots until
   all tests pass (skill requires green tests before presenting code).
5. Implement `run_simulation.py` and execute it once to populate `results/`.
6. Add `requirements.txt`.
