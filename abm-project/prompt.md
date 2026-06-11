Help me plan a python project that builds and analyzes an agent-based model. Read the following descriptions and generate a plan. Do not write code yet.

# Scientific goal

## Model specification

The research question of this project is: How do individual traders' responses to the green energy transition narrative collectively lead to delayed price increases in green energy infrastructure stocks?The project constructs a model in which individual traders update their beliefs about green energy infrastructure stocks based on green energy-related information and buy or sell these stocks when their beliefs reach certain thresholds. The collective actions of traders should generate price increases in green energy infrastructure stocks, where the timing and pattern of the price dynamics are determined by the threshold parameters. The model should contain the following components:

### 1. Belief updating

Each agent (i = 1,...,n) represents an individual trader who updates belief following

$$
x_i(t + 1) = \gamma x_i(t) + \delta (I(t) - N(t)) + \epsilon_i(t)
$$

$$
b_i(t+1) = \tanh\left(x_i(t+1)\right)
$$

$b_i(t) \in (-1, 1)$ : belief of trader (i) at time (t) that green energy infrastructure will increase in value in the future

$I(t)= 0.2 + 0.05t$ : positive green energy information at time (t)

$N(t) = (1 + 0.3\sin(\frac{\pi t}{10}))/2$: negative green energy information at time (t)

$\epsilon_i(t) \sim N(0, 0.02^2)$: individual uncertainty at time (t)

$\gamma = 0.9$: belief retained rate

$\delta = 0.3$: information sensitivity

### 2. Trading actions

Each agent (i) takes action about green energy stocks following

$$
B_i(t)=
\begin{cases}
1, & b_i(t) > \theta_{u_i} \\
0, & \theta_{l_i}\le b_i(t) \le \theta_{u_i} \\
-1, & b_i(t) \le \theta_{l_i}
\end{cases}
$$

$B_i(t)$: buying (B_i(t) = 1), selling (B_i(t) = -1), or no behavior (B_i(t) = 0) of trader (i) at time (t)

$\theta_{l_i} \in (-1, 0)$: individual threshold that triggers selling

$\theta_{u_i} \in (0, 1)$: individual threshold that triggers buying

### 3. Price dynamics

Green energy infrastructure price updates following

$$
P(t + 1) = P(t) + B_{total}(t)
$$

$$
B_{total}(t) = \sum B_i(t)
$$

$P(t)$: price of green energy infrastructure stocks at (t)

$B_{total}(t)$: aggregate trader behavior at time (t)

## Simulation scenarios

The project simulates three different sets of model parameters and produces corresponding outputs to compare three scenarios:

- Aggressive buyers: traders have low buying thresholds $\theta_{u_i} \sim U(0.2, 0.4)$ and high selling thresholds $\theta_{l_i} \sim U(-0.8, -0.6)$.

- Conservative buyers: traders have high buying thresholds $\theta_{u_i} \sim U(0.6, 0.8)$ and low selling thresholds $\theta_{l_i} \sim U(-0.4, -0.2)$.

- Mixed buyers: traders have mixed thresholds (half has $\theta_{u_i} \sim U(0.2, 0.4)$ and $\theta_{l_i} \sim U(-0.8, -0.6)$, the other half has $\theta_{u_i} \sim U(0.6, 0.8)$ and $\theta_{l_i} \sim U(-0.4, -0.2)$, by random assignment).


# Engineering goals

## File structure

Under `abm-project`, create the following files

- a `PLAN.md` that generates the ABM implementation

- ABM implementation code

- a `run_simulation.py` script that reproducibly runs the simulation and saves the results to a `results/` folder

- ABM test code

## ABM implementation

- a class `Agent` that owns its belief and contains methods for belief updating and trading actions.

- a class `GreenStockABM` that owns the agent list, random seed, histories of belief, action, and price. It should include the following key behaviors:
  - allows specification of model parameters
  - create $n$ agents with initial belief value $0$
  - simulate agent belief updating and trading actions at each time $t$
  - compute aggregate trader behavior $B_{total}(t)$ and price of green energy infrastructure stocks $P(t)$ at time $t$

- Plotting functions that visualizes the simulation results. It should generate
  - a figure including three trajectories of $P(t)$ with respect to time $t$, each for one simluation scenario
  - a figure illustrating the belief trajectory for one of the agents

## ABM testing

a test suite for the ABM imlementation. It should checks if parameters remain in bounds and if the same seed produces the same results. Also, it should run simulations and save corresponding output figrues for reduced ABM models, including 
  - a model without $N(t)$ and $\epsilon_i(t)$
  - a model with constant $I(t)$ and $N(t)$ and without $\epsilon_i(t)$

# Context management

Always read and follow the specifications in `skill.md` and `.cursor/rules/project_rules.md` before generating or modifying code.

# What I want from you first

Before writing any code, produce:

- A brief description of the class hierarchy and how data flows from Agent up through GreenStockABM to the plotting functions.
- A detailed implementation plan stored in `PLAN.md`.