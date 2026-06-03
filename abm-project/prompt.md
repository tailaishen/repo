Help me plan a python project that builds and analyzes an agent-based model. Read the following descriptions and generate a plan. Do not write code yet.

# Scientific goal

## Model specification

The project constructus a model of individual traders updating their believes about AI infrastructure stocks given AI-related information and buying or selling AI infrastructure stocks when their believes reach certain thresholds. Together, traders' actions should lead to price increase in AI infrastructure stocks, where the exact behaviors of price are determined by the threshold parameters. The model should contain the following components:

### 1. Belief updating

Each agent (i = 1,...,n) represents an individual trader who updates belief following

$$
b_i(t + 1) = \gamma b_i(t) + \delta (I(t) - N(t)) + \epsilon_i(t)
$$

$b_i(t) \in [0,1]$ : belief of trader (i) at time (t) that AI infrastructure will increase in value in the future
$I(t)= 0.05 + 0.005t$ : positive AI-related information at time (t)
$N(t) = (1 + \sin(\pi t))/2$: negative AI-related information at time (t)
$\epsilon_i(t) \sim N(0, 0.05)$: individual uncertainty at time (t)
$\gamma = 0.05$: belief retained rate
$\delta = 0.02$: information sensitivity

### 2. Trading actions

Each agent (i) takes action about AI infrastructure stocks following

$$
B_i(t)=
\begin{cases}
1, & b_i(t) > \theta_{u_i} \\
0, & \theta_{l_i}\le b_i(t) \le \theta_{u_i} \\
-1, & b_i(t) \le \theta_{l_i}
\end{cases}
$$

$B_i(t)$: buying (B_i(t) = 1), selling (B_i(t) = -1), or no behavior (B_i(t) = 0) of trader (i) at time (t)
$\theta_{l_i} \in [0, 1]$: individual threshold that triggers selling
$\theta_{u_i} \in [0, 1]$: individual threshold that triggers buying

### 3. Price dynamics

AI infrastructure price updates following

$$
P(t + 1) = P(t) + B_{total}(t)
$$

$$
B_{total}(t) = \sum B_i(t)
$$

$P(t)$: price of AI infrastructure stocks at (t)
$B_{total}(t)$: aggregate trader behavior at time (t)

## Simulation scenarios

The project simulates three different sets of model parameters and produces corresponding outputs to compare three scenarios:

- Traders have low buying thresholds $\theta_{l_i} \sim U(0.2, 0.4)$ and high selling thresholds $\theta_{u_i} \sim U(-0.8, -0.6)$.

- Traders have high buying thresholds $\theta_{l_i} \sim U(0.6, 0.8)$ and low selling thresholds $\theta_{u_i} \sim U(-0.4, -0.2)$.

- Traders have mixed thresholds (half has $\theta_{l_i} \sim U(0.2, 0.4)$ and $\theta_{u_i} \sim U(-0.8, -0.6)$, the other half has $\theta_{l_i} \sim U(0.6, 0.8)$ and $\theta_{u_i} \sim U(-0.4, -0.2)$, by random assignment).


# Engineering goals

## File structure

Under `abm-project`, create the following files

- a `PROMPT.md` containing the prompt that describes the project to the AI for planning purposes

- a `PLAN.md` that generates the ABM implementation

- ABM implementation code

- a `run_simulation.py` script that reproducibly runs the simulation and saves the results to a `results/` folder

- a context-management artifact used in this project `SKILL.md`

- a `Dockerfile` that reproduces the project environment

<!-- - a `README.md` with three main sections:
  - **Model specification** describing agents, state variables, update rules, and metrics
  - **Results** summarizing the scientific conclusions (if any)
  - **Reflection** on how you ensured accuracy of the codebase, and whether you now trust the results -->

## ABM implementation

- a class `Agent` that owns its belief and contains methods for belief updating and trading actions.

- a class `AIStockABM` that owns the agent list, random seed, histories of belief, action, and price. It should include the following key behaviors:
  - allows specification of model parameters
  - create $n$ agents with initial belief value $0$
  - stimulate agent belief updating and trading actions at each time $t$
  - compute aggregate trader behavior $B_{total}(t)$ and price of AI infrastructure stocks $P(t)$ at time $t$

- Plotting functions that visualizes the stimulation results. It should create 
  - a figure including three trajectories of $P(t)$ with respect to time $t$, each for one simluation scenario
  - a figure illustrating the belief trajectory for one of the agents


# Context management

## Rules

- Do not generate files or modify files outside of `abm-project` folder

- Do not modify any test files again once generated. 

- Do not modify files without the code being reviewed by the human user.

## Skills

Create a `SKILL.md` for storing the following skill description

- Write clean and understandable code. Each class or function should contain concise annotation.

- Before presenting newly changed code about ABM implementation
  - run a test stimulation with a constant $I(t)$ and $N(t)$ = $\epsilon_i(t)$ = 0 and save the results
  - run a test suite that checks for unbounded parameter values and validity under edge cases.

