# Model specification

The research question of this project is: How do individual traders' responses to the green energy transition narrative collectively lead to delayed price increases in green energy infrastructure stocks?The project constructs a model in which individual traders update their beliefs about green energy infrastructure stocks based on green energy-related information and buy or sell these stocks when their beliefs reach certain thresholds. The collective actions of traders should generate price increases in green energy infrastructure stocks, where the timing and pattern of the price dynamics are determined by the threshold parameters. The model should contain the following components:

## 1. Belief updating

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

## 3. Price dynamics

Green energy infrastructure price updates following

$$
P(t + 1) = P(t) + B_{total}(t)
$$

$$
B_{total}(t) = \sum B_i(t)
$$

$P(t)$: price of green energy infrastructure stocks at (t)

$B_{total}(t)$: aggregate trader behavior at time (t)

## Simulation scenarios (hypotheses)

The project simulates three different sets of model parameters and produces corresponding outputs to compare three scenarios:

- Aggressive buyers: traders have low buying thresholds $\theta_{u_i} \sim U(0.2, 0.4)$ and high selling thresholds $\theta_{l_i} \sim U(-0.8, -0.6)$. This will lead to an early increase in the price of green energy infrastructure stocks, which is prone to fluctuation.

- Conservative buyers: traders have high buying thresholds $\theta_{u_i} \sim U(0.6, 0.8)$ and low selling thresholds $\theta_{l_i} \sim U(-0.4, -0.2)$. This will lead to a delayed but steady increase in the price of green energy infrastructure stocks.

- Mixed buyers: traders have mixed thresholds (half has $\theta_{u_i} \sim U(0.2, 0.4)$ and $\theta_{l_i} \sim U(-0.8, -0.6)$, the other half has $\theta_{u_i} \sim U(0.6, 0.8)$ and $\theta_{l_i} \sim U(-0.4, -0.2)$, by random assignment). The price behavior will fall between the above two scenarios.

# Results

The price trajectories for the conservative buyers and mixed buyers scenarios both epxerience an initial decrease between t=0 and t=10, followed by a short plateau and then a linear increase. The conservative buyers scenario shows a relatively larger decrease and a delayed growth than the mixed buyers scenario. In constrast, the price trajectory for the aggressive buyers scenario expereiences no decrease. It remains constants in early time points and then enters the same linear growth trend.

These patterns appear because at first negative information is stronger than positive information, but as positive information grows linearly while negative information remains bounded, more and more agents reach their thresholds and become buyers without returing to sellers. 

This behavior is also reflected in the belief trajectory of Agent 0, which is plotted as an example. The belief decreases in early stages and then increases cross the buying threshold, and eventually approaches the upper bound of 1. The increase is not linear, which is due to the hyperbolic tangent transformation that keeps belief between -1 and 1. The belief trajectory shows small fluctuations, indicating that the uncertainty term is correctly included in the model.

Overall, the result partly aligns with the hypothese that conservative buyers result in a more delayed increase of green energy infrastructure stock prices, aggressive buyers produce an early growth, and the price behavior of the mixed buyers scenario falls exactly in between the other two senarios. However, the price of aggressive buyers does not experience any fluctuations. This is again due to the fact that positive informatin grows linearly without bound, and the effect of negative information gradually becomes negligible. Once agents cross their thresholds and become buyers, they continue to buy and lead to a constant increase in price.

# Reflection

In general, I think Cursor accurately generated a plan, correctly implemented the code, and followed the rules and skills I specified. The following are several aspects that made me trust the generated output. Before forming the plan, Cursor presented two multiple choices to decide details about the organization of ABM code and the stimulation parameters (i.e. number of time steps and initial price). These were helpful for coding the project and indicated its ability to "understand" the details of the modeling task. The generated plan included clear implementation details, some of which I asked for specific modification. Also, after modifying the ABM implementation code, it reran the test suite again and acknowledged the rule "Do not modify any test files again once generated".

More importantly, testing helped me verify the implementation of the project. Cursor generated a comprehensive test suite `tests/test_abm.py`, including checks for belief, action, and thresholds in bounds, seed reproducibility, and updating rules. All checks are passed before Cursor presented new code changes each time. The test suite also includes the simulation of two reduced models. The first reduced model removes negative information and uncertainty. The resulting belief trajectory shows a monotonic increase untill it is bounded by 1, which matches the behavior of a belief updating model with only linearly increasing positive information. The second reduced model has constant positive information I(t)=0.6 and constant negative information N(T)=0.5 and no uncertainty. The resulting belief trajectory demonstrates an increase that quickly plateau, approaching to the theoretical equilibrium of this system  
$$x^{eq} = 0.9x^{eq} + 0.3(0.6-0.5)$$
$$x^{eq} = 0.3$$
$$b = tanh(0.3) = 0.291$$
which verifies the implementation. 

Overall, I feel that this project was well handled by Cursor (I'm using a paid version), likely due to the low complexity and the fact that an ABM of inidividual traders and stock prices is not something new. For more complex projects, I think writing my own code with a repreduction manual generated by AI should be a better approach.



