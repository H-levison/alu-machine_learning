policy gradients

Monte-Carlo policy gradient (REINFORCE) on the CartPole-v1 environment.

## Files

- `policy_gradient.py`
  - `policy(matrix, weight)` — computes the policy, a softmax over the
    state projected onto the weight matrix.
  - `policy_gradient(state, weight)` — samples an action from the
    policy and returns it with the gradient of the log probability of
    that action with respect to the weights.
- `train.py`
  - `train(env, nb_episodes, alpha=0.000045, gamma=0.98,
    show_result=False)` — plays full episodes, then nudges the weights
    along the gradient of every visited state scaled by the discounted
    return that followed it. Returns the score of each episode, and
    renders the environment every 1000 episodes when `show_result` is
    True.

## Usage

```python
import gym
import matplotlib.pyplot as plt
import numpy as np
from train import train

env = gym.make('CartPole-v1')
scores = train(env, 10000)
plt.plot(np.arange(len(scores)), scores)
plt.show()
env.close()
```

The score climbs towards the 500-step cap of CartPole-v1 as the policy
learns to keep the pole upright.
