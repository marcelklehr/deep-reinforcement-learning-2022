import numpy as np
import environment

env = environment.Environment()
env.visualize()

epsilon = 0.3
episodes = 10000
n_steps = 20
alpha = 0.2  # do not crak up too high -> inf/nan
gamma = 0.95
our_agent = environment.Agent(epsilon, alpha, gamma)
og_qtable = our_agent.q_table.copy()


for t in range(episodes):
    env.reset()
    our_agent.reset()

    # learning q value
    print(t)
    our_agent.n_sarsa(env, n_steps)
    env.visualize()
    our_agent.visualize_q_table()
    # print(our_agent.q_table)
    our_agent.epsilon *= 0.9999
    our_agent.alpha *= 0.9999
    print("epsilon = %f" % our_agent.epsilon)
    print("alpha = %f" % our_agent.alpha)
    if np.isnan(our_agent.q_table[0, 0, 0]):
        break
#print("OG QTABLE:")
# print(og_qtable)
# env.visualize()
print(our_agent.q_table)
