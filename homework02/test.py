import environment

env = environment.Environment()
env.visualize()
terminal = False
rewards = 0

# while not terminal:
#     key = input('Press WASD to move 🌴🦘🌴')
#     print(key)
#     state, reward, terminal = env.step(key)
#     env.visualize()
#     rewards += reward
#     print('REWARD: %i' % rewards)

# Defining all the required parameters
epsilon = 0.2
episodes = 4000
n_steps = 4
alpha = 0.15
gamma = 0.85
our_agent = environment.Agent(epsilon, alpha, gamma)
og_qtable = our_agent.q_table.copy()

for _ in range(episodes):
    t = 0
    state0 = env.reset()
    #action0 = our_agent.choose_action(state0)
    episodic_reward = 0

    # while t < 100:
    # getting next state and reward and if terminal
    #state1, reward, terminal = env.step(action0)

    # getting next action
    #action1 = our_agent.choose_action(state1)

    # learning q value
    our_agent.n_sarsa(env, n_steps)
    our_agent.visualize_q_table()
    print(our_agent.q_table)

    t += 1
    #episodic_reward += reward
print("OG QTABLE:")
print(og_qtable)
