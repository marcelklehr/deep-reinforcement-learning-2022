import environment

env = environment.Environment()
env.visualize()
terminal = False
rewards = 0

while not terminal:
    key = input('Press WASD to move 🌴🦘🌴')
    print(key)
    state, reward, terminal = env.step(key)
    env.visualize()
    rewards += reward
    print('REWARD: %i' % rewards)
