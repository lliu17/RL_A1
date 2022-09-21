import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAND_SEED = int(input("Set random seed to: "))
NUM_BANDITS = 10
NUM_REWARDS = 10
NUM_STEPS = 10000
EPSILON = 0.1
ALPHA = 0.5

np.random.seed(RAND_SEED)
# initial distribution: all 10 arms will return reward 0.5 
# distributions = np.full((NUM_BANDITS, NUM_REWARDS), 0.5)
distributions = np.random.normal(size=(NUM_BANDITS, NUM_REWARDS))
rewards = np.full(10, 0.0)
times_chosen = np.full(10, 0)
# print(distributions)
# print("init rewards =\n", rewards)

def random_walk():
    for bandit_idx in range(len(distributions)):
        curr_bandit = distributions[bandit_idx]
        for reward_idx in range(len(curr_bandit)):
            distributions[bandit_idx][reward_idx] += np.random.normal(loc = 0, scale = 0.01)

def epsilon_greedy(e):
    rand_num = np.random.rand()
    if rand_num <= e:
        bandit_idx = np.random.randint(10)
    else:
        bandit_idx = rewards.argmax()
        # print("max bandit = ", bandit_idx)
    prev_reward = rewards[bandit_idx]
    curr_reward = np.random.choice(distributions[bandit_idx])
    times_chosen[bandit_idx] += 1
    rewards[bandit_idx] = prev_reward + (curr_reward - prev_reward)/times_chosen[bandit_idx]

def runStationary():
    for i in range(NUM_STEPS):
        epsilon_greedy(EPSILON)

def runNonstationary():
    for i in range(NUM_STEPS):
        random_walk()
        epsilon_greedy(EPSILON)

def validate(testName):
    print("#################")
    print(testName)
    print("#################")
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

    means = distributions.mean(1)
    print()
    new_rewards = []
    for i in range(len(rewards)):
        temp = np.append(rewards[i], means[i])
        new_rewards.append(np.append(temp, times_chosen[i]))
    new_rewards = sorted(new_rewards, key=lambda x : x[1], reverse=True)

    print("[rewards, mean, times_chosen] =")
    for item in new_rewards:
        print(item)

    tot_rewards = 0
    for i in range(NUM_BANDITS):
        tot_rewards += rewards[i] * times_chosen[i]
    print()
    print("total rewards =", tot_rewards)
    print("average rewards =", tot_rewards/NUM_STEPS)
    print()



runStationary()
validate("stationary test")

runNonstationary()
validate("nonstationary test")

# for i in range(100):
#     # random_walk()
#     bandit_idx = np.random.randint(10)
#     curr_reward = np.random.choice(distributions[bandit_idx])
#     rewards[bandit_idx][1] += 1
#     rewards[bandit_idx][0] = rewards[bandit_idx][0] + (curr_reward - rewards[bandit_idx][0])/rewards[bandit_idx][1]
#     # print()
#     # print("bandit_idx =", bandit_idx)
#     # print("curr_reward =", curr_reward)
#     # print("prev reward =", rewards[bandit_idx][0])
#     # print("updated_reward =", rewards[bandit_idx][0])
#     # print()  






# # x = np.random.normal(loc = 0, scale = 0.01)
# # print(x)
# # distributions = np.random.normal(scale = 0.01, size=(10, 100))
# mins = distributions.min(1)
# maxes = distributions.max(1)
# means = distributions.mean(1)
# # print()
# # print(distributions)
# # print()
# # print("min = ", mins)
# # print()
# # print("max = ", maxes)
# # print()
# # print("means = ", means)
# # mean_arr = []
# # dist_min = []
# # dist_max = []
# # for oneDist in distributions:
# #     mean_arr.append(np.mean(oneDist))
# #     dist_min.append(max(oneDist))
# #     dist_max.append(max(oneDist))
# #     sns.distplot(oneDist, hist=False)

# # plt.errorbar(np.arange(10), means, [means - mins, maxes - means], 
# #     linestyle='None', marker='.')

# # plt.show()