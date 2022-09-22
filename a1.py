'''
Multi-armed Bandit 
Luxi Liu
Sep 2022

Experiments for demonstrating the difficulties that sample-average methods have
for nonstationary problems.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Bandit:
    def __init__(self, num_arms, variance):
        self.__num_arms = num_arms
        self.__means = np.full(num_arms, 0.0)
        self.__variance = variance

    def changeMean(self, arm_idx, delta):
        self.__means[arm_idx] += delta

    def getMean(self, arm_idx):
        return self.__means[arm_idx]
    
    def getArmNum(self):
        return self.__num_arms

    def pull_arm(self, arm_idx):
        return np.random.normal(loc = self.__means[arm_idx], scale = self.__variance)

    # for testing - might delete
    def set_random_means(self):
        for i in range(self.__num_arms):
            self.__means[i] = np.random.normal(loc = 0, scale = 1)

    def printAllMeans(self):
        print(self.__means)

bandit = None
rewards = None
times_chosen = None
epsilon = None
alpha = None

def init(random_seed, num_arms, variance):
    np.random.seed(random_seed)

    global bandit, rewards, times_chosen, epsilon, alpha
    alpha = 0.1
    epsilon = 0.1
    bandit = Bandit(num_arms, variance)
    rewards = np.full(num_arms, 0.0)
    times_chosen = np.full(num_arms, 0)

# adjust distribution mean for all the arms in a Bandit
def random_walk(variance, debug=False):
    for arm_idx in range(bandit.getArmNum()):
        delta = np.random.normal(loc = 0, scale = variance)
        bandit.changeMean(arm_idx, delta)
        if debug:
            print("delta = ", end="")
            print("{0:.5f}".format(delta))


def sample_average(arm_idx, prev_reward, curr_reward, debug=False):
    if debug:
        print("prev_reward=", prev_reward, ", curr_reward", curr_reward)
        print("times_chosen", times_chosen)
    rewards[arm_idx] = prev_reward + (curr_reward - prev_reward)/times_chosen[arm_idx]

def weighted_average(arm_idx, prev_reward, curr_reward, alpha, debug=False):
    rewards[arm_idx] = prev_reward + alpha * (curr_reward - prev_reward)

def epsilon_greedy(e, method, debug=False):
    rand_num = np.random.rand()
    if rand_num <= e:
        arm_idx = np.random.randint(10)
    else:
        arm_idx = rewards.argmax()
    
    prev_reward = rewards[arm_idx]
    curr_reward = bandit.pull_arm(arm_idx)
    times_chosen[arm_idx] += 1

    sample_average(arm_idx, prev_reward, curr_reward, debug)
    weighted_average(arm_idx, prev_reward, curr_reward, alpha, debug)

def runNonstationary(num_steps, random_walk_variance, reward_method, debug=False):
    for i in range(num_steps):
        random_walk(random_walk_variance, debug)
        epsilon_greedy(epsilon, reward_method, debug)
        if debug:
            print()
            bandit.printAllMeans()

def validate(test_name, num_steps):
    print("\n------------------------------------")
    print("Testing", end=" ")
    print(test_name)
    print("------------------------------------")

    new_rewards = []
    for i in range(bandit.getArmNum()):
        temp = np.append(rewards[i], bandit.getMean(i))
        new_rewards.append(np.append(temp, times_chosen[i]))
    new_rewards = sorted(new_rewards, key=lambda x : x[1], reverse=True)

    print("[rewards, mean, times_chosen]")
    for item in new_rewards:
        print(item)

    tot_rewards = 0
    for i in range(bandit.getArmNum()):
        tot_rewards += rewards[i] * times_chosen[i]
    print()
    print("total rewards =", tot_rewards)
    print("average rewards =", tot_rewards/num_steps)
    print()

def calcAvgRewards():
    tot_rewards = 0
    for i in range(bandit.getArmNum()):
        tot_rewards += rewards[i] * times_chosen[i]
    return tot_rewards/len(times_chosen)

def reset_rewards(num_arms):
    global rewards, times_chosen
    rewards = np.full(num_arms, 0.0)
    times_chosen = np.full(num_arms, 0)

def runExperiment(num_steps, num_experiments):
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

    num_arms = 10
    variance = 0.1
    random_walk_variance = 0.01

    global alpha, epsilon
    alpha = 0.1
    epsilon = 0.1

    avg_rewards_sample = np.array([])
    avg_rewards_weight = np.array([])

    for i in range(num_experiments):
        # rand_seed = np.random.randint(0, num_experiments * 1000)
        # print("rand_seed=", rand_seed)

        global bandit
        bandit = Bandit(num_arms, variance)
        reset_rewards(num_arms)

        method = "sample_average"
        # init(rand_seed, num_arms, variance)
        runNonstationary(num_steps, random_walk_variance, method, debug=False)
        avg_rewards_sample = np.append(avg_rewards_sample, calcAvgRewards())
        validate(method, num_steps)
        reset_rewards(num_arms)

        method = "weighted_average"
        # init(rand_seed, num_arms, variance)
        runNonstationary(num_steps, random_walk_variance, method, debug=False)
        avg_rewards_weight = np.append(avg_rewards_weight, calcAvgRewards())
        validate(method, num_steps)


        
    
    outperform = 0
    for i in range(num_experiments):
        if avg_rewards_sample[i] < avg_rewards_weight[i]:
            outperform += 1
    print()
    print("tot experiment =", num_experiments, ", weighted outperform =", outperform)
    # print()
    # print("avg_rewards_sample:", avg_rewards_sample)
    # print()
    # print("avg_rewards_weight:", avg_rewards_weight)
    # print()

runExperiment(10000, 2)

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

plt.show()