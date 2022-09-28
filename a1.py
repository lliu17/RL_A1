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
        # for i in range(self.__num_arms):
        #     self.__means[i] = np.random.normal(loc = 0, scale = 1)
        # self.__means[0] = 0.2
        # self.__means[1] = -0.8
        # self.__means[2] = 1.7
        # self.__means[3] = 0.3
        # self.__means[4] = 1.6
        # self.__means[5] = -1.5
        # self.__means[6] = -0.1
        # self.__means[7] = -1.1
        # self.__means[8] = 0.9
        # self.__means[9] = -0.5
        self.__variance = variance

    def changeMean(self, arm_idx, delta):
        self.__means[arm_idx] += delta

    def getMean(self, arm_idx):
        return self.__means[arm_idx]
    
    def getArmNum(self):
        return self.__num_arms

    def pull_arm(self, arm_idx):
        x = np.random.normal(loc = self.__means[arm_idx], scale = self.__variance)
        return x
    
    def random_walk(self, rand_walk_var, debug=False):
        deltas = np.random.normal(size=self.__num_arms, loc = 0, scale = rand_walk_var)
        self.__means = np.add(self.__means, deltas)

    def isOptimal(self, idx):
        max_arm_idx = self.__means.argmax()
        return idx == max_arm_idx

    def reset_means(self):
        self.__means = np.full(self.__num_arms, 0.0)
    
    def set_random_means(self):
        for i in range(self.__num_arms):
            self.__means[i] = np.random.normal(loc = 0, scale = 1)

    def printAllMeans(self):
        print(self.__means)

count_rand = 0
bandit = None
estimated_rewards = None
times_chosen = None
EPSILON = None
ALPHA = None
avg_obtained_rewards_sample = None
avg_obtained_rewards_weight = None
avg_optimal_choice_sample = None
avg_optimal_choice_weight = None

# after finishing running the test, sort arms by mean, and show how many
# times each arm has been chosen
def validate(test_name, num_steps):
    print("\n------------------------------------")
    print("Testing", end=" ")
    print(test_name)
    print("------------------------------------")

    new_rewards = []
    for i in range(bandit.getArmNum()):
        temp = np.append(estimated_rewards[i], bandit.getMean(i))
        new_rewards.append(np.append(temp, times_chosen[i]))
    new_rewards = sorted(new_rewards, key=lambda x : x[1], reverse=True)

    print("[estimated_rewards, mean, times_chosen]")
    for item in new_rewards:
        print(item)

def init(random_seed, num_arms, variance):
    np.random.seed(random_seed)

    global bandit, estimated_rewards, times_chosen
    bandit = Bandit(num_arms, variance)
    estimated_rewards = np.full(num_arms, 0.0)
    times_chosen = np.full(num_arms, 0)

def epsilon_greedy(step_idx, exp_idx, e, method, debug=False):
    rand_num = np.random.rand()
    if rand_num < e:
        arm_idx = np.random.randint(bandit.getArmNum())
    else:   
        arm_idx = estimated_rewards.argmax()

    # print("pulling on =", arm_idx)   
    prev_reward = estimated_rewards[arm_idx]
    curr_reward = bandit.pull_arm(arm_idx)
    times_chosen[arm_idx] += 1
    isOptimal = 1.0 if bandit.isOptimal(arm_idx) else 0.0

    if method == "sample_average":
        estimated_rewards[arm_idx] = \
            prev_reward + (curr_reward - prev_reward)/times_chosen[arm_idx]
        avg_obtained_rewards_sample[step_idx] = \
            (curr_reward + exp_idx * avg_obtained_rewards_sample[step_idx])/(exp_idx + 1)
        avg_optimal_choice_sample[step_idx] = \
            (isOptimal + exp_idx * avg_optimal_choice_sample[step_idx])/(exp_idx + 1)

    elif method == "weighted_average":
        estimated_rewards[arm_idx] = \
            prev_reward + ALPHA * (curr_reward - prev_reward)
        avg_obtained_rewards_weight[step_idx] = \
            (curr_reward + exp_idx * avg_obtained_rewards_weight[step_idx])/(exp_idx + 1)
        avg_optimal_choice_weight[step_idx] = \
            (isOptimal + exp_idx * avg_optimal_choice_weight[step_idx])/(exp_idx + 1)

def runNonstationary(num_steps, exp_idx, random_walk_variance, reward_method, debug=False):
    for i in range(num_steps):
        if i % 1000 == 0:
            bandit.reset_means()
        bandit.random_walk(random_walk_variance, debug)
        epsilon_greedy(i, exp_idx, EPSILON, reward_method, debug)
        if debug:
            print()
            bandit.printAllMeans()

def runExperiment(num_steps, num_experiments, arms, var, rand_walk_var, alpha, epsilon):
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

    num_arms = arms
    variance = var
    random_walk_variance = rand_walk_var

    global ALPHA, EPSILON
    ALPHA = alpha
    EPSILON = epsilon

    global avg_obtained_rewards_sample, avg_obtained_rewards_weight
    global avg_optimal_choice_sample, avg_optimal_choice_weight
    avg_obtained_rewards_sample = np.zeros(num_steps)
    avg_obtained_rewards_weight = np.zeros(num_steps)
    avg_optimal_choice_sample = np.zeros(num_steps)
    avg_optimal_choice_weight = np.zeros(num_steps)

    for i in range(num_experiments):
        # print("================= sample ================")
        method = "sample_average"
        init(i, num_arms, variance)
        runNonstationary(num_steps, i, random_walk_variance, method, debug=False)

        # print("================= weighted ================")
        method = "weighted_average"
        init(i, num_arms, variance)
        runNonstationary(num_steps, i, random_walk_variance, method, debug=False)

        # print("print final means")
        # bandit.printAllMeans()

    print()
    tot_sample = 0
    for rew in avg_obtained_rewards_sample:
        tot_sample += rew
    print("tot_sample =", tot_sample)

    tot_weighted = 0
    for rew in avg_obtained_rewards_weight:
        tot_weighted += rew
    print("tot_weighted =", tot_weighted)

    plt.title(str(bandit.getArmNum()) + '-Armed Bandit (Stationary)')
    x = np.arange(num_steps)
    
    # plt.ylabel('Average Reward at Step')
    # plt.plot(x, avg_obtained_rewards_sample, label = "Sample Average")
    # plt.plot(x, avg_obtained_rewards_weight, label = "Weighted Average")
    
    plt.ylabel('Optimal Choice Proportion')
    plt.ylim([0.0, 1.0])
    plt.plot(x, avg_optimal_choice_sample, label = "Sample Average Optimal")
    plt.plot(x, avg_optimal_choice_weight, label = "Weighted Optimal")
    
    plt.xlabel('Number of Steps')
    plt.legend()
    plt.show()

# print("avg sample", avg_obtained_rewards_sample)
# print("avg weight", avg_obtained_rewards_weight)

runExperiment(num_steps=10000, num_experiments=1000, arms=10, var=1,
              rand_walk_var=0.01, alpha=0.1, epsilon=0.1)