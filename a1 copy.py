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
        self.__means = np.arange(10.0)
        # self.__means[0] = 1
        # self.__means[1] = -1
        # self.__means[2] = 2
        # self.__means[3] = 0.5
        # self.__means[4] = 1.7
        # self.__means[5] = -1.5
        # self.__means[6] = -0.1
        # self.__means[7] = -1.1
        # self.__means[8] = -1.3
        # self.__means[9] = -0.5
        # print("self.__means =", self.__means)
        self.__variance = variance

    def changeMean(self, arm_idx, delta):
        self.__means[arm_idx] += delta

    def getMean(self, arm_idx):
        return self.__means[arm_idx]
    
    def getArmNum(self):
        return self.__num_arms

    def pull_arm(self, arm_idx):
        # print("arm mean = ", self.__means[arm_idx])
        # print("variance = ", self.__variance)
        x = np.random.normal(loc = self.__means[arm_idx], scale = self.__variance)
        # print("got =", x)
        return x
    # for testing - might delete
    def set_random_means(self):
        for i in range(self.__num_arms):
            self.__means[i] = np.random.normal(loc = 0, scale = 1)

    def printAllMeans(self):
        print(self.__means)

count_rand = 0
bandit = None
estimated_rewards = None
times_chosen = None
epsilon = None
alpha = None
avg_obtained_rewards_sample = None
avg_obtained_rewards_weight = None

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

    # tot_rewards_sample = 0
    # for rew in avg_obtained_rewards_sample:
    #     tot_rewards_sample += rew
    # print("sample, tot rew =", tot_rewards_sample)
    # print(avg_obtained_rewards_sample)

    # tot_rewards_weighted = 0
    # for rew in avg_obtained_rewards_weight:
    #     tot_rewards_weighted += rew
    # print("weighted, tot rew =", tot_rewards_weighted)
    # print(avg_obtained_rewards_weight)
    # for i in range(bandit.getArmNum()):
    #     tot_rewards += rewards[i] * times_chosen[i]
    # print()
    # print("total rewards =", tot_rewards)
    # print("average rewards =", tot_rewards/num_steps)
    # print()

def init(random_seed, num_arms, variance):
    np.random.seed(random_seed)

    global bandit, estimated_rewards, times_chosen
    bandit = Bandit(num_arms, variance)
    estimated_rewards = np.full(num_arms, 0.0)
    times_chosen = np.full(num_arms, 0)

# adjust distribution mean for all the arms in a Bandit
def random_walk(variance, debug=False):
    for arm_idx in range(bandit.getArmNum()):
        delta = np.random.normal(loc = 0, scale = variance)
        bandit.changeMean(arm_idx, delta)
        if debug:
            print("delta = ", end="")
            print("{0:.5f}".format(delta))

def epsilon_greedy(step_idx, exp_idx, e, method, debug=False):
    global count_rand
    # rand_num = np.random.randint(100)
    rand_num = np.random.rand()
    if rand_num < e:
        count_rand += 1
        # print("step_idx = ", step_idx)
        arm_idx = np.random.randint(10)
    else:   
        arm_idx = estimated_rewards.argmax()

    # print("pulling on =", arm_idx)   
    prev_reward = estimated_rewards[arm_idx]
    curr_reward = bandit.pull_arm(arm_idx)
    times_chosen[arm_idx] += 1

    if method == "sample_average":
        estimated_rewards[arm_idx] = \
            prev_reward + (curr_reward - prev_reward)/times_chosen[arm_idx]
        avg_obtained_rewards_sample[step_idx] = \
            (curr_reward + exp_idx * avg_obtained_rewards_sample[step_idx])/(exp_idx + 1)
    elif method == "weighted_average":
        estimated_rewards[arm_idx] = \
            prev_reward + alpha * (curr_reward - prev_reward)
        avg_obtained_rewards_weight[step_idx] = \
            (curr_reward + exp_idx * avg_obtained_rewards_weight[step_idx])/(exp_idx + 1)

def runNonstationary(num_steps, exp_idx, random_walk_variance, reward_method, debug=False):
    for i in range(num_steps):
        random_walk(random_walk_variance, debug)
        epsilon_greedy(i, exp_idx, epsilon, reward_method, debug)
        if debug:
            print()
            bandit.printAllMeans()

def runExperiment(num_steps, num_experiments):
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

    num_arms = 10
    variance = 1
    random_walk_variance = 0.01
    # random_walk_variance = 0

    global alpha, epsilon
    alpha = 0.1
    epsilon = 0.1

    global avg_obtained_rewards_sample, avg_obtained_rewards_weight
    avg_obtained_rewards_sample = np.zeros(num_steps)
    avg_obtained_rewards_weight = np.zeros(num_steps)

    for i in range(num_experiments):
        # rand_seed = np.random.randint(0, num_experiments * 1000)
        # print("rand_seed=", rand_seed)

        # print("================= sample ================")
        method = "sample_average"
        init(i, num_arms, variance)
        runNonstationary(num_steps, i, random_walk_variance, method, debug=False)

        # print("================= weighted ================")
        method = "weighted_average"
        init(i, num_arms, variance)
        runNonstationary(num_steps, i, random_walk_variance, method, debug=False)

    tot_sample = 0
    for rew in avg_obtained_rewards_sample:
        tot_sample += rew
    print("tot_sample =", tot_sample)

    tot_weighted = 0
    for rew in avg_obtained_rewards_weight:
        tot_weighted += rew
    print("tot_weighted =", tot_weighted)

    # print(avg_obtained_rewards_sample)
    x = np.arange(num_steps)
    # print("avg_obtained_rewards_sample:\n", avg_obtained_rewards_sample)
    plt.plot(x, avg_obtained_rewards_sample, label = "Sample Average")
    plt.plot(x, avg_obtained_rewards_weight, label = "Weighted Average")
    plt.legend()
    # plt.ylim([-0.2, 1.6])
    plt.show()

runExperiment(1000, 2000)
print("count_rand =", count_rand)
# print("avg sample", avg_obtained_rewards_sample)
# print("avg weight", avg_obtained_rewards_weight)





# fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.text(num_steps/3, 0, 'Initial mean = 0, Initial Variance = 0.1', style='italic', 
    #          bbox={'facecolor': 'green', 'ALPHA': 0.5, 'pad': 10})

    # plt.text(num_steps/3, -0.1, 'Random Walk Variance = 0.01, Learning Rate = 0.1, Epsilon = 0.1', style='italic', 
    #         bbox={'facecolor': 'blue', 'ALPHA': 0.5, 'pad': 10})
    # plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9)


# def getMax():
#     curr_max = [estimated_rewards[0]]
#     for i in range(1, len(estimated_rewards))：
#         if 

# outperform = 0
# for i in range(num_experiments):
#     if avg_obtained_rewards_sample[i] < avg_obtained_rewards_weight[i]:
#         outperform += 1
# print()
# print("tot experiment =", num_experiments, ", weighted outperform =", outperform)


# def calcAvgRewards():
#     tot_rewards = 0
#     for i in range(bandit.getArmNum()):
#         tot_rewards += rewards[i] * times_chosen[i]
#     return tot_rewards/len(times_chosen)

# def reset_rewards(num_arms):
#     global rewards, times_chosen
#     rewards = np.full(num_arms, 0.0)
#     times_chosen = np.full(num_arms, 0)

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
