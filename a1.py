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

    def showMean(self, arm_idx):
        return self.__means[arm_idx]
    
    def showArmNum(self):
        return self.__num_arms

    def pull_arm(self, arm_idx):
        return np.random.normal(loc = self.__means[arm_idx], scale = self.__variance)

    def printAllMeans(self):
        print(self.__means)

NUM_BANDITS = 10
NUM_REWARDS = 10
NUM_STEPS = 10000
EPSILON = 0.1
ALPHA = 0.5
VARIANCE = 1
RAND_SEED = 0

rewards = None
times_chosen = None
bandit = None

def init():
    # RAND_SEED = int(input("Set random seed to: "))
    np.random.seed(RAND_SEED)
    global bandit 
    bandit = Bandit(10, 0.1)
    rewards = np.full(10, 0.0)
    times_chosen = np.full(10, 0)


def random_walk():
    for arm_idx in range(bandit.showArmNum()):
        bandit.changeMean(arm_idx, np.random.normal(loc = 0, scale = 0.01))

init()
bandit.printAllMeans()
random_walk()
bandit.printAllMeans()


def sample_average(arm_idx):
    prev_reward = rewards[arm_idx]
    curr_reward = np.random.choice(distributions[arm_idx])
    times_chosen[arm_idx] += 1
    rewards[arm_idx] = prev_reward + (curr_reward - prev_reward)/times_chosen[arm_idx]

# def weighted_average(arm_idx):


def epsilon_greedy(e, method):
    rand_num = np.random.rand()
    if rand_num <= e:
        arm_idx = np.random.randint(10)
    else:
        arm_idx = rewards.argmax()
    
    if method == "sample_average":
        sample_average(arm_idx)
    # else:



def runStationary():
    for i in range(NUM_STEPS):
        epsilon_greedy(EPSILON, "sample_average")

def runNonstationary():
    for i in range(NUM_STEPS):
        random_walk()
        epsilon_greedy(EPSILON, "sample_average")

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


# init()

# runStationary()
# validate("stationary test")

# runNonstationary()
# validate("nonstationary test")

# mean = 5
# scale = 1
# num = np.random.normal(5, scale)
# print(num)

# for i in range(100):
#     # random_walk()
#     arm_idx = np.random.randint(10)
#     curr_reward = np.random.choice(distributions[arm_idx])
#     rewards[arm_idx][1] += 1
#     rewards[arm_idx][0] = rewards[arm_idx][0] + (curr_reward - rewards[arm_idx][0])/rewards[arm_idx][1]
#     # print()
#     # print("arm_idx =", arm_idx)
#     # print("curr_reward =", curr_reward)
#     # print("prev reward =", rewards[arm_idx][0])
#     # print("updated_reward =", rewards[arm_idx][0])
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

plt.show()