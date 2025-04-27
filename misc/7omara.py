import math
import numpy as np
import random

def exp3(numActions, reward, gamma):
    weights = [1.0] * numActions
    t = 0
    while True:
        probabilityDistribution = distr(weights, gamma)
        #choice = draw(probabilityDistribution)
        choice = np.random.choice(numActions, p=probabilityDistribution)
        theReward = reward(choice, t)
        estimatedReward = 1.0 * theReward / probabilityDistribution[choice]
        weights[choice] *= math.exp(estimatedReward * gamma / numActions) # important that we use estimated reward here!
        yield choice, theReward, estimatedReward, weights
        t = t + 1

def distr(weights, gamma=0.0):
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)

numActions = 10
numRounds = 100000

biases = [1.0 / k for k in range(2,12)]
rewardVector = [[1 if random.random() < bias else 0 for bias in biases] for _ in range(numRounds)]
rewards = lambda choice, t: rewardVector[t][choice]

bestAction = max(range(numActions), key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]))

bestUpperBoundEstimate = 2 * numRounds / 3
gamma = math.sqrt(numActions * math.log(numActions) / ((math.e - 1) * bestUpperBoundEstimate))
gamma = 0.07

cumulativeReward = 0
bestActionCumulativeReward = 0
weakRegret = 0

t = 0
for (choice, reward, est, weights) in exp3(numActions, rewards, gamma):
    cumulativeReward += reward
    bestActionCumulativeReward += rewardVector[t][bestAction]

    weakRegret = (bestActionCumulativeReward - cumulativeReward)
    regretBound = (math.e - 1) * gamma * bestActionCumulativeReward + (numActions * math.log(numActions)) / gamma

    t += 1
    if t >= numRounds:
        break
    
#print("reward vector:", rewardVector)
print("best action:", bestAction)
# print("cum award:", cumulativeReward)
# print("best action cum award:", bestActionCumulativeReward)
# print("regret:", weakRegret)