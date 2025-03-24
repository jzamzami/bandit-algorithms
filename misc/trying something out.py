import random
import math
import numpy as np
import matplotlib.pyplot as plt

def categorical_draw(probs):
    choice = random.uniform(0, sum((probs)))
    choiceIndex = 0

    for probability_of_arm in (probs):
        choice -= probability_of_arm
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1

class Exp3():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weights = [1.0 for i in range(n_arms)]
        self.best_arm = random.randint(0, n_arms - 1)
        return

    def finding_probabilities(self):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = [0.0 for i in range(n_arms)]
        for arm in range(n_arms):
            probs[arm] = (1 - self.learning_rate) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + (self.learning_rate) * (1.0 / float(n_arms))
        return probs
        
    def select_arm(self):
        probs = self.finding_probabilities()
        return categorical_draw(probs)

    def update(self, chosen_arm, reward):
        probs = self.finding_probabilities()
        x = reward / probs[chosen_arm]
        growth_factor = math.exp((self.learning_rate / n_arms) * x)
        self.weights[chosen_arm] = self.weights[chosen_arm] * growth_factor
    
    def assign_reward(self, chosen_arm):
        if chosen_arm == self.best_arm:
            if random.random() < 0.7:
                return 1
            else:
                return 0
        else:
            if random.random() < 0.3:
                return 1
            else:
                return 0
    
random.seed(1)
np.random.seed(1)

n_arms = 10
n_rounds = 100000
learning_rate = 0.01

adversarialExp3Environment = Exp3(learning_rate)
regret = []
cumulative_reward = 0

for t in range(n_rounds):
    chosen_arm = adversarialExp3Environment.select_arm()
    reward = adversarialExp3Environment.assign_reward(chosen_arm)
    adversarialExp3Environment.update(chosen_arm, reward)
    
    cumulative_reward += reward
    optimal_reward = (t + 1) * 0.7
    regret.append(optimal_reward - cumulative_reward) 


plt.figure(figsize=(10, 6))
plt.plot(regret, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Exp3 Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.show()