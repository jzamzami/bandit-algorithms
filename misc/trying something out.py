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
    def __init__(self, learning_rate, number_of_arms):
        self.learning_rate = learning_rate
        self.number_of_arms = number_of_arms
        self.estimated_rewards = [1.0 for i in range(number_of_arms)]
        self.best_arm = random.randint(0, number_of_arms - 1)
        return

    def finding_probabilities(self):
        number_of_arms = self.number_of_arms
        total_estimated_rewards = sum(self.estimated_rewards)
        probs = [0.0 for i in range(number_of_arms)]
        for arm in range(number_of_arms):
            probs[arm] = (1 - self.learning_rate) * (self.estimated_rewards[arm] / total_estimated_rewards)
            probs[arm] = probs[arm] + (self.learning_rate) * (1.0 / float(number_of_arms))
        return probs
        
    def select_arm(self):
        probs = self.finding_probabilities()
        action_chosen = categorical_draw(probs)
        return action_chosen

    def update(self, chosen_arm, reward):
        probs = self.finding_probabilities()
        if probs[chosen_arm] > 0:
            x = reward / probs[chosen_arm]
            growth_factor = math.exp((self.learning_rate / number_of_arms) * x)
            self.estimated_rewards[chosen_arm] = self.estimated_rewards[chosen_arm] * growth_factor
        else:
            pass
    
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

number_of_arms = 10
n_rounds = 100000
learning_rate = 0.01

adversarialExp3Environment = Exp3(learning_rate, number_of_arms)
regret = []
cumulative_reward = 0
# simulations = 30

# for simulation in range(simulations):
for t in range(n_rounds):
    chosen_arm = adversarialExp3Environment.select_arm()
    reward = adversarialExp3Environment.assign_reward(chosen_arm)
    adversarialExp3Environment.update(chosen_arm, reward)
        
    cumulative_reward += reward
    optimal_reward = (t + 1) * 0.7
    regret.append(optimal_reward - cumulative_reward)

# class AdversarialArm():
#   def __init__(self, t, active_start, active_end):
#     self.t = t
#     self.active_start = active_start
#     self.active_end = active_end

#   def draw(self):
#     self.t = self.t + 1
#     if self.active_start <= self.t <= self.active_end:
#       return 1.0
#     else:
#       return 0.0

plt.figure(figsize=(10, 6))
plt.plot(regret, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Exp3 Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.show()