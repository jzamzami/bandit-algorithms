import math
import matplotlib.pyplot as plt
import random
import numpy as np

class stochastic_Exp3:
    def __init__(self, learning_rate, n_arms):
        self.learning_rate = learning_rate
        self.n_arms = n_arms
        self.weights = np.ones(n_arms)
        self.arm_means = np.random.binomial(0.1, 0.9, n_arms)

    def finding_probability_distributions(self):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = np.zeros(n_arms)
        for arm in range(self.n_arms):
            first_term = (1 - self.learning_rate) * (self.weights[arm] / total_weight)
            second_term = (self.learning_rate / n_arms)
            update_rule_for_arm = first_term + second_term
            probs[arm] = update_rule_for_arm
        return probs
    
    def select_arm(self):
        probs = self.finding_probability_distributions()
        action_chosen = np.random.choice(n_arms, p=probs)
        return action_chosen
    
    def assign_reward(self, chosen_arm):
        best_arm = np.argmax(self.arm_means)
        probablity = random.random()
        reward = 0
        if chosen_arm == best_arm:
            if probablity < 0.7:
                reward += 1
            else:
                reward += 0
        else:
            if probablity < 0.3:
                reward += 1
            else:
                reward += 0
        return reward
    
    def update(self, chosen_arm, reward):
        probs = self.finding_probability_distributions()
        if probs[chosen_arm] > 0:
            reward_estimate = reward / probs[chosen_arm]
        else:
            reward_estimate = 0
        growth_factor = math.exp((self.learning_rate / n_arms) * reward_estimate)
        self.weights[chosen_arm] *= growth_factor

n_arms = 10
n_rounds = 100000
learning_rate = 0.01

stochasticExp3Environment = stochastic_Exp3(learning_rate, n_arms)
regret = []
cumulative_reward = 0

for t in range(n_rounds):
    chosen_arm = stochasticExp3Environment.select_arm()
    reward = stochasticExp3Environment.assign_reward(chosen_arm)
    stochasticExp3Environment.update(chosen_arm, reward)
    
    cumulative_reward += reward
    optimal_reward = (t + 1) * 0.7
    regret.append(optimal_reward - cumulative_reward)

plt.figure(figsize=(10, 6))
plt.plot(regret, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Exp3 stochastic Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.show()