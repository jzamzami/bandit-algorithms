import math
import matplotlib.pyplot as plt
import random
import numpy as np

class Adversarial_Exp3:
    def __init__(self, learning_rate, n_arms):
        self.learning_rate = learning_rate
        self.n_arms = n_arms
        self.weights = np.ones(n_arms)
        self.best_arm = random.randint(0, n_arms - 1)

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
    
    def update_best_arm(self):
        probability = random.random()
        if probability <= 0.35:
            best_arm = random.randint(0, n_arms - 1)
        else:
            best_arm = self.best_arm
        return best_arm
    
    def assign_reward(self, chosen_arm):
        best_arm = self.update_best_arm()
        reward = 0
        if chosen_arm == best_arm:
            reward += 1
        else:
            reward += 0
        return reward
    
    def update(self, chosen_arm, reward_vector):
        probs = self.finding_probability_distributions()
        if probs[chosen_arm] > 0:
            reward_estimate = reward_vector[chosen_arm] / probs[chosen_arm]
        else:
            reward_estimate = 0
        growth_factor = math.exp((self.learning_rate / n_arms) * reward_estimate)
        self.weights[chosen_arm] *= growth_factor

n_arms = 10
time_horizon = 10000
learning_rate = 0.005

adversarialExp3Environment = Adversarial_Exp3(learning_rate, n_arms)

rewards_for_all_rounds = np.zeros((time_horizon, n_arms))
for round_played in range(time_horizon):
    rewards_for_each_round = np.zeros(n_arms)
    best_arm = adversarialExp3Environment.update_best_arm()
    for arm in range(n_arms):
        reward_of_arm = 0
        if arm == best_arm:
            reward_of_arm += 1
        else:
            reward_of_arm += 0
        rewards_for_each_round[arm] = reward_of_arm
    rewards_for_all_rounds[round_played] = rewards_for_each_round
    
best_arms_in_each_round = []
for reward_vector in range(len(rewards_for_all_rounds)):
    best_arm_in_this_round = np.argmax(rewards_for_all_rounds[reward_vector])
    best_arms_in_each_round.append(best_arm_in_this_round)
frequency_of_each_best_arm = np.bincount(best_arms_in_each_round)
optimal_arm = np.argmax(frequency_of_each_best_arm)

cumulative_optimal_reward = 0
cumulative_reward = 0
regrets = []

for reward_vector in range(len(rewards_for_all_rounds)):
    for round_played in range(len(rewards_for_all_rounds[reward_vector])):
        round_vector = rewards_for_all_rounds[reward_vector]
        chosen_arm = adversarialExp3Environment.select_arm()
        adversarialExp3Environment.update(chosen_arm, rewards_for_all_rounds[reward_vector])
        optimal_reward = round_vector[optimal_arm]
        actual_reward = round_vector[chosen_arm]
        cumulative_optimal_reward += optimal_reward
        cumulative_reward += actual_reward
        regret_for_this_round = cumulative_optimal_reward - cumulative_reward
        regrets.append(regret_for_this_round)

plt.figure(figsize=(10, 6))
plt.plot(regrets, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Exp3 Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.show()