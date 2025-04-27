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
learning_rate = 0.005

adversarialExp3Environment = Adversarial_Exp3(learning_rate, n_arms)
regret = []
cumulative_reward = 0

for t in range(n_rounds):
    chosen_arm = adversarialExp3Environment.select_arm()
    best_arm = adversarialExp3Environment.update_best_arm()
    reward = adversarialExp3Environment.assign_reward(chosen_arm)
    adversarialExp3Environment.update(chosen_arm, reward)
    
    cumulative_reward += reward
    optimal_reward = (t + 1) * 0.7
    regret.append(optimal_reward - cumulative_reward)

"""
variables needed (all the code here is just pseudocode so i can think of how to find these values):
1) reward vector -> array of arrays that contains the rewards of each arm from every round:
    reward_vector = []
    reward_vector_for_each_round = []
    for round in range(time_horizon):
        for arm in range(number_of_arms):
            reward = get_reward
            reward_vector_for_each_round.append(reward)
        reward_vector.append(reward_vector_for_each_round)
    return reward_vector
    
2) optimal arm -> index of arm that consistently gives us best rewards (shloon i find optimal arm from this
reward vector):
best_arm_in_each_round = []
for round in range(len(reward_vector_for_each_round)):
    max_reward_in_this_round = max(reward_vector)
    best_arm_in_round = reward_vector_for_each_round.index(max_reward_in_this_round)
    best_arm_in_each_round.append(best_arm_in_round)
best_arms_overall = np.bincount(best_arm_in_each_round)
best_arm_overall = np.argmax(best_arms_overall) #best_arm_overall would then just be our optimal arm
return best_arm_overall

3) rewards of that optimal arm -> so now that we have our optimal arm lazem we go back in time to see what rewards that optimal arm
would have given us (and like actual regret calculation):
cumulative_best_reward = 0
cumulative_actual_reward = 0
regrets = []
for round in range(time_horizon):
    best_reward = reward_vector_for_each_round[best_arm_overall]
    actual_reward = reward_vector_for_each_round[arm_pulled]
    cumulative_best_reward += best_reward
    cumulative_actual_reward += actual_reward
    regret_for_this_round = cumulative_best_reward - cumulative_actual_reward
    regrets.append(regret_for_this_round)
return regrets
"""

plt.figure(figsize=(10, 6))
plt.plot(regret, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Exp3 Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.show()