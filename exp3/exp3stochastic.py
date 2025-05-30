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

# n_arms = 10
# time_horizon = 100000
# learning_rate = 0.005
# #this one is shwya 3bee6 mdri leh

# stochasticEXP3Environment = stochastic_Exp3(learning_rate, n_arms)
# # best_arm = stochasticEXP3Environment.update_best_arm()
# arm_means = np.random.uniform(0, 1, n_arms)
# # alpha = np.random.randint(1, n_arms+1, n_arms)
# # arm_means = np.random.dirichlet(alpha, size = 1).squeeze(0) #stolen from github, idk if this is better than just using a uniform distribution for the means
# # arm_means = stochasticEXP3Environment.theseAreOurWeights()
# rewards_for_arm_in_each_round = []
# regrets = []
# # cumulative_reward = 0
# # cumulative_optimal_reward = 0
# cumulative_reward_mean = 0
# cumulative_optimal_reward_mean = 0

# for round_played in range(time_horizon):
#     chosen_arm = stochasticEXP3Environment.select_arm()
#     reward_for_arm = np.random.binomial(1, arm_means[chosen_arm]) #np.random gives us 0 or 1 reward based on probability of success
#     rewards_for_arm_in_each_round.append(reward_for_arm)
#     stochasticEXP3Environment.update(chosen_arm, rewards_for_arm_in_each_round[round_played])
#     optimal_arm = np.argmax(arm_means)
#     # reward_for_optimal_arm = np.random.binomial(1, arm_means[optimal_arm])
#     # cumulative_reward += reward_for_arm
#     # cumulative_optimal_reward += reward_for_optimal_arm
#     # regret_for_this_round = cumulative_optimal_reward - cumulative_reward
#     cumulative_reward_mean += arm_means[chosen_arm]
#     cumulative_optimal_reward_mean += arm_means[optimal_arm]
#     regret_for_this_round = cumulative_optimal_reward_mean - cumulative_reward_mean
#     # regret_for_this_round = arm_means[optimal_arm] - arm_means[chosen_arm]
#     regrets.append(regret_for_this_round)
    
    
# n_arms = 10
# n_rounds = 100000
# learning_rate = 0.01

# stochasticExp3Environment = stochastic_Exp3(learning_rate, n_arms)
# regret = []
# cumulative_reward = 0

# for t in range(n_rounds):
#     chosen_arm = stochasticExp3Environment.select_arm()
#     reward = stochasticExp3Environment.assign_reward(chosen_arm)
#     stochasticExp3Environment.update(chosen_arm, reward)
    
#     cumulative_reward += reward
#     optimal_reward = (t + 1) * 0.7
#     regret.append(optimal_reward - cumulative_reward)


n_arms = 10
time_horizon = 100000
learning_rate = 0.005

stochasticEXP3Environment = stochastic_Exp3(learning_rate, n_arms)
arm_means = np.random.uniform(0, 1, n_arms)
rewards_for_arm_in_each_round = []
num_iterations = 1
regrets = np.zeros([time_horizon, num_iterations])

for iter in range(num_iterations):
    for round_played in range(time_horizon):
        chosen_arm = stochasticEXP3Environment.select_arm()
        reward_for_arm = np.random.binomial(1, arm_means[chosen_arm])
        rewards_for_arm_in_each_round.append(reward_for_arm)
        stochasticEXP3Environment.update(chosen_arm, rewards_for_arm_in_each_round[round_played])
        optimal_arm = np.argmax(arm_means)
        regrets[round_played, iter] = arm_means[optimal_arm] - arm_means[chosen_arm]
        
cumulative_regret = np.cumsum(np.mean(regrets, axis=1))

plt.figure(figsize=(10, 6))
plt.plot(cumulative_regret, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Exp3 stochastic Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.show()