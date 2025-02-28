import numpy as np
import matplotlib.pyplot as plt
import random

class BernoulliBandit:
    def __init__(self, means):
        self.means = means
        self.totalActionsTaken = []
        if any(x < 0 or x > 1 for x in means):
            raise ValueError('All parameters should be floats in [0, 1].')
        if len(means) < 2:
            raise ValueError("list of means should consist of k >= 2 floats")
    
    def K(self):
        return len(self.means)

    def pull(self, a):
        if a < 0 or a > self.K() - 1:
            raise ValueError("invalid action")
        self.totalActionsTaken.append(a)
        return 1 if random.random() <= self.means[a] else 0

    def regret(self):
        max_mean = max(self.means)
        return sum(max_mean - self.means[a] for a in self.totalActionsTaken)

    def FollowTheLeader(self, n, initial_exploration=1):
        num_arms = self.K()
        frequency_of_each_arm = [0] * num_arms
        sum_of_rewards = [0.0] * num_arms

        # Enhanced Initial Exploration
        for arm in range(num_arms):
            for _ in range(initial_exploration):
                reward = self.pull(arm)
                frequency_of_each_arm[arm] += 1
                sum_of_rewards[arm] += reward

        for t in range(num_arms * initial_exploration, n):
            mean_rewards = [
                sum_of_rewards[arm] / frequency_of_each_arm[arm]
                for arm in range(num_arms)
            ]
            max_mean = max(mean_rewards)
            best_arms = [arm for arm in range(num_arms) if mean_rewards[arm] == max_mean]
            chosen_arm = random.choice(best_arms)
            reward = self.pull(chosen_arm)
            frequency_of_each_arm[chosen_arm] += 1
            sum_of_rewards[chosen_arm] += reward

# Experiment Parameters
means = [0.5, 0.6]
horizons = range(100, 1100, 100)
num_simulations = 1000  # Increased simulations
average_regrets = []
error_bars = []

for n in horizons:
    regrets = []
    for _ in range(num_simulations):
        bandit = BernoulliBandit(means)
        bandit.FollowTheLeader(n, initial_exploration=5)  # Enhanced exploration
        regrets.append(bandit.regret())
    avg_regret = np.mean(regrets)
    average_regrets.append(avg_regret)
    error_bars.append(np.std(regrets) / np.sqrt(num_simulations))  # Standard Error

# Plotting the Results
plt.errorbar(horizons, average_regrets, yerr=error_bars, fmt='o-', capsize=5, label="Follow-The-Leader")
plt.xlabel("Horizon (n)")
plt.ylabel("Average Regret")
plt.legend()
plt.grid()
plt.show()

"""
(c) Explain the plot. Do you think follow-the-leader is a good algorithm?
Why/why not?
follow-the-leader isn't necessarily a good algoritm due to the limitations explained in the 
bar graph analysis, but even this plot shows that after a certain number of rounds our regret 
starts to grow since we're missing out on potentially optimal arms by continuing to use this arm
-> again one round is not enough to determine each arm's mean distribution of rewards and this plot
shows how as we increase the number of trials our regret also increases 
"""