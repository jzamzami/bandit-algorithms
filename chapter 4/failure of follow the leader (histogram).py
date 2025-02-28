import random
import matplotlib.pyplot as plt


class BernoulliBandit:
    def __init__(self, means):
        self.means = means
        self.totalActionsTaken = []
        if any(x < 0 or x > 1 for x in means):
            raise ValueError('All parameters should be floats in [0, 1].')
        if len(means) < 2:
            raise ValueError("list of means should consist of k >= 2 floats")
    
    def __str__(self):
        return f"{self.means}"
    
    def K(self):
        return len(self.means)

    def pull(self, a):
        if a < 0 or a > self.K() - 1:
            raise ValueError("invalid action")
        self.totalActionsTaken.append(a)
        if random.random() <= self.means[a]:
            return 1
        return 0

    def regret(self, optimal_mean):
        accumulatedRegret = 0
        for a in self.totalActionsTaken:
            regret = optimal_mean - self.means[a]
            accumulatedRegret += regret
        return accumulatedRegret

    def FollowTheLeader(self, n):
        num_arms = self.K()
        frequency_of_each_arm = [0] * num_arms
        sum_of_rewards = [0.0] * num_arms
        for arm in range(num_arms):
            reward = self.pull(arm)
            frequency_of_each_arm[arm] += 1
            sum_of_rewards[arm] += reward
        for t in range(num_arms, n):
            mean_rewards = [
                sum_of_rewards[arm] / frequency_of_each_arm[arm] for arm in range(num_arms)
            ]
            max_mean = max(mean_rewards)
            best_arms = [arm for arm in range(num_arms) if mean_rewards[arm] == max_mean]
            chosen_arm = random.choice(best_arms)
            reward = self.pull(chosen_arm)
            frequency_of_each_arm[chosen_arm] += 1
            sum_of_rewards[chosen_arm] += reward

# Simulation
def simulate_follow_the_leader(means, n, simulations):
    pseudo_regrets = []
    optimal_mean = max(means)
    for _ in range(simulations):
        bandit = BernoulliBandit(means)
        bandit.FollowTheLeader(n)
        regret = bandit.regret(optimal_mean)
        pseudo_regrets.append(regret)
    return pseudo_regrets

# Parameters
means = [0.5, 0.6]
n = 100
simulations = 1000

# Run simulation
pseudo_regrets = simulate_follow_the_leader(means, n, simulations)

# Plot histogram
plt.hist(pseudo_regrets, bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.title('Histogram of Pseudo-Regret for Follow-the-Leader')
plt.xlabel('Pseudo-Regret')
plt.ylabel('Frequency')
plt.show()

"""
(c) Explain the results in the figure:
the histogram shows that this is a good strategy until a certain point in the trials
where the pseudo-regret could increase since our arm wont always give us a reward 
-> multiple trials show the limitation in this strategy as getting to a certain point means that 
we start missing out on other arms that could also have a high chance of giving us rewards, also
exploring once isn't the most accurate way of determining each arm's average, since an arm with 
a high probability could have given us a bad result the first time -> also first point in
histogram has like high regret since that is when we're exploring so yeah theres gonna be regret
"""