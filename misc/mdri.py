import numpy as np
import matplotlib.pyplot as plt

def UCB(arm_means, num_arms, time_horizon, delta):
    optimal_arm = np.argmax(arm_means)
    num_iterations = 30
    regret = np.zeros([time_horizon, num_iterations])

    for iter in range(num_iterations):
        ucb = 100 * np.ones(num_arms)
        emp_means = np.zeros(num_arms) 
        num_pulls = np.zeros(num_arms)
        for round in range(time_horizon):
            greedy_arm = np.argmax(ucb)
            reward = np.random.uniform(1, arm_means[greedy_arm])
            num_pulls[greedy_arm] += 1 
            regret[round, iter] = arm_means[optimal_arm] - arm_means[greedy_arm]
            emp_means[greedy_arm] += (reward - emp_means[greedy_arm]) / num_pulls[greedy_arm]
            ucb[greedy_arm] = emp_means[greedy_arm] + np.sqrt(2 * np.log(1 / delta) / num_pulls[greedy_arm])
    return regret

num_arms = 10 
time_horizon = 100000
arm_means = np.random.uniform(0.1, 0.9, num_arms)
delta = 0.1

regret = UCB(arm_means, num_arms, time_horizon, delta)  

cumulative_regret = np.cumsum(np.mean(regret, axis=1))

plt.figure(figsize=(10, 6))
plt.plot(cumulative_regret, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("UCB Stochastic Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.show()