import math
import matplotlib.pyplot as plt
import random
import numpy as np

def UCB(arm_means, num_arms, time_horizon, delta):
    optimal_arm = np.argmax(arm_means)
    num_iterations = 1
    regret = np.zeros([time_horizon, num_iterations])
    for iter in range(num_iterations):
        ucb = 100 * np.ones(num_arms)
        emp_means = np.zeros(num_arms)
        num_pulls = np.zeros(num_arms)
        for round in range(time_horizon):
            greedy_arm = np.argmax(ucb)
            reward = np.random.binomial(1, arm_means[greedy_arm])
            num_pulls[greedy_arm] += 1
            regret[round, iter] = arm_means[optimal_arm] - arm_means[greedy_arm]
            emp_means[greedy_arm] += (reward - emp_means[greedy_arm]) / num_pulls[greedy_arm]
            ucb[greedy_arm] = emp_means[greedy_arm] + np.sqrt(2 * np.log(1 / delta) / num_pulls[greedy_arm])
    return regret

class stochastic_OMD_Environment:
    def __init__(self, learning_rate,omd_number_of_arms):
        self.learning_rate = learning_rate
        self.normalization_factor = 200*math.sqrt(10)
        self.estimated_loss_vector = np.zeros(omd_number_of_arms)
        self.number_of_arms = omd_number_of_arms
    
    def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate):
        weights_for_arms = np.zeros(omd_number_of_arms)
        epsilon = 1.0e-9
        previous_normalization_factor = normalization_factor
        updated_normalization_factor = normalization_factor
        while True:
            for arm in range(omd_number_of_arms):
                inner_product = abs((learning_rate * (estimated_loss_vector[arm] - updated_normalization_factor)))
                exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2)
                weight_of_arm = 4 * exponent_of_inner_product
                weights_for_arms[arm] = weight_of_arm  
            sum_of_weights = sum(weights_for_arms)
            numerator = sum_of_weights - 1
            sum_of_arms_taken_to_power = 0
            for arm_weight in range(omd_number_of_arms):
                updated_normalization_factor_arm_weight = math.pow(weights_for_arms[arm_weight], 3/2)
                sum_of_arms_taken_to_power += updated_normalization_factor_arm_weight
            denominator = (learning_rate * sum_of_arms_taken_to_power) + epsilon
            updated_normalization_factor = previous_normalization_factor - (numerator / denominator)
            difference_in_normalization_factors = abs(updated_normalization_factor - previous_normalization_factor)
            previous_normalization_factor = updated_normalization_factor
            if(difference_in_normalization_factors < epsilon):
                break
            else:
                continue
        return weights_for_arms, updated_normalization_factor

    def selectArm(self):
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        action_chosen = np.random.choice(omd_number_of_arms, p = weights_of_arms)
        return action_chosen
    
    def updateLossVector(self, chosen_arm, loss):
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        if weights_of_arms[chosen_arm] > 0:
            new_loss_estimate = loss / weights_of_arms[chosen_arm]
        else:
            new_loss_estimate = 0
        self.estimated_loss_vector[chosen_arm] += new_loss_estimate

num_arms = 10
time_horizon = 100000
arm_means = np.random.uniform(0, 1, num_arms)
delta = 0.005
regret = UCB(arm_means, num_arms, time_horizon, delta)  

ucb_cumulative_regret = np.cumsum(np.mean(regret, axis=1))

# number_of_arms = 10
# time_horizon = 100000
# learning_rate = 0.005

# stochasticOMDEnvironment = stochastic_OMD_Environment(learning_rate, number_of_arms)
# arm_means = np.random.uniform(0, 1, number_of_arms)
# losses_for_arm_in_each_round = []
# regrets = []
# cumulative_loss = 0
# cumulative_optimal_loss = 0

# for round_played in range(time_horizon):
#     chosen_arm = stochasticOMDEnvironment.selectArm()
#     loss_for_arm = 1 - np.random.binomial(1, arm_means[chosen_arm])
#     losses_for_arm_in_each_round.append(loss_for_arm)
#     stochasticOMDEnvironment.updateLossVector(chosen_arm, losses_for_arm_in_each_round[round_played])
#     optimal_arm = np.argmax(arm_means)
#     loss_for_optimal_arm = 1 - np.random.binomial(1, arm_means[optimal_arm])
#     cumulative_loss += loss_for_arm
#     cumulative_optimal_loss += loss_for_optimal_arm
#     regret_for_this_round = cumulative_loss - cumulative_optimal_loss
#     regrets.append(regret_for_this_round)

omd_number_of_arms = 10
omd_time_horizon = 100000
omd_learning_rate = 0.005
omd_num_omd_iterations = 1

stochasticOMDEnvironment = stochastic_OMD_Environment(omd_learning_rate, omd_number_of_arms)
omd_arm_means = np.random.uniform(0, 1, omd_number_of_arms)
omd_losses_for_each_round = []
omd_optimal_arm = np.argmax(omd_arm_means)
#loss_for_omd_optimal_arm = 1 - np.random.binomial(1, optimal_arm)
mean_of_omd_optimal_arm = 1 - omd_arm_means[omd_optimal_arm ]

def find_regret(omd_time_horizon, omd_num_omd_iterations):
    omd_regrets = np.zeros([omd_time_horizon, omd_num_omd_iterations])
    for omd_iter in range(omd_num_omd_iterations):
        for round_played in range(omd_time_horizon):
            omd_chosen_arm = stochasticOMDEnvironment.selectArm()
            omd_loss_for_arm = 1 - np.random.binomial(1, omd_arm_means[omd_chosen_arm])
            mean_of_omd_chosen_arm = 1 - omd_arm_means[omd_chosen_arm]
            omd_losses_for_each_round.append(omd_loss_for_arm)
            stochasticOMDEnvironment.updateLossVector(omd_chosen_arm, omd_losses_for_each_round[round_played])
            omd_regrets[round_played, omd_iter] = mean_of_omd_chosen_arm - mean_of_omd_optimal_arm
    return omd_regrets

omd_regrets_to_plot = find_regret(omd_time_horizon, omd_num_omd_iterations)
omd_cum_regret = np.cumsum(np.mean(omd_regrets_to_plot, axis=1))

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
        
exp3_cumulative_regret = np.cumsum(np.mean(regrets, axis=1))

plt.figure(figsize=(10, 6))
plt.plot(ucb_cumulative_regret, label="UCB Cumulative Regret", color = "blue")
plt.plot(omd_cum_regret, label="OMD Cumulative Regret", color = "green")
plt.plot(exp3_cumulative_regret, label="EXP3 Cumulative Regret", color = "red")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("UCB vs Exp3 vs OMD Stochastic Cumulative Regret")
plt.legend()
plt.grid(True)
plt.show()