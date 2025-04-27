import math
import matplotlib.pyplot as plt
import random
import numpy as np
        
class stochastic_OMD_Environment:
    def __init__(self, learning_rate, number_of_arms):
        self.learning_rate = learning_rate
        self.normalization_factor = 200*math.sqrt(10)
        self.estimated_loss_vector = np.zeros(number_of_arms)
        self.number_of_arms = number_of_arms
        self.arm_means = np.random.binomial(0.1, 0.9, number_of_arms)
    
    def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate):
        weights_for_arms = np.zeros(number_of_arms)
        epsilon = 1.0e-9
        previous_normalization_factor = normalization_factor
        updated_normalization_factor = normalization_factor
        for arm in range(number_of_arms):
                    inner_product = abs((learning_rate * (estimated_loss_vector[arm] - updated_normalization_factor)))
                    exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2)
                    weight_of_arm = 4 * exponent_of_inner_product
                    weights_for_arms[arm] = weight_of_arm
        return weights_for_arms, updated_normalization_factor
    
    def normalizingWeights(self, weights_for_arms):
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        sum_of_weights = sum(weights_of_arms)
        for arm_weight in range(number_of_arms):
            normalized_arm_weight = weights_of_arms[arm_weight] / sum_of_weights
            weights_of_arms[arm_weight] = normalized_arm_weight
        return weights_of_arms

    def selectArm(self):
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        normalized_weights = self.normalizingWeights(weights_of_arms)
        action_chosen = np.random.choice(number_of_arms, p=normalized_weights)
        return action_chosen
    
    def getLoss(self, chosen_arm):
        best_arm = np.argmax(self.arm_means)
        probablity = random.random()
        loss = 0
        if chosen_arm == best_arm:
            if probablity < 0.7:
                loss += 1
            else:
                loss += 0
        else:
            if probablity < 0.3:
                loss += 1
            else:
                loss += 0
        return loss
    
    def updateLossVector(self, chosen_arm, loss):
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        normalized_weights = self.normalizingWeights(weights_of_arms)
        if normalized_weights[chosen_arm] > 0:
            new_loss_estimate = loss / normalized_weights[chosen_arm]
        else:
            new_loss_estimate = 0
        self.estimated_loss_vector[chosen_arm] += new_loss_estimate

learning_rate = 0.005
number_of_arms = 10
time_horizon = 100000
simulations = 1

for simulation in range(simulations):
    omd_stochastic = stochastic_OMD_Environment(learning_rate, number_of_arms)
    omd_regret = []
    cumulative_loss = 0

    for round_played in range(time_horizon):
        chosen_arm = omd_stochastic.selectArm()
        loss = omd_stochastic.getLoss(chosen_arm)
        cumulative_loss += loss
        omd_stochastic.updateLossVector(chosen_arm, loss)
        optimal_loss = (round_played + 1) * 0.3
        omd_regret.append(cumulative_loss - optimal_loss)
        
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

#plt.plot(regret, label="exp3_regret", color = "red")
plt.plot(omd_regret, label='omd_regret', color = "blue")
plt.plot(cumulative_regret, label="ucb_regret", color = "green")
plt.xlabel("Round")
plt.ylabel('Cumulative Regret')
plt.title("Stochastic Cumulative Regret Comparison")
plt.legend()
plt.grid()
plt.show()