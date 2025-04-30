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
    
    def update_arm_reward_estimate(self, chosen_arm, reward_vector):
        probs = self.finding_probability_distributions()
        if probs[chosen_arm] > 0:
            reward_estimate = reward_vector[chosen_arm] / probs[chosen_arm]
        else:
            reward_estimate = 0
        growth_factor = math.exp((self.learning_rate / n_arms) * reward_estimate)
        self.weights[chosen_arm] *= growth_factor

n_arms = 10
time_horizon = 100000
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
exp3_regrets = []

for round_reward_vector in range(len(rewards_for_all_rounds)):
    this_rounds_reward_vector = rewards_for_all_rounds[round_reward_vector]
    chosen_arm = adversarialExp3Environment.select_arm()
    adversarialExp3Environment.update_arm_reward_estimate(chosen_arm, this_rounds_reward_vector)
    optimal_reward = this_rounds_reward_vector[optimal_arm]
    actual_reward = this_rounds_reward_vector[chosen_arm]
    cumulative_optimal_reward += optimal_reward
    cumulative_reward += actual_reward
    regret_for_this_round = cumulative_optimal_reward - cumulative_reward
    exp3_regrets.append(regret_for_this_round)

class Adversarial_OMD_Environment:
    def __init__(self, learning_rate, number_of_arms):
        self.learning_rate = learning_rate
        self.normalization_factor = 200*math.sqrt(10)
        self.estimated_loss_vector = np.zeros(number_of_arms)
        self.number_of_arms = number_of_arms
        self.best_arm = random.randint(0, number_of_arms - 1)
    
    def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate):
        weights_for_arms = np.zeros(number_of_arms)
        epsilon = 1.0e-9
        previous_normalization_factor = normalization_factor
        updated_normalization_factor = normalization_factor
        while True:
            for arm in range(number_of_arms):
                inner_product = abs((learning_rate * (estimated_loss_vector[arm] - updated_normalization_factor)))
                exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2)
                weight_of_arm = 4 * exponent_of_inner_product
                weights_for_arms[arm] = weight_of_arm
            sum_of_weights = sum(weights_for_arms)
            numerator = sum_of_weights - 1
            sum_of_arms_taken_to_power = 0
            for arm_weight in range(number_of_arms):
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
        action_chosen = np.random.choice(number_of_arms, p = weights_of_arms)
        return action_chosen
    
    def update_best_arm(self):
        probability = random.random()
        if probability < 0.7:
            best_arm = random.randint(0, number_of_arms - 1)
        else:
            best_arm = self.best_arm
        return best_arm
    
    def updateLossVector(self, chosen_arm, loss_vector):
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        if weights_of_arms[chosen_arm] > 0:
            new_loss_estimate = loss_vector[chosen_arm] / weights_of_arms[chosen_arm]
        else:
            new_loss_estimate = 0
        self.estimated_loss_vector[chosen_arm] += new_loss_estimate

number_of_arms = 10
time_horizon = 100000
learning_rate = 0.005

adversarialExp3Environment = Adversarial_OMD_Environment(learning_rate, number_of_arms)

losses_for_all_rounds = np.zeros((time_horizon, number_of_arms))
for round_played in range(time_horizon):
    losses_for_each_round = np.zeros(number_of_arms)
    best_arm = adversarialExp3Environment.update_best_arm()
    for arm in range(number_of_arms):
        loss_of_arm = 0
        if arm == best_arm:
            loss_of_arm += 0
        else:
            loss_of_arm += 1
        losses_for_each_round[arm] = loss_of_arm
    losses_for_all_rounds[round_played] = losses_for_each_round
    
best_arms_in_each_round = []
for loss_vector in range(len(losses_for_all_rounds)):
    best_arm_in_this_round = np.argmin(losses_for_all_rounds[loss_vector])
    best_arms_in_each_round.append(best_arm_in_this_round)
frequency_of_each_best_arm = np.bincount(best_arms_in_each_round)
optimal_arm = np.argmax(frequency_of_each_best_arm)

cumulative_optimal_loss = 0
cumulative_loss = 0
regrets = []

for round_loss_vector in range(len(losses_for_all_rounds)):
    this_rounds_loss_vector = losses_for_all_rounds[round_loss_vector]
    chosen_arm = adversarialExp3Environment.selectArm()
    adversarialExp3Environment.updateLossVector(chosen_arm, this_rounds_loss_vector)
    optimal_loss = this_rounds_loss_vector[optimal_arm]
    actual_loss = this_rounds_loss_vector[chosen_arm]
    cumulative_optimal_loss += optimal_loss
    cumulative_loss += actual_loss
    regret_for_this_round = cumulative_loss - cumulative_optimal_loss
    regrets.append(regret_for_this_round)

plt.figure(figsize=(10, 6))
plt.plot(exp3_regrets, label="Exp3 Cumulative Regret", color = "red")
plt.plot(regrets, label="OMD Cumulative Regret", color = "blue")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Exp3 vs OMD Adversarial Cumulative Regret")
plt.legend()
plt.grid(True)
plt.show()