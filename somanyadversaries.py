import numpy as np
import math
import matplotlib.pyplot as plt
import random

def drawArm(probabilities_of_choosing_arms):
    choice = random.uniform(0, sum(probabilities_of_choosing_arms))
    choiceIndex = 0

    for probability_of_arm in probabilities_of_choosing_arms:
        choice -= probability_of_arm
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1

def newtons_approximation_for_arm_weights(normalization_factor, estimated_loss_vector, learning_rate):
    weights_for_arms = []
    epsilon = 0.001
    sum_of_weights = 0
    for arm in range(len(estimated_loss_vector)): 
        inner_product = learning_rate * (estimated_loss_vector[arm] - normalization_factor)
        exponent_of_inner_product = math.pow(inner_product + epsilon, -2)
        weight_of_arm = 4 * exponent_of_inner_product
        weights_for_arms.append(weight_of_arm)
        sum_of_weights = 0 
        for arm_weight in range(len(weights_for_arms)):
            sum_of_weights += weights_for_arms[arm_weight]
        numerator = sum_of_weights - 1 
        denominator = learning_rate * math.pow(sum_of_weights, 3/2)
        updated_normalization_factor = normalization_factor - (numerator / denominator)
        difference_in_normalization_factors = updated_normalization_factor - normalization_factor
        if(difference_in_normalization_factors < epsilon):
            break
        else:
            continue
    return weights_for_arms, updated_normalization_factor

def OMD_for_bandits(regularizer, time_horizon, number_of_arms, loss_function, number_of_simulations):
    normalization_factor = 0
    learning_rate = 0.01
    estimated_loss_vector = []
    for simulation in range(number_of_simulations):
        for arm in range(number_of_arms):
            estimated_loss_vector.append(0.0)
        for round in range(time_horizon):
            probability_distribution, normalization_factor = newtons_approximation_for_arm_weights(normalization_factor, estimated_loss_vector, learning_rate)
            action_chosen = drawArm(probability_distribution)
            loss_from_action = loss_function.getLoss(action_chosen)
            for arm in range(number_of_arms):
                if arm == action_chosen:
                    old_loss_estimate = estimated_loss_vector[arm]
                    loss_estimate_of_arm = estimated_loss_vector[arm] / probability_distribution[arm]
                    updated_loss_estimate = old_loss_estimate + loss_estimate_of_arm
                    estimated_loss_vector.append(updated_loss_estimate)
        return estimated_loss_vector


class AdversarialEnvironment:
    def __init__(self, number_of_arms):
        self.number_of_arms = number_of_arms
        self.history = np.zeros(number_of_arms)
        self.best_arm = random.randint(0, number_of_arms - 1)

    def getLoss(self, chosen_arm):
        self.history[chosen_arm] += 1
        if chosen_arm == self.best_arm:
            return 1 if random.random() < 0.7 else 0
        else:
            return 1 if random.random() < 0.3 else 0


learning_rate = 0.01
number_of_arms = 10
T = 100000
number_of_simulations = 10000
temp_regularizer = 5

loss_function = AdversarialEnvironment(number_of_arms)

ugly_graph_omd_regrets = []
cumulative_loss = 0

for t in range(T):
    estimated_losses = OMD_for_bandits(temp_regularizer, 1, number_of_arms, loss_function, number_of_simulations)
    probabilities = []
    for arm in range(number_of_arms):
        probabilities.append(1 / number_of_arms)

    chosen_arm = drawArm(probabilities)
    loss = loss_function.getLoss(chosen_arm)
    cumulative_loss += loss
    optimal_loss = (t + 1) * 0.3
    ugly_graph_omd_regrets.append(cumulative_loss - optimal_loss)


class OMD:
    def __init__(self, number_of_arms, learning_rate, regularizer=5):
        self.number_of_arms = number_of_arms
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.weights = [1.0] * number_of_arms
        self.normalization_factor = 0
        self.estimated_loss_vector = [0.0] * number_of_arms

    def select_arm(self):
        self.weights, self.normalization_factor = newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        probabilites_of_arms = self.weights
        action_chosen = drawArm(probabilites_of_arms)
        return action_chosen

    def update(self, chosen_arm, loss):
        if self.weights[chosen_arm] > 0:
            x = loss / self.weights[chosen_arm]
        else:
            x = 0
        growth_factor = math.exp((self.learning_rate / self.number_of_arms) * x)
        self.weights[chosen_arm] *= growth_factor
        self.estimated_loss_vector[chosen_arm] += loss
        self.weights, self.normalization_factor = newtons_approximation_for_arm_weights(
            self.normalization_factor, self.estimated_loss_vector, self.learning_rate
    )

class AdversarialEnvironment:
    def __init__(self, number_of_arms):
        self.number_of_arms = number_of_arms
        self.history = np.zeros(number_of_arms)
        self.best_arm = random.randint(0, number_of_arms - 1)

    def getLoss(self, chosen_arm):
        self.history[chosen_arm] += 1
        if chosen_arm == self.best_arm:
            return 1 if random.random() < 0.7 else 0
        else:
            return 1 if random.random() < 0.3 else 0


learning_rate = 0.01
number_of_arms = 10
T = 100000 
loss_function = AdversarialEnvironment(number_of_arms)
omd = OMD(number_of_arms, learning_rate)

omd_regrets = []
cumulative_loss = 0

for t in range(T):
    chosen_arm = omd.select_arm()
    loss = loss_function.getLoss(chosen_arm)
    cumulative_loss += loss
    omd.update(chosen_arm, loss)
    
    optimal_loss = (t + 1) * 0.3
    omd_regrets.append(cumulative_loss - optimal_loss)

def categorical_draw(probs):
    choice = random.uniform(0, sum((probs)))
    choiceIndex = 0
    for probability_of_arm in (probs):
        choice -= probability_of_arm
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1


class Exp3:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weights = []

    def initialize(self, n_arms): 
        for arm in range(n_arms):
            self.weights.append(1.0)

    def select_arm(self):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = []
        for arm in range(n_arms):
            update_rule_for_arm = (1 - self.learning_rate) * (self.weights[arm] / total_weight) + (self.learning_rate / n_arms)
            probs.append(update_rule_for_arm)
        return categorical_draw(probs)

    def update(self, chosen_arm, reward):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = []
        for arm in range(n_arms):
            update_rule_for_arm = (1 - self.learning_rate) * (self.weights[arm] / total_weight) + (self.learning_rate / n_arms)
            probs.append(update_rule_for_arm) 
        if probs[chosen_arm] > 0:
            x = reward / probs[chosen_arm]
        else:
            0
        growth_factor = math.exp((self.learning_rate / n_arms) * x)
        self.weights[chosen_arm] *= growth_factor

class AdversarialEnvironment:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.history = np.zeros(n_arms)
        self.best_arm = random.randint(0, n_arms - 1)
    def assign_reward(self, chosen_arm):
        self.history[chosen_arm] += 1
        if chosen_arm == self.best_arm:
            return 1 if random.random() < 0.7 else 0
        else:
            return 1 if random.random() < 0.3 else 0

random.seed(1)
np.random.seed(1)

n_arms = 10
n_rounds = 100000
learning_rate = 0.01

adversary = AdversarialEnvironment(n_arms)
exp3 = Exp3(learning_rate)
exp3.initialize(n_arms)

regret = []
cumulative_reward = 0

for t in range(n_rounds):
    chosen_arm = exp3.select_arm()
    reward = adversary.assign_reward(chosen_arm)
    exp3.update(chosen_arm, reward)
    cumulative_reward += reward
    optimal_reward = (t + 1) * 0.7
    regret.append(optimal_reward - cumulative_reward)

plt.plot(ugly_graph_omd_regrets, label='OMD_for_bandits method', color='blue')
plt.plot(omd_regrets, label='OMD Class', color='red')
plt.plot(regret, label="EXP3 Regret", color ="green")
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Adversarial Cumulative Regret Comparison")
plt.legend()
plt.grid()
plt.show()