import numpy as np
import math
import matplotlib.pyplot as plt
import random

def drawArm(probabilities_of_choosing_arms):
    z = random.random()
    total_probability = 0.0

    for i in range(len(probabilities_of_choosing_arms)):
        prob = probabilities_of_choosing_arms[i]
        total_probability += prob

        if total_probability > z:
            return i
        
    return len(probabilities_of_choosing_arms) - 1

def newtons_approximation_for_arm_weights(normalization_factor, estimated_loss_vector, learning_rate):
    weights_for_arms = []
    epsilon = 0.001
    sum_of_weights = 0

    for arm in range(len(estimated_loss_vector)):
        inner_product = learning_rate * (estimated_loss_vector[arm] - normalization_factor)
        exponent_of_inner_product = math.pow(inner_product + epsilon, -2)
        weight_of_arm = 4 * exponent_of_inner_product
        weights_for_arms.append(weight_of_arm)
        # sum_of_weights = 0
        # for arm_weight in weights_for_arms:
        #     sum_of_weights += weights_for_arms[arm_weight]
        sum_of_weights += weight_of_arm
        numerator = sum_of_weights - 1
        denominator = learning_rate * math.pow(sum_of_weights, 3 / 2)
        updated_normalization_factor = normalization_factor - (numerator / denominator)
        difference_in_normalization_factors = updated_normalization_factor - normalization_factor

        if(difference_in_normalization_factors < epsilon):
            break

    return weights_for_arms


def OMD_for_bandits(regularizer, time_horizon, number_of_arms, loss_function):
    normalization_factor = 0  #temp value
    learning_rate = 0.01  #temp value
    estimated_loss_vector = []
    for arm in range(number_of_arms):
        estimated_loss_vector.append(0.0)

    for round in range(time_horizon):
        probability_distribution = newtons_approximation_for_arm_weights(normalization_factor, estimated_loss_vector, learning_rate)
        action_chosen = drawArm(probability_distribution)
        loss_from_action = loss_function.getLoss(action_chosen)

        for arm in range(number_of_arms):
            if arm == action_chosen:
                loss_estimate_of_arm = estimated_loss_vector[arm] / probability_distribution[arm]
                updated_loss_estimate = estimated_loss_vector[arm] + loss_estimate_of_arm
                estimated_loss_vector.append(updated_loss_estimate)
            else:
                updated_loss_estimate = estimated_loss_vector[arm]
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

# regularizer = 5
# time_horizon = 50000
# number_of_arms = 10
learning_rate = 0.01
number_of_arms = 10
T = 50000

loss_function = AdversarialEnvironment(number_of_arms)

regrets = []
cumulative_loss = 0

for t in range(T):
    estimated_losses = OMD_for_bandits(None, 1, number_of_arms, loss_function)
    probabilities = []
    for arm in range(number_of_arms):
        probabilities.append(1 / number_of_arms)

    chosen_arm = drawArm(probabilities)
    loss = loss_function.getLoss(chosen_arm)
    cumulative_loss += loss
    optimal_loss = (t + 1) * 0.3
    regrets.append(cumulative_loss - optimal_loss)

plt.plot(regrets, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid()
plt.show()