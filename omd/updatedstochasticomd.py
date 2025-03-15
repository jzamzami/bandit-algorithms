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

class OMD:
    def __init__(self, number_of_arms, learning_rate, regularizer):
        self.number_of_arms = number_of_arms
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.weights = [1.0] * number_of_arms
        self.normalization_factor = 10
        self.estimated_loss_vector = [0.0] * number_of_arms
        
    def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate):
        weights_for_arms = []
        epsilon = 0.001
        sum_of_weights = 0
        for arm in range(len(estimated_loss_vector)):
            inner_product = learning_rate * (estimated_loss_vector[arm] - normalization_factor)
            exponent_of_inner_product = math.pow(inner_product + epsilon, -2)
            weight_of_arm = 4 * exponent_of_inner_product
            weights_for_arms.append(weight_of_arm)
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

    def select_arm(self):
        self.weights, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        probabilites_of_arms = self.weights
        action_chosen = drawArm(probabilites_of_arms)
        return action_chosen
    
    def update(self, chosen_arm, loss):
        self.weights, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        if self.weights[chosen_arm] > 0:
            x = loss / self.weights[chosen_arm]
        else:
            x = 0
        growth_factor = math.exp((self.learning_rate / self.number_of_arms) * x)
        self.weights[chosen_arm] *= growth_factor
        self.estimated_loss_vector[chosen_arm] += loss

    # w_t = omd_bandit.choose_distribution(t)  # Get probability distribution
    # I_t = omd_bandit.sample_arm(w_t)  # Sample arm
    # loss_t = np.random.binomial(1, true_means[I_t])  # Simulated stochastic loss (Bernoulli)
    # optimal_loss = np.min(true_means)  # Best possible expected loss

    # omd_bandit.update(I_t, loss_t, w_t)

    # regret = true_means[I_t] - optimal_loss
    # cumulative_regret += regret
    # regrets.append(cumulative_regret)
    
np.random.seed(123456)

class StochasticEnvironment:
    def __init__(self, number_of_arms):
        self.number_of_arms = number_of_arms
        self.history = np.zeros(number_of_arms)
        self.arm_means = np.random.uniform(0.1, 0.9, self.number_of_arms)
        self.best_arm = np.argmax(self.arm_means)

    def getLoss(self, chosen_arm):
        self.history[chosen_arm] += 1
        return np.random.binomial(1, self.arm_means[chosen_arm])
        # return 1 - reward

learning_rate = 0.01
number_of_arms = 10
T = 100000
regularizer = 5
simulations = 1

np.random.seed(123456)

for simulation in range(simulations):
    loss_function = StochasticEnvironment(number_of_arms)
    omd = OMD(number_of_arms, learning_rate, regularizer)
    regrets = []
    cumulative_loss = 0

    for t in range(T):
        chosen_arm = omd.select_arm()
        loss = loss_function.getLoss(chosen_arm)
        cumulative_loss += loss
        omd.update(chosen_arm, loss)
        optimal_loss = (t + 1) * (1 - loss_function.arm_means[loss_function.best_arm])
        regrets.append(cumulative_loss - optimal_loss)

plt.plot(regrets, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid()
plt.show()