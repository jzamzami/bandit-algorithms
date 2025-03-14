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

class OMD:
    def __init__(self, number_of_arms, learning_rate, regularizer=5):
        self.number_of_arms = number_of_arms
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.weights = [1.0] * number_of_arms
        self.normalization_factor = 0
        self.estimated_loss_vector = [0.0] * number_of_arms

    def select_arm(self):
        # sum_weights = sum(self.weights)
        # probabilities = []
        # for w in self.weights:
        #     probability = w /sum_weights
        #     probabilities.append(probability)
        # return drawArm(probabilities)
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
# simulations = 1000

loss_function = AdversarialEnvironment(number_of_arms)
omd = OMD(number_of_arms, learning_rate)

regrets = []
cumulative_loss = 0

# for simulation in range(simulations):
for t in range(T):
    chosen_arm = omd.select_arm()
    loss = loss_function.getLoss(chosen_arm)
    cumulative_loss += loss
    omd.update(chosen_arm, loss)
        
    optimal_loss = (t + 1) * 0.3
    regrets.append(cumulative_loss - optimal_loss)

plt.plot(regrets, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid()
plt.show()