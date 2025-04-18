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
        epsilon = 0.000001
        sum_of_weights = 0
        for arm in range(len(estimated_loss_vector)):
            inner_product = learning_rate * (estimated_loss_vector[arm] - normalization_factor)
            exponent_of_inner_product = math.pow((abs(inner_product + epsilon)), -2)
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
            new_loss_estimate = loss / self.weights[chosen_arm]
            # self.estimated_loss_vector[chosen_arm] += new_loss_estimate
            self.estimated_loss_vector[chosen_arm] += loss
        else:
            new_loss_estimate = 0
            self.estimated_loss_vector[chosen_arm] += new_loss_estimate
    
np.random.seed(123456)

class StochasticEnvironment:
    def __init__(self, number_of_arms):
        self.number_of_arms = number_of_arms
        self.history = np.zeros(number_of_arms)
        self.arm_means = np.random.uniform(0.1, 0.9, self.number_of_arms)
        self.best_arm = np.argmax(self.arm_means)

    def getLoss(self, chosen_arm):
        self.history[chosen_arm] += 1
        reward = np.random.binomial(1, self.arm_means[chosen_arm])
        return 1 - reward

learning_rate = 0.01
number_of_arms = 10
T = 100000
regularizer = 5
simulations = 30

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
        
# class BernoulliArm():
#   def __init__(self, p):
#     self.p = p
  
#   def draw(self):
#     if random.random() > self.p:
#       return 0.0
#     else:
#       return 1.0

# def test_algorithm(algo, arms, num_sims, horizon):
#   chosen_arms = [0.0 for i in range(num_sims * horizon)]
#   rewards = [0.0 for i in range(num_sims * horizon)]
#   cumulative_rewards = [0.0 for i in range(num_sims * horizon)]
#   sim_nums = [0.0 for i in range(num_sims * horizon)]
#   times = [0.0 for i in range(num_sims * horizon)]
  
#   for sim in range(num_sims):
#     sim = sim + 1
#     algo.initialize(len(arms))
    
#     for t in range(horizon):
#       t = t + 1
#       index = (sim - 1) * horizon + t - 1
#       sim_nums[index] = sim
#       times[index] = t
      
#       chosen_arm = algo.select_arm()
#       chosen_arms[index] = chosen_arm
      
#       reward = arms[chosen_arms[index]].draw()
#       rewards[index] = reward
      
#       if t == 1:
#         cumulative_rewards[index] = reward
#       else:
#         cumulative_rewards[index] = cumulative_rewards[index - 1] + reward
      
#       algo.update(chosen_arm, reward)
  
#   return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]

plt.plot(regrets, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Stochastic Cumulative Regret Over Time")
plt.legend()
plt.grid()
plt.show()