import math
import matplotlib.pyplot as plt
import random

def drawArm(probabilities_of_choosing_arms):
    rounded_sum_of_probabilities = round(sum(probabilities_of_choosing_arms)) #only reason we're rounding is because sometimes the sum is veryyy close to 1 like 0.9999 so this is just temporary until i know how to make sure the sum is 1 fr
    choice = random.uniform(0, sum(probabilities_of_choosing_arms))
    choiceIndex = 0
    for probability_of_arm in probabilities_of_choosing_arms:
        if probability_of_arm < 0 or probability_of_arm > 1 or rounded_sum_of_probabilities != 1: #ensures that input is valid probability distribution
            raise ValueError("This is not a valid probability distribution (you can't pull arm 1 with probability 400)!!")
        else:
            choice -= probability_of_arm
            if choice <= 0:
                return choiceIndex
            choiceIndex += 1
        
class Adversarial_OMD_Environment:
    def __init__(self, learning_rate, number_of_arms):
        self.learning_rate = learning_rate
        self.normalization_factor = 1000
        self.estimated_loss_vector = [0.0 for arm in range(number_of_arms)]
        self.number_of_arms = number_of_arms
        self.best_arm = random.randint(0, number_of_arms - 1)
    
    # def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate):
    #     weights_for_arms = [0.1 for arm in range(number_of_arms)]
    #     epsilon = 1.0e-9
    #     previous_normalization_factor = normalization_factor
    #     updated_normalization_factor = normalization_factor
        
    #     while True:
    #         for arm in range(number_of_arms):
    #             inner_product = abs((learning_rate * (estimated_loss_vector[arm] - updated_normalization_factor)))
    #             exponent_of_inner_product = math.pow(((inner_product)), -2)
    #             weight_of_arm = 4 * exponent_of_inner_product
    #             weights_for_arms[arm] = weight_of_arm
            
    #         sum_of_weights = sum(weights_for_arms)
    #         numerator = sum_of_weights - 1
    #         sum_of_arms_taken_to_power = 0
    #         for arm_weight in range(number_of_arms):
    #             updated_normalization_factor_arm_weight = math.pow(weights_for_arms[arm_weight], 3/2)
    #             sum_of_arms_taken_to_power += updated_normalization_factor_arm_weight
            
    #         denominator = (learning_rate * sum_of_arms_taken_to_power) + epsilon
    #         updated_normalization_factor = previous_normalization_factor - (numerator / denominator)
    #         difference_in_normalization_factors = abs(updated_normalization_factor - previous_normalization_factor)
    #         previous_normalization_factor = updated_normalization_factor
            
    #         if(difference_in_normalization_factors < epsilon):
    #             break
    #         else:
    #             continue
            
    #     return weights_for_arms, updated_normalization_factor
    
    def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate):
        weights_for_arms = [0.1 for arm in range(number_of_arms)]
        epsilon = 1.0e-9
        sum_of_weights = 0
        for arm in range(number_of_arms):
            inner_product = abs((learning_rate * (estimated_loss_vector[arm] - normalization_factor)))
            exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2)
            weight_of_arm = 4 * exponent_of_inner_product
            weights_for_arms[arm] = weight_of_arm
            # for arm_weight in range(number_of_arms):
            #     sum_of_weights += weights_for_arms[arm_weight]
            # numerator = sum_of_weights - 1
            # denominator = learning_rate * math.pow(sum_of_weights, 3/2)
            # updated_normalization_factor = normalization_factor - (numerator / denominator)
            # difference_in_normalization_factors = abs(updated_normalization_factor - normalization_factor)
            # if(difference_in_normalization_factors < epsilon):
            #     break
            # else:
            #     continue
        return weights_for_arms, normalization_factor
    
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
        #action_chosen = drawArm(weights_of_arms)
        action_chosen = drawArm(normalized_weights)
        return action_chosen
    
    def getLoss(self, chosen_arm):
        if chosen_arm == self.best_arm:
            if random.random() < 0.7:
                return 1
            else:
                return 0
        else:
            if random.random() < 0.3:
                return 1
            else:
                return 0
    
    def updateLossVector(self, chosen_arm, loss):
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        normalized_weights = self.normalizingWeights(weights_of_arms)
        #if weights_of_arms[chosen_arm] > 0:
        if normalized_weights[chosen_arm] > 0:
            #new_loss_estimate = loss / weights_of_arms[chosen_arm]
            new_loss_estimate = loss / normalized_weights[chosen_arm]
        else:
            new_loss_estimate = 0
        self.estimated_loss_vector[chosen_arm] += new_loss_estimate

learning_rate = 0.01
number_of_arms = 10
time_horizon = 100000
simulations = 30

for simulation in range(simulations):
    omd_adversarial = Adversarial_OMD_Environment(learning_rate, number_of_arms)
    regrets = []
    cumulative_loss = 0

    for round_played in range(time_horizon):
        chosen_arm = omd_adversarial.selectArm()
        loss = omd_adversarial.getLoss(chosen_arm)
        cumulative_loss += loss
        omd_adversarial.updateLossVector(chosen_arm, loss)
        optimal_loss = (round_played + 1) * 0.3
        regrets.append(cumulative_loss - optimal_loss)

plt.plot(regrets, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Class Adversarial Environment Cumulative Regret Over Time")
plt.legend()
plt.grid()
plt.show()