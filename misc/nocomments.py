import math
import matplotlib.pyplot as plt
import random

def drawArm(probabilities_of_choosing_arms):
    choice = random.uniform(0, sum(probabilities_of_choosing_arms))
    choiceIndex = 0
    for probability_of_arm in probabilities_of_choosing_arms:
        if probability_of_arm < 0 or probability_of_arm > 1:
            raise ValueError("These are not probabilities!! (you can't pull arm 1 with probability 100000)")
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
    
    def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate):
        weights_for_arms = [0.0 for arm in range(number_of_arms)]
        epsilon = 1.0e-9
        sum_of_weights = 0
        
        for arm in range(number_of_arms):
            inner_product = abs((learning_rate * (estimated_loss_vector[arm] - normalization_factor)))
            exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2)
            weight_of_arm = 4 * exponent_of_inner_product
            weights_for_arms[arm] = weight_of_arm
            
            sum_of_weights += weights_for_arms[arm]
            numerator = sum_of_weights - 1
            denominator = (learning_rate * math.pow(sum_of_weights, 3/2)) + epsilon
            updated_normalization_factor = normalization_factor - (numerator / denominator)
            difference_in_normalization_factors = abs(updated_normalization_factor - normalization_factor)

            if(difference_in_normalization_factors < epsilon):
                break
            else:
                continue
        return weights_for_arms, updated_normalization_factor

    def selectArm(self):
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        action_chosen = drawArm(weights_of_arms)
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
        if weights_of_arms[chosen_arm] > 0:
            new_loss_estimate = loss / weights_of_arms[chosen_arm]
        else:
            new_loss_estimate = 0
        self.estimated_loss_vector[chosen_arm] += loss

learning_rate = 0.01
number_of_arms = 10
T = 100000
simulations = 1

for simulation in range(simulations):
    omd_adversarial = Adversarial_OMD_Environment(learning_rate, number_of_arms)
    regrets = []
    cumulative_loss = 0

    for t in range(T):
        chosen_arm = omd_adversarial.selectArm()
        loss = omd_adversarial.getLoss(chosen_arm)
        cumulative_loss += loss
        omd_adversarial.updateLossVector(chosen_arm, loss) 
        optimal_loss = (t + 1) * 0.3
        regrets.append(cumulative_loss - optimal_loss)

plt.plot(regrets, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Class Adversarial Environment Cumulative Regret Over Time")
plt.legend()
plt.grid()
plt.show()