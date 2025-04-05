import numpy as np
import math
import matplotlib.pyplot as plt
import random

def drawArm(probabilities_of_choosing_arms):
    """
    helper function for selecting arm based off of calculated probabilities
    
    arguments:
    1) probabilities_of_choosing_arms => list of probabilities returned from our newton's method
    for arm weight approximation (this is used in the select_arm function within omd class)
    
    returns:
    1) choiceIndex => index of arm to pull
    """
    choice = random.uniform(0, sum(probabilities_of_choosing_arms))
    choiceIndex = 0
    for probability_of_arm in probabilities_of_choosing_arms:
        choice -= probability_of_arm
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1

class Adversarial_OMD_Environment: #adversarial omd class
    def __init__(self, learning_rate, number_of_arms):
        """
        class constructor
        
        arguments:
        1) learning_rate = controls the explore-exploit rate when we get to newtons method
        2) number_of_arms = number of arms available for the agent to choose from
        
        variables initialized:
        1) self.learning_rate = user input
        2) self.normalization_factor = for now is temporarily 10 (i tried other values between 0 and 15, and 10 gave me the best 
        results for some reason)
        3) self.estimated_loss_vector = empty list -> losses are initialized to be 0
        4) self.number_of_arms = user input
        5) self.best_arm = index of best arm that is randomly chosen from our list of arms
        """
        self.learning_rate = learning_rate
        self.normalization_factor = 0
        self.estimated_loss_vector = [0.0 for arm in range(number_of_arms)] #initializing the losses as 0 or 1 gives the same results
        self.number_of_arms = number_of_arms
        self.best_arm = random.randint(0, number_of_arms - 1)
    
    def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate):
        """
        method for finding weights of arms (unfortunately so many bugs with this one)
        
        arguments:
        1) normalization_factor = self.normalization_factor is also just 10, want the normalization factor
        to control how our weights are being calculated and also want to find the optimal normalization factor (until convergence, which
        is until the difference in normalization factors is less than epsilon)
        2) estimated_loss_vector = self.estimated loss vector (want this to be updated after our update method but also important
        since we want to know the loss of a specific arm at a given point so we can also use it for finding our arm weights)
        3) learning_rate = self.learning rate and is just whatever we decide it is (doesn't change and is constant)
        
        returns:
        1) weights_for_arms = want to return a list that contains the weights of each arm so we can then use the weights to sample an action
        2) updated_normalization_factor = want to update our normalization factor with time and return the updated/optimal one
        """
        weights_for_arms = [0.0 for arm in range(number_of_arms)] #having the intial weights as 1/K results in larger regret than initializing them as 0
        epsilon = 1.0e-9
        sum_of_weights = 0
        for arm in range(number_of_arms):
            inner_product = abs((learning_rate * (estimated_loss_vector[arm] - normalization_factor)))
            exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2)
            weight_of_arm = 4 * exponent_of_inner_product
            weights_for_arms[arm] = weight_of_arm
            """very large weights for arms being found -> for example in the first iteration arm 1 has loss of 0 and the normalization factor is 0
            so inner product is 0 and exponent of inner product becomes epsilon^-2 which is a huge number like 1.0e18 and then that huge number times 4
            is an even bigger number -> having a larger normalization factor does result in a smaller weight like if it was 10 intially then we'd get 400 but 
            that still doesn't fix the issue since the weights are supposed to be probablities so it doesn't make sense for it to be greater than 1 (especially by
            that much), maybe the calculation of the exponent of the inner product is incorrect? or im not using the correct initial normalization factor"""
            for arm_weight in range(number_of_arms):
                sum_of_weights += weights_for_arms[arm_weight]
            numerator = sum_of_weights - 1
            denominator = learning_rate * math.pow(sum_of_weights, 3/2)
            updated_normalization_factor = normalization_factor - (numerator / denominator)
            difference_in_normalization_factors = abs(updated_normalization_factor - normalization_factor)
            if(difference_in_normalization_factors < epsilon):
                break
            else:
                continue
        return weights_for_arms, updated_normalization_factor
    
    def normalizingWeights(self, weights_for_arms):
        """_summary_

        Args:
            weights_for_arms (_type_): _description_

        Returns:
            _type_: _description_
        """
        sum_of_weigths = sum(weights_for_arms)
        for arm_weight in range(number_of_arms):
            normalized_arm_weight = arm_weight / sum_of_weigths
            weights_for_arms[arm_weight] = normalized_arm_weight
        return weights_for_arms

    def selectArm(self):
        """
        selectArm method for selecting an arm based on our new arm weights
        
        arguments: none
        
        returns:
        1) action_chosen = index of arm chosen
        """
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        normalized_weights_of_arms = self.normalizingWeights(weights_of_arms)
        action_chosen = drawArm(normalized_weights_of_arms)
        return action_chosen
    
    def getLoss(self, chosen_arm):
        """
        get Loss method for getting the loss of the action we chose to take 
        
        arguments:
        1) chosen_arm = index of arm we chose to play
        
        returns:
        1) the loss the agent gets from a specific action
        """
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
        """
        update method for updating loss vector after getting losses
        
        arguments:
        1) chosen_arm = index of arm we played in round
        2) loss = loss observed from that action
        
        returns: nothing
        """
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        normalized_weights_of_arms = self.normalizingWeights(weights_of_arms)
        if weights_of_arms[chosen_arm] > 0: #this should guarantee that we're only updating arm that's been played 
            new_loss_estimate = loss / weights_of_arms[chosen_arm]
            self.estimated_loss_vector[chosen_arm] += loss
            """this should be self.estimated_loss_vector[chosen_arm] += new_loss_estimate but this returns
            a super weird looking graph so this line is just temporary, i think the reason it's giving a really weird graph is because of the
            fact that the weights for arms aren't being found correctly so the losses are either 0 or 1 so this means our new loss estimate is either
            0 or 1/W_ti (weight of arm) so if the weights for the arms are really huge numbers dividing those huge numbers by 1 gives us super tiny numbers that
            approach 0 so the estimated losses are always either 0 or a number very very close to 0 (in theory fixing the weights for the arms should also fix
            this issue i think/i hope)"""
        else:
            new_loss_estimate = 0
            self.estimated_loss_vector[chosen_arm] += new_loss_estimate
        #self.estimated_loss_vector[chosen_arm] += new_loss_estimate -> should be able to just have this line 

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