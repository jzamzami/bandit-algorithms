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
        self.normalization_factor = 5
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
        weights_for_arms = [0.0 for arm in range(number_of_arms)]
        epsilon = 0.000001
        sum_of_weights = 0
        for arm in range(number_of_arms):
            inner_product = (learning_rate * (estimated_loss_vector[arm] - normalization_factor))
            exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2)
            weight_of_arm = 4 * exponent_of_inner_product
            weights_for_arms[arm] = weight_of_arm
            """problem here with weight of arm being almost 400 when our normalization factor is
            initially 10 (and that like should not be happening, but i did the math manually and
            i also got 400 so idk what im misunderstanding about the algorithm),and i get an even 
            bigger number for the weight if the normalization factor is 0, but again i have no clue 
            why because im just following the algorithm, only thing i can think of is that its 
            (4*inner product)^-2 and not 4(inner product)^-2 but that doesn't really make sense 
            because i feel like it's pretty clear from the pseudocode in the paper that it should 
            be the second choice: what's happening here is that i'm getting inner product to be -0.1 (or 
            0.1 if i had abs which i did but it didnt really make a difference) and then exponent of
            inner product is 100 so the weight becomes 400, and then for the initial normalization factor
            of 0 the weight is an even larger value"""
            for arm_weight in range(len(weights_for_arms)):
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

    def selectArm(self):
        """
        selectArm method for selecting an arm based on our new arm weights
        
        arguments: none
        
        returns:
        1) action_chosen = index of arm chosen
        """
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        action_chosen = drawArm(weights_of_arms)
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
        if weights_of_arms[chosen_arm] > 0:
            new_loss_estimate = loss / weights_of_arms[chosen_arm]
            self.estimated_loss_vector[chosen_arm] += loss
            """this should be self.estimated_loss_vector[chosen_arm] += new_loss_estimate but this returns
            a super weird looking graph so this line is just temporary until i can figure out why the loss estimates
            are being wrongly calculated"""
        else:
            new_loss_estimate = 0
            self.estimated_loss_vector[chosen_arm] += new_loss_estimate

learning_rate = 0.01
number_of_arms = 10
T = 100000
simulations = 30

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