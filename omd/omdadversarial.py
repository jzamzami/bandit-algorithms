import math
import matplotlib.pyplot as plt
import random
import numpy as np

# def drawArm(probabilities_of_choosing_arms):
#     """
#     helper function for selecting arm based off of calculated probabilities
    
#     arguments:
#     1) probabilities_of_choosing_arms => list of probabilities returned from our newton's method
#     for arm weight approximation (this is used in the select_arm function within omd class)
    
#     returns:
#     1) choiceIndex => index of arm to pull
    
#     update: added a check for if the input is a valid probability distribution or not, if it isn't then it raises a value error
#     """
#     rounded_sum_of_probabilities = round(sum(probabilities_of_choosing_arms)) #only reason we're rounding is because sometimes the sum is veryyy close to 1 like 0.9999 so this is just temporary until i know how to make sure the sum is 1 fr
#     choice = random.uniform(0, sum(probabilities_of_choosing_arms))
#     choiceIndex = 0
#     for probability_of_arm in probabilities_of_choosing_arms:
#         if probability_of_arm < 0 or probability_of_arm > 1 or rounded_sum_of_probabilities != 1: #ensures that input is valid probability distribution
#         #if probability_of_arm < 0 or probability_of_arm > 1: #ensures that input is valid probability distribution
#             raise ValueError("This is not a valid probability distribution (you can't pull arm 1 with probability 400)!!")
#         else:
#             choice -= probability_of_arm
#             if choice <= 0:
#                 return choiceIndex
#             choiceIndex += 1
        
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
        self.normalization_factor = 200*math.sqrt(10) #big normalization factor should give better results since it gives us values between 0 and 1
        #self.normalization_factor = 632.46
        #self.estimated_loss_vector = [0.0 for arm in range(number_of_arms)] #initializing the losses as 0 or 1 gives the same results but going to stick with 0 because thats what the paper says
        self.estimated_loss_vector = np.zeros(number_of_arms)
        self.number_of_arms = number_of_arms
        self.best_arm = random.randint(0, number_of_arms - 1)
    
    def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate): #need to fix how normalization factor is getting updated -> the normalization factors here never converge so the code never compiles (infinite while loop moment)
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
        
        update: i think the problem might be with finding the optimal normalization factor because we literally never exit out of the loop (the runtime is insane (longer than 5 mins)) because
        the difference in the previous and updated normalization factors keeps changing and they are *very* different values so idk what the problem is
        """
        #weights_for_arms = [0.0 for arm in range(number_of_arms)]
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
                # """very large weights for arms being found -> for example in the first iteration arm 1 has loss of 0 and the normalization factor is 0
                # so inner product is 0 and exponent of inner product becomes epsilon^-2 which is a huge number like 1.0e18 and then that huge number times 4
                # is an even bigger number -> having a larger normalization factor does result in a smaller weight like if it was 10 intially then we'd get 400 but 
                # that still doesn't fix the issue since the weights are supposed to be probablities so it doesn't make sense for it to be greater than 1 (especially by
                # that much), maybe the calculation of the exponent of the inner product is incorrect? or im not using the correct initial normalization factor
                # update: i think the code following this part is the issue, because if we had a large normalization factor then we get actual probabilities so need to 
                # figure out how to fix the until convergence part and also just general issues that could be present in the second part of the algorithm"""
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
            if(difference_in_normalization_factors < epsilon): #this condition is never met (or takes unbelievably long)
                break
            else:
                continue
        return weights_for_arms, updated_normalization_factor
    
    # def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate): 
    #     """this is just temporary so the code actually runs -> the problem is the normalization factor here isn't updated so we
    #     use the same constant normalization factor for the entire time horizon which means we never find the optimal value that gives
    #     us the "best" probability distribution according to mr newton"""
    #     weights_for_arms = [0.1 for arm in range(number_of_arms)]
    #     epsilon = 1.0e-9
    #     sum_of_weights = 0
    #     for arm in range(number_of_arms):
    #         inner_product = abs((learning_rate * (estimated_loss_vector[arm] - normalization_factor)))
    #         exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2)
    #         weight_of_arm = 4 * exponent_of_inner_product
    #         weights_for_arms[arm] = weight_of_arm
    #         # for arm_weight in range(number_of_arms):
    #         #     sum_of_weights += weights_for_arms[arm_weight]
    #         # numerator = sum_of_weights - 1
    #         # denominator = learning_rate * math.pow(sum_of_weights, 3/2)
    #         # updated_normalization_factor = normalization_factor - (numerator / denominator)
    #         # difference_in_normalization_factors = abs(updated_normalization_factor - normalization_factor)
    #         # if(difference_in_normalization_factors < epsilon):
    #         #     break
    #         # else:
    #         #     continue
    #     return weights_for_arms, normalization_factor
    
    # def normalizingWeights(self, weights_for_arms):
    #     """
    #     method for normalizing weights so they're actually between 0 and 1 and add up to 1 
    #     (like an actual probability distribution lol) -> also didn't have to shift values because no negative weights are being found (thankfully lol)
        
    #     Args:
    #         weights_for_arms (int): probability distribution for how likely we're going to pull each arm

    #     Returns:
    #         weights_for_arms (int): normalized list of weights so theyre actually probabilities
    #     """
    #     weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
    #     sum_of_weights = sum(weights_of_arms)
    #     for arm_weight in range(number_of_arms):
    #         normalized_arm_weight = weights_of_arms[arm_weight] / sum_of_weights
    #         weights_of_arms[arm_weight] = normalized_arm_weight
    #     return weights_of_arms

    def selectArm(self):
        """
        selectArm method for selecting an arm based on our new arm weights
        
        arguments: none
        
        returns:
        1) action_chosen = index of arm chosen
        """
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        #normalized_weights = self.normalizingWeights(weights_of_arms)
        #action_chosen = drawArm(weights_of_arms)
        #action_chosen = drawArm(normalized_weights)
        action_chosen = np.random.choice(number_of_arms, p = weights_of_arms)
        #action_chosen = np.random.choice(number_of_arms, p=normalized_weights)
        return action_chosen
    
    def update_best_arm(self):
        probability = random.random()
        if probability <= 0.35:
            best_arm = random.randint(0, number_of_arms - 1)
        else:
            best_arm = self.best_arm
        return best_arm
    
    # def getLoss(self, chosen_arm):
    #     """
    #     get Loss method for getting the loss of the action we chose to take 
        
    #     arguments:
    #     1) chosen_arm = index of arm we chose to play
        
    #     returns:
    #     1) the loss the agent gets from a specific action
    #     """
    #     if chosen_arm == self.best_arm:
    #         if random.random() < 0.7:
    #             return 1
    #         else:
    #             return 0
    #     else:
    #         if random.random() < 0.3:
    #             return 1
    #         else:
    #             return 0
    
    # def getLoss(self, chosen_arm):
    #     """
    #     get Loss method for getting the loss of the action we chose to take 
        
    #     arguments:
    #     1) chosen_arm = index of arm we chose to play
        
    #     returns:
    #     1) the loss the agent gets from a specific action
    #     """
    #     best_arm = self.update_best_arm()
    #     loss = 0
    #     if chosen_arm == best_arm:
    #         loss += 0
    #     else:
    #         loss += 1
    #     return loss
    
    # def getLoss(self, chosen_arm):
    #     best_arm = self.update_best_arm()
    #     probability = random.random()
    #     loss = 0
    #     if chosen_arm == best_arm:
    #         if probability < 0.7:
    #             loss += 1
    #         else:
    #             loss += 0
    #     else:
    #         if probability < 0.3:
    #             loss += 1
    #         else:
    #             loss += 0
    #     return loss
    
    def updateLossVector(self, chosen_arm, loss_vector):
        """
        update method for updating loss vector after getting losses
        
        arguments:
        1) chosen_arm = index of arm we played in round
        2) loss = loss observed from that action
        
        returns: nothing
        """
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        #normalized_weights = self.normalizingWeights(weights_of_arms)
        if weights_of_arms[chosen_arm] > 0: #this should guarantee that we're only updating arm that's been played
        #if normalized_weights[chosen_arm] > 0:
            new_loss_estimate = loss_vector[chosen_arm] / weights_of_arms[chosen_arm]
            #new_loss_estimate = loss / normalized_weights[chosen_arm]
        else:
            new_loss_estimate = 0
        #self.estimated_loss_vector[chosen_arm] += loss -> what i previously had because adding the loss estimates would result in either a linear graph or graph with negative regret???
        self.estimated_loss_vector[chosen_arm] += new_loss_estimate
        # """this should be self.estimated_loss_vector[chosen_arm] += new_loss_estimate but this returns
        # a super weird looking graph so this line is just temporary, i think the reason it's giving a really weird graph is because of the
        # fact that the weights for arms aren't being found correctly so the losses are either 0 or 1 so this means our new loss estimate is either
        # 0 or 1/W_ti (weight of arm) so if the weights for the arms are really huge numbers dividing those huge numbers by 1 gives us super tiny numbers that
        # approach 0 so the estimated losses are always either 0 or a number very very close to 0 (in theory fixing the weights for the arms should also fix
        # this issue i think/i hope)
        
        # update: adding the loss estimates kind of works now! (not the best looking graph but like better than before)
        # """

# learning_rate = 0.005
# number_of_arms = 10
# time_horizon = 100000
# simulations = 1

# for simulation in range(simulations):
#     omd_adversarial = Adversarial_OMD_Environment(learning_rate, number_of_arms)
#     regrets = []
#     cumulative_loss = 0

#     for round_played in range(time_horizon):
#         chosen_arm = omd_adversarial.selectArm()
#         best_arm = omd_adversarial.update_best_arm()
#         loss = omd_adversarial.getLoss(chosen_arm)
#         cumulative_loss += loss
#         omd_adversarial.updateLossVector(chosen_arm, loss)
#         optimal_loss = (round_played + 1) * 0.3
#         regrets.append(cumulative_loss - optimal_loss)

number_of_arms = 10
time_horizon = 100000
learning_rate = 0.005

adversarialOMDEnvironment = Adversarial_OMD_Environment(learning_rate, number_of_arms)

losses_for_all_rounds = np.zeros((time_horizon, number_of_arms))
for round_played in range(time_horizon):
    losses_for_each_round = np.zeros(number_of_arms)
    best_arm = adversarialOMDEnvironment.update_best_arm()
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
    chosen_arm = adversarialOMDEnvironment.selectArm()
    adversarialOMDEnvironment.updateLossVector(chosen_arm, this_rounds_loss_vector)
    optimal_loss = this_rounds_loss_vector[optimal_arm]
    actual_loss = this_rounds_loss_vector[chosen_arm]
    cumulative_optimal_loss += optimal_loss
    cumulative_loss += actual_loss
    regret_for_this_round = cumulative_loss - cumulative_optimal_loss
    regrets.append(regret_for_this_round)

plt.plot(regrets, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Adversarial Cumulative Regret")
plt.legend()
plt.grid()
plt.show()