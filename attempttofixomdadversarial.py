import numpy as np
import math
import matplotlib.pyplot as plt
import random

# def drawArm(probabilities_of_choosing_arms): 
#     """helper function for drawing arms based on the probability distribution our algorithm returns
#     arguments: 1) probabilities_of_choosing_arms = list containing the probability of picking each arm
#     """
#     z = random.random()
#     total_probability = 0.0

#     for i in range(len(probabilities_of_choosing_arms)):
#         prob = probabilities_of_choosing_arms[i]
#         total_probability += prob

#         if total_probability > z:
#             return i
        
#     return len(probabilities_of_choosing_arms) - 1

def drawArm(probabilities_of_choosing_arms):
    """ helper function for pulling an arm after finding the probability distribution(the
    weights for all arms)
    arguments: 1) probabilities_of_choosing_arms = list of each arm's probability of being pulled
    after running the OMD algorithm
    returns the arm (index of arm) we're gonna pull
    """
    choice = random.uniform(0, sum(probabilities_of_choosing_arms))
    choiceIndex = 0

    for probability_of_arm in probabilities_of_choosing_arms:
        choice -= probability_of_arm
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1

def newtons_approximation_for_arm_weights(normalization_factor, estimated_loss_vector, learning_rate):
    """ helper function for calculating the probability distribution/weights of arms using newton's
    method approximation
    arguments: 1) normalization_factor (x) = what we're aiming to approximate so we can better find
    the weights for each arm 
    -> in general just kind of means a factor that we use to like prepare our data so that 
    they're kind of like using the same scale (so like if we wanna scale by 2 or 5 or 10),
    we start with some normalization factor then keep using the factors from previous iterations
    to reach our optimal solution (kind of like dynamic programming but not really but like
    same vibes)
    2) estimated_loss_vector (l_hat) = the loss vector that's returned from running the omd algorithm
    3) learning_rate (eta) = input between 0 and 1 that controls the exploration/exploitation ratio

    returns weights of arms/probability distribution (need to fix inconsistency between using
    weights and probabilities but like shouldnt be a big deal i hope)
    """
    weights_for_arms = [] #initialize empty weights list 
    epsilon = 0.001 #epsilon so we have the until convergence condition met (what the difference
    #to be as small as possible)
    sum_of_weights = 0 #initialize sum as 0

    for arm in range(len(estimated_loss_vector)): #iterating through all possible arms so we can find
        #each of their weights/probabilities
        inner_product = learning_rate * (estimated_loss_vector[arm] - normalization_factor) #formula
        exponent_of_inner_product = math.pow(inner_product + epsilon, -2) #also just formula -> the plus epsilon
        #is to avoid division by 0 but im not sure why inner product would be 0 :(
        weight_of_arm = 4 * exponent_of_inner_product #fr finding the weight of our arm
        weights_for_arms.append(weight_of_arm) #weight of arm added to our list of arm weights
        sum_of_weights = 0 #initialize sum of weights as 0
        for arm_weight in range(len(weights_for_arms)): #iterating through list of arm weights
            sum_of_weights += weights_for_arms[arm_weight] # add each arm weight to our sum of arm weights
        numerator = sum_of_weights - 1 #formula
        denominator = learning_rate * math.pow(sum_of_weights, 3/2) #formula
        updated_normalization_factor = normalization_factor - (numerator / denominator) #formula for finding
        #updated normalization factor (x)
        difference_in_normalization_factors = updated_normalization_factor - normalization_factor #this is 
        #to check for until convergence condition
        if(difference_in_normalization_factors < epsilon): #if tiny difference then we can stop
            #-> until convergence is until we get what we expect so thats what this checks for
            break
        else: #if not just continue repeating calculations
            continue
    return weights_for_arms #return our weights for arms

"""possible issue here (could be misunderstanding the logic of this algorithm tho)
    is that we keep finding normalization factors but we're not really using them in further iterations
    so need to figure out way to store x from our first run such that we can use it in our next iteration
    (kind of like dynamic programming??? (or maybe this is just dynamic programming) either
    way 311 throwback fr)so this could be a potential issue since we're not finding the optimal 
    value of x also why is inner product ever zero? like x could be 0 but unless we have an optimal arm
    we wouldnt really have l_hat be 0 but maybe thats why
"""


def OMD_for_bandits(regularizer, time_horizon, number_of_arms, loss_function, number_of_simulations):
    """our amazing online mirror descent for bandits algorithm <3
    arguments: 1) regularizer (psi) = used to control the model by balancing overfitting and
    underfitting (geeksforgeeks explanation) -> don't really get how it's applicable
    here since we're aleady using newton's method for finding our weight estimates 
    2) time_horizon(n) = how many total rounds r we gonna play
    3) loss_function = how is our adversary gonna generate losses
    4) number_of_simulations = how many times we run the same experiment to reduce the noise
    and try to prevent randomness
    """
    normalization_factor = 0  #temp value -> supposed to find its optimal value by using newton's
    #approximation method (like this value is supposed to update)
    learning_rate = 0.01  #temp value -> just some random eta
    estimated_loss_vector = [] #initialize list to store estimated losses
    for simulation in range(number_of_simulations): #repeat process for however many simulations
        for arm in range(number_of_arms): #couldve used a list comphrension thing here
            estimated_loss_vector.append(0.0) #anyways initialize each arms estimate to be 0

        for round in range(time_horizon): #iterating through number of rounds in our time horizon
            probability_distribution = newtons_approximation_for_arm_weights(normalization_factor, 
            estimated_loss_vector, learning_rate) #find probability distribution using helper function
            """something is definitely wrong with the newton's approximation method because the same graph
            generates no matter what the normalization factor is like 0 and 500 yield almost the same results
            which should def not be happening
            """
            action_chosen = drawArm(probability_distribution) #use helper function to determine which 
            #action to take
            loss_from_action = loss_function.getLoss(action_chosen) #find loss from that action
            #(finding loss also requires using helpers but those arr defined later below)

            for arm in range(number_of_arms): #update estimated loss vector after the round
                if arm == action_chosen: #for the arm chosen we go through the process of using importance
                    #weighted estimators to update our loss estimate
                    old_loss_estimate = estimated_loss_vector[arm] #store our loss estimate of the arm
                    loss_estimate_of_arm = estimated_loss_vector[arm] / probability_distribution[arm] #this is finding new loss estimate
                    updated_loss_estimate = old_loss_estimate + loss_estimate_of_arm #we take the new and add it to the old
                    #to update the total loss estimate
                    estimated_loss_vector.append(updated_loss_estimate) #value is added to loss estimate
                    """problem here could be something about the fact that we are just adding the updated loss estimate
                    instead of like fr updating its value so this is something that needs to get fixed ->
                    look at how in the exp3 class we had update and select arm functions something like that could
                    probably help fix this issue -> what we're doing here essentially is just finding new loss estimates
                    and adding them to our vector such that we just end up with an unbelievably long vector 
                    -> so if we started with [0,0,0] and arm 1 had updated loss estimate 0.4 we'll now have
                    [0,0,0,0.4] instead of [0,0.4,0] so this is something else that needs to get fixed 
                    """
                # else: 
                    # updated_loss_estimate = estimated_loss_vector[arm]
                    # estimated_loss_vector.append(updated_loss_estimate)
                    """this was just unnecessary since we were adding the same value again instead we
                    should keep it the same value -> doesn't change so if we're iterating through our estimates
                    again for example and updating each value then this would have to add 0 to our og value
                    or just not touch it
                    """
        
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
T = 100000
number_of_simulations = 10000
temp_regularizer = 5

loss_function = AdversarialEnvironment(number_of_arms)

regrets = []
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
    regrets.append(cumulative_loss - optimal_loss)

plt.plot(regrets, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid()
plt.show()