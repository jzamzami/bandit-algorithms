import random
import math
import numpy as np
import matplotlib.pyplot as plt
#just importing libraries lol

#helper function that helps us choose a distribution to sample our actions from
# def categorical_draw(probs): #takes in probabilities 
#     """
#     helper function to fr draw an arm based on our probabilities
#     arguments: probability array thats returned by our algo
#     returns decremented array bc now we're considering one less arm 
#     """
#     z = random.random() #generates floating point number between 0 and 1
#     cum_prob = 0.0 #initialize floating point number as 0 for our probability
#     for i in range(len(probs)): #iterate through our probabilities 
#         prob = probs[i] #our probability is an element of the array
#         cum_prob += prob #also keep track of cumulative probability and add our current probability
#         if cum_prob > z: #if our cummulative probability is greater than the random number generated
#             return i #then we return its position
#     return len(probs) - 1 #decrement the length of our probability array

def categorical_draw(probs):
    """ helper function for pulling an arm after finding the probability distribution(the
    weights for all arms)
    arguments: 1) probabilities_of_choosing_arms = list of each arm's probability of being pulled
    after running the EXP3 algorithm
    returns the arm (index of arm) we're gonna pull
    """
    choice = random.uniform(0, sum((probs)))
    choiceIndex = 0

    for probability_of_arm in (probs):
        choice -= probability_of_arm
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1


class Adversarial_Exp3: #class for our exp3 algortihm -> class lets us have constructors so makes 
    #the environment easy to think about/create
    def __init__(self, learning_rate, n_arms): #initializing our learning which affects how much we 
        #explore/exploit and also array/vector for our loss estimators 
        self.learning_rate = learning_rate
        self.n_arms = n_arms
        self.weights = [1.0] * n_arms
        self.best_arm = random.randint(0, n_arms - 1)

    # def initialize(self, n_arms): 
    #     # self.weights = [1.0 for _ in range(n_arms)] #initializes the weights for the arms to be 1
    #     for arm in range(n_arms):
    #         self.weights.append(1.0)

    def finding_probability_distributions(self):
        pass
    
    def select_arm(self): #function for selecting arms
        n_arms = len(self.weights) #the total number of arms should be the same as the length
        #of our weights array/vector
        total_weight = sum(self.weights) #total weight is the sum of the array
        # probs = [(1 - self.learning_rate) * (self.weights[arm] / total_weight) + 
        #         (self.learning_rate / n_arms) for arm in range(n_arms)] #calculating the 
        #probability of selective a certain arm 
        probs = []
        for arm in range(n_arms):
            update_rule_for_arm = (1 - self.learning_rate) * (self.weights[arm] / total_weight) + (self.learning_rate / n_arms)
            probs.append(update_rule_for_arm)
        action_chosen = categorical_draw(probs)
        return action_chosen #based on this probability we sample an action

    def update(self, chosen_arm, loss): #function for updating our array of loss estimators
        #here we taken in the arm the agent chose and the loss sampled which we need for our update
        n_arms = len(self.weights) #once again the number of arms should be the same
        total_weight = sum(self.weights) #also total weight calculation doesnt change
        # probs = [(1 - self.learning_rate) * (self.weights[arm] / total_weight) + 
        #         (self.learning_rate / n_arms) for arm in range(n_arms)] #same formula for 
        #calculating probability again
        probs = []
        for arm in range(n_arms): #list comprehension getting obliterated again
            update_rule_for_arm = (1 - self.learning_rate) * (self.weights[arm] / total_weight) + (self.learning_rate / n_arms)
            probs.append(update_rule_for_arm)
        # x = loss / probs[chosen_arm] if probs[chosen_arm] > 0 else 0 #loss estimator
        if probs[chosen_arm] > 0:
            loss_estimate = loss / probs[chosen_arm]
        else: #dont return anything (like update losses) if arm isn't chosen
            0
        growth_factor = math.exp((self.learning_rate / n_arms) * loss_estimate) #growth factor
        self.weights[chosen_arm] *= growth_factor #updating based off of the growth factor
        
    def assign_loss(self, chosen_arm):
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

#adversarial environment same as the one defined in the UCB algorithm 

random.seed(1)
np.random.seed(1)

n_arms = 10
n_rounds = 100000
learning_rate = 0.01

# adversary = Adversarial_Exp3(n_arms)
# exp3 = Adversarial_Exp3(learning_rate)

adversarialExp3Environment = Adversarial_Exp3(learning_rate, n_arms)
regret = []
# cumulative_loss = 0
cumulative_loss = 0

for t in range(n_rounds):
    chosen_arm = adversarialExp3Environment.select_arm()
    loss = adversarialExp3Environment.assign_loss(chosen_arm)
    adversarialExp3Environment.update(chosen_arm, loss)
    
    cumulative_loss += loss
    optimal_loss = (t + 1) * 0.7
    regret.append(optimal_loss - cumulative_loss) 
    """i know this says loss but its not using losses its still using rewards so need 
    to change it to losses instead fr """
    
    # chosen_arm = adversarialExp3Environment.select_arm()
    # loss = adversarialExp3Environment.assign_loss(chosen_arm)
    # adversarialExp3Environment.update(chosen_arm, loss)
    
    # cumulative_loss += loss
    # optimal_loss = 1 - ((t + 1) * 0.7)
    # regret.append(cumulative_loss - optimal_loss)

plt.figure(figsize=(10, 6))
plt.plot(regret, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Exp3 Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.show()