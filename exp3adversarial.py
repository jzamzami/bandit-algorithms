import random
import math
import numpy as np
import matplotlib.pyplot as plt
#just importing libraries lol

#helper function that helps us choose a distribution to sample our actions from
def categorical_draw(probs): #takes in probabilities 
    z = random.random() #generates floating point number between 0 and 1
    cum_prob = 0.0 #initialize floating point number as 0 for our probability
    for i in range(len(probs)): #iterate through our probabilities 
        prob = probs[i] #our probability is an element of the array
        cum_prob += prob #also keep track of cumulative probability and add our current probability
        if cum_prob > z: #if our cummulative probability is greater than the random number generated
            return i #then we return its position
    return len(probs) - 1 #decrement the length of our probability array


class Exp3: #class for our exp3 algortihm -> class lets us have constructors so makes 
    #the environment easy to think about/create
    def __init__(self, learning_rate): #initializing our learning which affects how much we 
        #explore/exploit and also array/vector for our reward estimators 
        self.learning_rate = learning_rate
        self.weights = []

    def initialize(self, n_arms): 
        # self.weights = [1.0 for _ in range(n_arms)] #initializes the weights for the arms to be 1
        for arm in range(n_arms):
            self.weights.append(1.0)

    def select_arm(self): #function for selecting arms
        n_arms = len(self.weights) #the total number of arms should be the same as the length
        #of our weights array/vector
        total_weight = sum(self.weights) #total weight is the sum of the array
        # probs = [(1 - self.learning_rate) * (self.weights[arm] / total_weight) + 
        #         (self.learning_rate / n_arms) for arm in range(n_arms)] #calculating the 
        #probability of selective a certain arm 
        probs = []
        for arm in range(n_arms): #list comprehension getting obliterated 
            update_rule_for_arm = (1 - self.learning_rate) * (self.weights[arm] / total_weight) + (self.learning_rate / n_arms)
            probs.append(update_rule_for_arm)
        return categorical_draw(probs) #based on this probability we sample an action

    def update(self, chosen_arm, reward): #function for updating our array of reward estimators
        #here we taken in the arm the agent chose and the reward sampled which we need for our update
        n_arms = len(self.weights) #once again the number of arms should be the same
        total_weight = sum(self.weights) #also total weight calculation doesnt change
        # probs = [(1 - self.learning_rate) * (self.weights[arm] / total_weight) + 
        #         (self.learning_rate / n_arms) for arm in range(n_arms)] #same formula for 
        #calculating probability again
        probs = []
        for arm in range(n_arms): #list comprehension getting obliterated again
            update_rule_for_arm = (1 - self.learning_rate) * (self.weights[arm] / total_weight) + (self.learning_rate / n_arms)
            probs.append(update_rule_for_arm) 
        # x = reward / probs[chosen_arm] if probs[chosen_arm] > 0 else 0 #reward estimator
        if probs[chosen_arm] > 0:
            x = reward / probs[chosen_arm]
        else: #dont return anything (like update losses) if arm isn't chosen
            0
        growth_factor = math.exp((self.learning_rate / n_arms) * x) #growth factor
        self.weights[chosen_arm] *= growth_factor #updating based off of the growth factor

#adversarial environment same as the one defined in the UCB algorithm 
class AdversarialEnvironment:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.history = np.zeros(n_arms)
        self.best_arm = random.randint(0, n_arms - 1)

    def assign_reward(self, chosen_arm):
        self.history[chosen_arm] += 1

        if chosen_arm == self.best_arm:
            return 1 if random.random() < 0.7 else 0
        else:
            return 1 if random.random() < 0.3 else 0

random.seed(1)
np.random.seed(1)

n_arms = 10
n_rounds = 50000
learning_rate = 0.01

adversary = AdversarialEnvironment(n_arms)
exp3 = Exp3(learning_rate)
exp3.initialize(n_arms)

regret = []
cumulative_reward = 0

for t in range(n_rounds):
    chosen_arm = exp3.select_arm()
    reward = adversary.assign_reward(chosen_arm)
    exp3.update(chosen_arm, reward)
    cumulative_reward += reward
    optimal_reward = (t + 1) * 0.7
    regret.append(optimal_reward - cumulative_reward)

plt.figure(figsize=(10, 6))
plt.plot(regret, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Exp3 Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.show()