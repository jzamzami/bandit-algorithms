import random
import numpy as np
import matplotlib.pyplot as plt

#same thing but now with an adversarial environment instead
def UCB(arm_means, num_arms, time_horizon, delta):
    optimal_arm = np.argmax(arm_means)
    num_iterations = 10
    cumulative_regret = np.zeros([time_horizon, num_iterations])
    for iter in range(num_iterations):
        ucb = 100 * np.ones(num_arms)
        emp_means = np.zeros(num_arms)
        num_pulls = np.zeros(num_arms)
        
        for round in range(time_horizon):
            greedy_arm = np.argmax(ucb)
            reward = np.random.uniform(1, arm_means[greedy_arm])
            num_pulls[greedy_arm] += 1 
            cumulative_regret[round, iter] = arm_means[optimal_arm] - arm_means[greedy_arm]
            emp_means[greedy_arm] += (reward - emp_means[greedy_arm]) / num_pulls[greedy_arm]
            ucb[greedy_arm] = emp_means[greedy_arm] + np.sqrt(2 * np.log(1 / delta
        ) / num_pulls[greedy_arm])
    return cumulative_regret

#creating class to simulate adversarial environment
class AdversarialEnvironment:
    def __init__(self, n_arms): #constructor for the number of arms in our setting,
        #the learners history, and what the best arm is
        self.n_arms = n_arms #number of arms is just an input
        self.history = np.zeros(n_arms) #the history starts off as 0 as the learner/agent/whatever
        #hasn't done anything yet -> also the length of the history is litch just how many arms we have
        #since this is meant to help us know how many times the learner chose each arm
        self.best_arm = random.randint(0, n_arms - 1) #choose a random arm to be the best/optimal one
        #this is gonna affect how we assign rewards hehehe

    def assign_reward(self, chosen_arm): #function for assigning rewards to the arms which will be
        #dependent on the history and if the idiot learner chose the best arm or not (shouldve known
        #buddy)
        self.history[chosen_arm] += 1 #the history keeps track of how many times each arm was played
        #so at this step increment our chosen arm's index since it was chosen (in the beginning
        #none were chosen so each index was just 0)
        if chosen_arm == self.best_arm: #if they happened to pick the best arm
            return 1 if random.random() <= 0.7 else 0  #this is pretty nice and if the learner does 
        #choose the best arm then a random number is generated and if that number is less than 0.7
        #then they get a reward -> best arm so they get a reward almost 70% of the time, if its not
        #less than 0.7 then they just dont get a reward
        else:
            return 1 if random.random() <= 0.3 else 0  #if they pick litch any other arm then its the same
        #idea but now they only have around a 30% chance of getting a reward compared to before 

#smth with needing to initialize random and numpy libraries/modules smth?
random.seed(1)
np.random.seed(1)

#same parameters once again
n_arms = 10 
n_rounds = 20000
delta = 0.1

#initializing our adversarial environment hehehe
adversary = AdversarialEnvironment(n_arms)

#initialzing ucb environment which will be used to find our exp3 regret 
arm_means = np.random.uniform(0.1, 0.9, n_arms)  
exp3 = UCB(arm_means, n_arms, n_rounds, delta)

#keeping track of our cummulative regret 
cumulative_regret = []
cumulative_reward = 0

#iterating through number of rounds
for t in range(n_rounds):
    chosen_arm = np.argmax(arm_means)  #same greedy approach
    reward = adversary.assign_reward(chosen_arm)  #but now we get the reward from the adversial function
    #instead of using a uniform distribution
    cumulative_reward += reward #updating cumulative regret
    
    #finding cumulative regret -> diff between optimal arm and chosen arm
    optimal_reward = (t + 1) * 0.7  #here we're assuming we fr get the reward from the optimal arm
    cumulative_regret.append(optimal_reward - cumulative_reward) #add this rounds regret to the
    #array of cumulative regrets

#plotting graphs
plt.figure(figsize=(10, 6))
plt.plot(cumulative_regret, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("UCB Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.show()
