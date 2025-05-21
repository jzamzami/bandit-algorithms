import numpy as np
import matplotlib.pyplot as plt
# importing libraries

#UCB function definition with following parameters, the mean of the arms, the number of arms
#the time time horizon, and our exploration parameter delta
def UCB(arm_means, num_arms, time_horizon, delta):
    optimal_arm = np.argmax(arm_means) #best arm is obvi the one with max mean (but the agent
    #doesnt know what it is </3)
    num_iterations = 1  #number of times we repeat our experiment so its less random
    regret = np.zeros([time_horizon, num_iterations]) #initialize regret array to be 0 
    #this 2-d array is meant to store the regret for a given time horizon and the number of iterations

    for iter in range(num_iterations): #looping through each iteration
        ucb = 100 * np.ones(num_arms) #initializng the upper confidence bound for each arm to be 100
        #(remember that ucb starts off super optimistic then we adjust it based off what we find)
        emp_means = np.zeros(num_arms) #all arms start off with 0 recorded means (they all start
        #off as 0 when they havent been played yet) -> this is supposed to be the means we find
        #when we've fr started playing
        num_pulls = np.zeros(num_arms) #the number of pulls is also initialized to be 0 duh
        for round in range(time_horizon): #iterating through the rounds in the time horizon
            greedy_arm = np.argmax(ucb) #the arm with the higest upper confidence bound will be the
            #that's chosen most of the time -> whole point of the algo (thats also why its called the
            #"greedy arm" since in theory the arm with the highest ucb should be the best one,
            #since the agent doesn't know the optimal arm this is their way of having a best arm)
            reward = np.random.binomial(1, arm_means[greedy_arm]) #generating a random uniformly distributed
            #reward for the greedy arm
            num_pulls[greedy_arm] += 1 #increment the number of times the arm has been pulled
            regret[round, iter] = arm_means[optimal_arm] - arm_means[greedy_arm] #now we can fr 
            #find the regret for a given round and iteration which is just the reward from the 
            #optimal arm - the one of the greedy arm
            emp_means[greedy_arm] += (reward - emp_means[greedy_arm]) / num_pulls[greedy_arm]
            #we also need to update the recorded mean of the greedy arm -> this is an important step
            #since the recorded means are what we use to find the ucb and select the best arms
            ucb[greedy_arm] = emp_means[greedy_arm] + np.sqrt( 2 * np.log(1 / delta) / num_pulls[greedy_arm])
            #ucb calculations!
    return regret #need the regret to show the cummulative regret

#parameters for our simulation of the UCB algorithm
num_arms = 10  #number of arms
time_horizon = 100  #time horizon -> basically how many rounds are we playing in total
arm_means = np.random.uniform(0, 1, num_arms)
# alpha = np.random.randint(1, num_arms+1, num_arms)
# arm_means = np.random.dirichlet(alpha, size = 1).squeeze(0)
# arm_means = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
#generate uniformly distributed arm means
#between 0.1-0.9
# arm_means = np.random.binomial(num_arms, )
delta = 0.005 #tiny exploration parameter -> explore like 10% of the time

#running the algorithm to find the regret for this given simulating
regret = UCB(arm_means, num_arms, time_horizon, delta) 
print(regret)
print(np.mean(regret, axis=1))

#finding the cummulative regret in order to graph it
# cumulative_regret = np.cumsum(np.mean(regret, axis=1))

#generating the graph
# plt.figure(figsize=(10, 6))
# plt.plot(cumulative_regret, label="Cumulative Regret")
# plt.xlabel("Round")
# plt.ylabel("Cumulative Regret")
# plt.title("UCB Stochastic Cumulative Regret")
# plt.legend()
# plt.grid(True)
# plt.show()