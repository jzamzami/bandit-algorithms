import math
import matplotlib.pyplot as plt
import random
import numpy as np
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

# def categorical_draw(probs):
#     """ helper function for pulling an arm after finding the probability distribution(the
#     weights for all arms)
#     arguments: 1) probabilities_of_choosing_arms = list of each arm's probability of being pulled
#     after running the EXP3 algorithm
#     returns the arm (index of arm) we're gonna pull
#     """
#     choice = random.uniform(0, sum((probs)))
#     choiceIndex = 0
#     rounded_sum = round(sum(probs))
#     for probability_of_arm in (probs):
#         if probability_of_arm < 0 or probability_of_arm > 1 or rounded_sum != 1:
#             raise ValueError("This is not a valid probability distribution (you can't pull arm 1 with probability 400)!!")
#         else:
#             choice -= probability_of_arm
#             if choice <= 0:
#                 return choiceIndex
#             choiceIndex += 1


class Adversarial_Exp3: #class for our exp3 algortihm -> class lets us have constructors so makes 
    #the environment easy to think about/create
    def __init__(self, learning_rate, n_arms): #initializing our learning which affects how much we 
        #explore/exploit and also array/vector for our reward estimators 
        self.learning_rate = learning_rate
        self.n_arms = n_arms
        #self.weights = [1.0] * n_arms
        self.weights = np.ones(n_arms)
        self.best_arm = random.randint(0, n_arms - 1)

    # def initialize(self, n_arms):
    #     # self.weights = [1.0 for _ in range(n_arms)] #initializes the weights for the arms to be 1
    #     for arm in range(n_arms):
    #         self.weights.append(1.0)

    def finding_probability_distributions(self):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = np.zeros(n_arms)
        for arm in range(self.n_arms):
            first_term = (1 - self.learning_rate) * (self.weights[arm] / total_weight)
            second_term = (self.learning_rate / n_arms)
            update_rule_for_arm = first_term + second_term
            probs[arm] = update_rule_for_arm
        return probs
    
    def select_arm(self): #function for selecting arms
        # n_arms = len(self.weights) #the total number of arms should be the same as the length
        # #of our weights array/vector
        # total_weight = sum(self.weights) #total weight is the sum of the array
        # # probs = [(1 - self.learning_rate) * (self.weights[arm] / total_weight) + 
        # #         (self.learning_rate / n_arms) for arm in range(n_arms)] #calculating the 
        # #probability of selective a certain arm 
        # probs = []
        # for arm in range(n_arms):
        #     update_rule_for_arm = (1 - self.learning_rate) * (self.weights[arm] / total_weight) + (self.learning_rate / n_arms)
        #     probs.append(update_rule_for_arm)
        probs = self.finding_probability_distributions()
        # action_chosen = categorical_draw(probs)
        action_chosen = np.random.choice(n_arms, p=probs)
        return action_chosen #based on this probability we sample an action
    
    def update_best_arm(self):
        probability = random.random()
        if probability <= 0.35:
            best_arm = random.randint(0, n_arms - 1)
        else:
            best_arm = self.best_arm
        return best_arm
        
    # def assign_reward(self, chosen_arm):
    #     best_arm = self.update_best_arm()
    #     probability = random.random()
    #     reward = 0
    #     if chosen_arm == best_arm:
    #         if probability < 0.7:
    #             reward += 1
    #         else:
    #             reward += 0
    #     else:
    #         if probability < 0.3:
    #             reward += 1
    #         else:
    #             reward += 0
    #     return reward
    
    # def assign_reward(self, chosen_arm):
    #     best_arm = self.update_best_arm()
    #     reward = 0
    #     if chosen_arm == best_arm:
    #         reward += 1
    #     else:
    #         reward += 0
    #     return reward
    
    def update_arm_reward_estimate(self, chosen_arm, reward_vector): #function for updating our array of reward estimators
        #here we taken in the arm the agent chose and the reward sampled which we need for our update
        # n_arms = len(self.weights) #once again the number of arms should be the same
        # total_weight = sum(self.weights) #also total weight calculation doesnt change
        # # probs = [(1 - self.learning_rate) * (self.weights[arm] / total_weight) + 
        # #         (self.learning_rate / n_arms) for arm in range(n_arms)] #same formula for 
        # #calculating probability again
        # probs = []
        # for arm in range(n_arms): #list comprehension getting obliterated again
        #     update_rule_for_arm = (1 - self.learning_rate) * (self.weights[arm] / total_weight) + (self.learning_rate / n_arms)
        #     probs.append(update_rule_for_arm)
        # x = reward / probs[chosen_arm] if probs[chosen_arm] > 0 else 0 #reward estimator
        probs = self.finding_probability_distributions()
        if probs[chosen_arm] > 0:
            reward_estimate = reward_vector[chosen_arm] / probs[chosen_arm]
        else: #dont return anything (like update rewardes) if arm isn't chosen
            reward_estimate = 0
        growth_factor = math.exp((self.learning_rate / n_arms) * reward_estimate) #growth factor
        self.weights[chosen_arm] *= growth_factor #updating based off of the growth factor

#adversarial environment same as the one defined in the UCB algorithm 

# n_arms = 10
# n_rounds = 100000
# learning_rate = 0.005

# # adversary = Adversarial_Exp3(n_arms)
# # exp3 = Adversarial_Exp3(learning_rate)

# adversarialExp3Environment = Adversarial_Exp3(learning_rate, n_arms)
# regret = []
# # cumulative_reward = 0
# cumulative_reward = 0

# for t in range(n_rounds):
#     chosen_arm = adversarialExp3Environment.select_arm()
#     best_arm = adversarialExp3Environment.update_best_arm()
#     reward = adversarialExp3Environment.assign_reward(chosen_arm)
#     adversarialExp3Environment.update(chosen_arm, reward)
    
#     cumulative_reward += reward
#     optimal_reward = (t + 1) * 0.7
#     """note: optimal_reward was defined this way because it was supposed to be like what if the agent
#     got reward of 1 after each round as that's the highest possible reward (then the times 0.7 is because
#     that was the probability of getting that reward (or actually it was like the best case scenario was getting
#     optimal arm then with probability 70% you'd get the good reward))"""
#     regret.append(optimal_reward - cumulative_reward)
#     """i know this says reward but its not using rewardes its still using rewards so need 
#     to change it to rewardes instead fr"""
    
# print(regret)
    
    # chosen_arm = adversarialExp3Environment.select_arm()
    # reward = adversarialExp3Environment.assign_reward(chosen_arm)
    # adversarialExp3Environment.update(chosen_arm, reward)
    
    # cumulative_reward += reward
    # optimal_reward = 1 - ((t + 1) * 0.7)
    # regret.append(cumulative_reward - optimal_reward)
    
# adversarialExp3Environment = Adversarial_Exp3(learning_rate, n_arms)
# regret = []
# # cumulative_reward = 0
# cumulative_reward = []

# for round_played in range(n_rounds):
#     chosen_arm = adversarialExp3Environment.select_arm()
#     best_arm = adversarialExp3Environment.update_best_arm()
#     reward = adversarialExp3Environment.assign_reward(chosen_arm)
#     adversarialExp3Environment.update(chosen_arm, reward)
#     cumulative_reward.append(reward)
# # print(cumulative_reward)
    
# optimal_reward = np.max(cumulative_reward)
# regrets = []
# for round_played in range(n_rounds):
#     regret = optimal_reward - cumulative_reward[round_played]
#     regrets.append(regret)
    
# print(regrets)

# print(cumulative_reward)
# optimal_reward = np.max(cumulative_reward)
# print(optimal_reward)
# regret = optimal_reward - reward_observed

# """ 
# optimal_reward = np.max(cumulative_reward)
# regrets = []
# cumulative_reward = 0 
# for round in range time_horizon:
#     cumulative_reward += reward 
#     regret = optimal_reward - reward
#     regrets.append(regret) 
# """

"""
need to keep track of two things: 1) the cumulative reward (like the reward we accumulate after each round)
2) our reward vector (need to keep track of the reward vector so we know which arm yields best reward)
"""

""" 
for regret calculation we are NOT the agent (we are an all knowing entity 3shan e7na n3rf kol shy) so basically 3shan e7na
n3rf kol shy fa n3rf ayat arm 7ayidina a7san reward so 3shan kida lazem we keep track of the rewards of each arm after every round
so y3ni ka2ano we're playing in a full feedback setting w mo bandit setting so alzibda:
    1) after every round yikon fe a reward vector for every arm's rewards (mdri lw we implement this using a dictionary yikoon a7san aw shy)
    2) w b3den after playing ALL the rounds (yikoon 5ala9 the time horizon) nishoof ayat arm was usually the best one most of the time
    3) w for the optimal reward it's basically na5od hada albest arm w we find esh alreward lw a5tarnaha every round (so the optimal policy
    would've just been consistently choosing hada albest arm for every round) so we take this reward
    4) w for each round's regret its just gonna be the optimal reward - actual observed reward (note: aham shy iftakri inno
    the observed reward for each round is like adding the rewards from previous rounds kaman bc they dont just kida disapear)
    5) w obvi append the regret for each round into an array so then you just plot that obvi

tyeb whats going to be shwya yl5b6:
    1) i can easily keep track of the cummulative reward bs how do i know which arm consistently gave me the best arm
    2) w how the hell do i go back in time to know what that arm's reward was in each round 
    -> mdri a7is this makes sense in theory bs how do i implement this
    
variables needed (all the code here is just pseudocode so i can think of how to find these values):
1) reward vector -> array of arrays that contains the rewards of each arm from every round:
    reward_vector = []
    reward_vector_for_each_round = []
    for round in range(time_horizon):
        for arm in range(number_of_arms):
            reward = get_reward
            reward_vector_for_each_round.append(reward)
        reward_vector.append(reward_vector_for_each_round)
    return reward_vector
    
2) optimal arm -> index of arm that consistently gives us best rewards (shloon i find optimal arm from this
reward vector):
best_arm_in_each_round = []
for round in range(len(reward_vector_for_each_round)):
    max_reward_in_this_round = max(reward_vector)
    best_arm_in_round = reward_vector_for_each_round.index(max_reward_in_this_round)
    best_arm_in_each_round.append(best_arm_in_round)
best_arms_overall = np.bincount(best_arm_in_each_round)
best_arm_overall = np.argmax(best_arms_overall) #best_arm_overall would then just be our optimal arm
return best_arm_overall

3) rewards of that optimal arm -> so now that we have our optimal arm lazem we go back in time to see what rewards that optimal arm
would have given us (and like actual regret calculation):
cumulative_best_reward = 0
cumulative_actual_reward = 0
regrets = []
for round in range(time_horizon):
    best_reward = reward_vector_for_each_round[best_arm_overall]
    actual_reward = reward_vector_for_each_round[arm_pulled]
    cumulative_best_reward += best_reward
    cumulative_actual_reward += actual_reward
    regret_for_this_round = cumulative_best_reward - cumulative_actual_reward
    regrets.append(regret_for_this_round)
return regrets
"""

n_arms = 10
time_horizon = 100000
learning_rate = 0.01

adversarialExp3Environment = Adversarial_Exp3(learning_rate, n_arms)

rewards_for_all_rounds = np.zeros((time_horizon, n_arms))
for round_played in range(time_horizon):
    rewards_for_each_round = np.zeros(n_arms)
    best_arm = adversarialExp3Environment.update_best_arm()
    for arm in range(n_arms):
        reward_of_arm = 0
        if arm == best_arm:
            reward_of_arm += 1
        else:
            reward_of_arm += 0
        rewards_for_each_round[arm] = reward_of_arm
    rewards_for_all_rounds[round_played] = rewards_for_each_round
    
best_arms_in_each_round = []
for reward_vector in range(len(rewards_for_all_rounds)):
    best_arm_in_this_round = np.argmax(rewards_for_all_rounds[reward_vector])
    best_arms_in_each_round.append(best_arm_in_this_round)
frequency_of_each_best_arm = np.bincount(best_arms_in_each_round)
optimal_arm = np.argmax(frequency_of_each_best_arm)

cumulative_optimal_reward = 0
cumulative_reward = 0
regrets = []

for round_reward_vector in range(len(rewards_for_all_rounds)):
    this_rounds_reward_vector = rewards_for_all_rounds[round_reward_vector]
    chosen_arm = adversarialExp3Environment.select_arm()
    adversarialExp3Environment.update_arm_reward_estimate(chosen_arm, this_rounds_reward_vector)
    optimal_reward = this_rounds_reward_vector[optimal_arm]
    actual_reward = this_rounds_reward_vector[chosen_arm]
    cumulative_optimal_reward += optimal_reward
    cumulative_reward += actual_reward
    regret_for_this_round = cumulative_optimal_reward - cumulative_reward
    regrets.append(regret_for_this_round)

# plt.figure(figsize=(10, 6))
# plt.plot(regret, label="Cumulative Regret")
# plt.xlabel("Round")
# plt.ylabel("Cumulative Regret")
# plt.title("Exp3 Adversarial Cumulative Regret Over Time")
# plt.legend()
# plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(regrets, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Exp3 Adversarial Cumulative Regret")
plt.legend()
plt.grid(True)
plt.show()