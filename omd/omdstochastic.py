import math
import matplotlib.pyplot as plt
import random
import numpy as np
        
class stochastic_OMD_Environment:
    def __init__(self, learning_rate, number_of_arms):
        self.learning_rate = learning_rate
        self.normalization_factor = 200*math.sqrt(10)
        self.estimated_loss_vector = np.zeros(number_of_arms)
        self.number_of_arms = number_of_arms
        # self.weights = np.random.uniform(0, 1, number_of_arms)
        # alpha = np.random.randint(1, number_of_arms+1, number_of_arms)
        # self.weights = np.random.dirichlet(alpha, size = 1).squeeze(0)

        # self.best_arm = random.randint(0, number_of_arms - 1)
    
    def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate):
        weights_for_arms = np.zeros(number_of_arms)
        # weights_for_arms = self.weights
        epsilon = 1.0e-9
        previous_normalization_factor = normalization_factor
        updated_normalization_factor = normalization_factor
        while True:
            for arm in range(number_of_arms):
                inner_product = abs((learning_rate * (estimated_loss_vector[arm] - updated_normalization_factor)))
                exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2)
                weight_of_arm = 4 * exponent_of_inner_product
                weights_for_arms[arm] = weight_of_arm
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
            if(difference_in_normalization_factors < epsilon):
                break
            else:
                continue
        return weights_for_arms, updated_normalization_factor

    def selectArm(self):
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        action_chosen = np.random.choice(number_of_arms, p = weights_of_arms)
        return action_chosen
    
    # def theseAreOurWeights(self):
    #     weights_for_our_arms = self.weights
    #     return weights_for_our_arms
    
    # def update_best_arm(self):
    #     return self.best_arm
    
    def updateLossVector(self, chosen_arm, loss):
        weights_of_arms, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        if weights_of_arms[chosen_arm] > 0:
            new_loss_estimate = loss / weights_of_arms[chosen_arm]
        else:
            new_loss_estimate = 0
        self.estimated_loss_vector[chosen_arm] += new_loss_estimate

# number_of_arms = 10
# time_horizon = 100000
# learning_rate = 0.005
#this one is shwya 3bee6 mdri leh

# stochasticOMDEnvironment = stochastic_OMD_Environment(learning_rate, number_of_arms)
# # alpha = np.random.randint(1, number_of_arms+1, number_of_arms)
# # arm_means = np.random.dirichlet(alpha, size = 1).squeeze(0) #stolen from github, idk if this is better than just using a uniform distribution for the means
# arm_means = np.random.uniform(0, 1, number_of_arms)
# losses_for_arm_in_each_round = []
# optimal_arm = np.argmax(arm_means)
# mean_of_optimal_arm = 1 - arm_means[optimal_arm]

#need regret to be similar to UCB

# def find_regret(time_horizon):
#     cumulative_optimal_mean = 0
#     cumulative_actual_mean = 0
#     regrets = []
#     for round_played in range(time_horizon):
#         chosen_arm = stochasticOMDEnvironment.selectArm()
#         loss_for_arm = 1 - np.random.binomial(1, arm_means[chosen_arm])
#         mean_of_chosen_arm = arm_means[chosen_arm]
#         losses_for_arm_in_each_round.append(loss_for_arm)
#         stochasticOMDEnvironment.updateLossVector(chosen_arm, losses_for_arm_in_each_round[round_played])
#         # optimal_arm = np.argmax(arm_means)
#         # mean_of_optimal_arm = arm_means[optimal_arm]
#         # loss_for_optimal_arm = 1 - np.random.binomial(1, arm_means[optimal_arm])
#         # cumulative_actual_mean += loss_for_arm
#         # cumulative_optimal_mean += loss_for_optimal_arm
#         cumulative_actual_mean += mean_of_chosen_arm
#         cumulative_optimal_mean += mean_of_optimal_arm
#         #regret_for_this_round = cumulative_actual_mean - cumulative_optimal_mean
#         regret_for_this_round = cumulative_optimal_mean - cumulative_actual_mean
#         # regret_for_this_round = mean_of_optimal_arm - mean_of_chosen_arm
#         regrets.append(regret_for_this_round)
#     return regrets

number_of_arms = 10
time_horizon = 100000
learning_rate = 0.005
num_iterations = 1

stochasticOMDEnvironment = stochastic_OMD_Environment(learning_rate, number_of_arms)
arm_means = np.random.uniform(0, 1, number_of_arms)
losses_for_arm_in_each_round = []
optimal_arm = np.argmax(arm_means)
mean_of_optimal_arm = 1 - arm_means[optimal_arm]

def find_regret(time_horizon, num_iterations):
    regrets = np.zeros([time_horizon, num_iterations])
    for iter in range(num_iterations):
        for round_played in range(time_horizon):
            chosen_arm = stochasticOMDEnvironment.selectArm()
            loss_for_arm = 1 - np.random.binomial(1, arm_means[chosen_arm])
            mean_of_chosen_arm = 1 - arm_means[chosen_arm]
            losses_for_arm_in_each_round.append(loss_for_arm)
            stochasticOMDEnvironment.updateLossVector(chosen_arm, losses_for_arm_in_each_round[round_played])
            regrets[round_played, iter] = mean_of_chosen_arm - mean_of_optimal_arm
    return regrets

regrets_to_plot = find_regret(time_horizon, num_iterations)
cumulative_regret = np.cumsum(np.mean(regrets_to_plot, axis=1))

plt.plot(cumulative_regret, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Stochastic Cumulative Regret")
plt.legend()
plt.grid()
plt.show()

"""
need to be able to keep track of every rounds regrets so we can keep track of the average regrets
(i dont know how helpful this is):
1) niba shy zy kida : [[0,1,2,2,0,1], [2,3,4,2,2,3], [4,5,6,6,3,4,5]]
each inner array is the regrets for each round like for round 1 we'll always start with a smaller regret so all the regrets 
we get in round one should be small b3den for round 2 binzeed 
2) so esh alfayda mn hada -> niba inno we flatten each inner array by just taking the mean value which thankfully fe numpy function that does this for us
bs almoshkila ana 3'abiya w mni 3arfa kef i store each round's regrets in its own inner array which i feel like is mrra a 110 skill issue
"""

# stochasticOMDEnvironment = stochastic_OMD_Environment(learning_rate, number_of_arms)
# # best_arm = stochasticOMDEnvironment.update_best_arm()
# # arm_means = np.random.uniform(0, 1, number_of_arms)
# alpha = np.random.randint(1, number_of_arms+1, number_of_arms)
# arm_means = np.random.dirichlet(alpha, size = 1).squeeze(0) #stolen from github, idk if this is better than just using a uniform distribution for the means
# # arm_means = stochasticOMDEnvironment.theseAreOurWeights()
# losses_for_arm_in_each_round = []
# # regret_for_trial = []
# # cumulative_actual_mean = 0
# # cumulative_optimal_mean = 0
# # simulations = 1
# # regrets = []

# regrets_for_specific_round = [0.0 for arm_played in range(simulations)]
# total_regrets_for_all_trials = [0.0 for list in range(time_horizon)]
# # for trial in range(simulations):
# for round_played in range(time_horizon):
#     regrets_for_our_simulation = find_regret(time_horizon)
#     regret_of_round = regrets_for_our_simulation[round_played]
#     # for round_played in range(time_horizon):
#     for trial in range(simulations):
#         # regret_for_each_round = []
#         # regret_of_round = regrets_for_our_simulation[round_played]
#         # regrets_for_specific_round.append(regret_of_round)
#         regrets_for_specific_round[round_played] = regrets_for_our_simulation[round_played]
#         # regret_for_each_round.append(regret_of_round)
#     # total_regrets_for_all_trials.append(regrets_for_specific_round)
#     total_regrets_for_all_trials[trial] = regrets_for_specific_round
# print("this is the regret of round", round_played, ":", total_regrets_for_all_trials)
        
# losses_for_all_rounds = np.zeros((time_horizon, simulations))
# regrets_using_trials = []
# total_regret_for_each_round = []
# # regret_for_each_round = []
# for trial in range(simulations):
#     regrets_for_our_simulation = find_regret(time_horizon)
#     # regrets_using_trials.append(regrets_for_our_simulation)
#     # regret_for_each_round = []
#     for round in range(time_horizon):
#         regret_for_each_round = []
#         regret_of_round = regrets_for_our_simulation[round]
#         regret_for_each_round.append(regret_of_round)
#         total_regret_for_each_round.append(regrets_for_our_simulation[round])
        # total_regret_for_each_round.append(regret_for_each_round) 
        # total_regret_for_each_round[] =
        #we dont want to append because that's just
        #adding stuff so 7g trial 1 gets added to trial 0 -> instead of appending we should initialize the array to be an array of arrays
        #then change the element values at the specific index which in this case is round w b3den that should solve our issue of 
        #accessing the individual arrays later on kaman to find the average
        # print("this is the regret of round", round, ":", regret_for_each_round)
# print(regrets_using_trials)
    # print("this is the regret of trial", trial, ":", total_regret_for_each_round)
    
# regrets_for_our_simulation = find_regret(time_horizon)

# plt.plot(regrets_for_our_simulation, label='Cumulative Regret')
# plt.xlabel('Round')
# plt.ylabel('Cumulative Regret')
# plt.title("OMD Stochastic Cumulative Regret")
# plt.legend()
# plt.grid()
# plt.show()
    

# for trial in range(simulations):
#     for round_played in range(time_horizon):
#         chosen_arm = stochasticOMDEnvironment.selectArm()
#         loss_for_arm = 1 - np.random.binomial(1, arm_means[chosen_arm]) #np.random gives us 0 or 1 reward based on probability of success
#         losses_for_arm_in_each_round.append(loss_for_arm)
#         stochasticOMDEnvironment.updateLossVector(chosen_arm, losses_for_arm_in_each_round[round_played])
#         optimal_arm = np.argmax(arm_means)
#         loss_for_optimal_arm = 1 - np.random.binomial(1, arm_means[optimal_arm])
#         cumulative_actual_mean += loss_for_arm
#         cumulative_optimal_mean += loss_for_optimal_arm
#         regret_for_this_round = cumulative_actual_mean - cumulative_optimal_mean
#         # regret_for_this_round = arm_means[optimal_arm] - arm_means[chosen_arm]
#         regret_for_trial.append(regret_for_this_round)
#         # print(regret_for_trial)
#     regrets.append(regret_for_trial)
# print(regrets)

    # regret_for_each_round = regret_for_trial[round_played]
    # average_regret_for_each_round = np.mean(regret_for_trial[round_played])
    # regrets.append(average_regret_for_each_round)
    
#     print("this is the loss for our arm", cumulative_actual_mean)
#     print("this is the optimal loss for our arm", cumulative_optimal_mean)
# print("these are the regrets", regrets)
    # stochasticOMDEnvironment.updateLossVector(chosen_arm, losses_for_arm_in_each_round)
    
    # print(optimal_arm)
# losses_for_all_rounds = np.zeros((time_horizon, number_of_arms))

# for round_played in range(time_horizon):
#     losses_for_each_round = np.zeros(number_of_arms)
#     for arm in range(number_of_arms):
#         loss_of_arm = 0
#         # probability = random.random()
#         loss_of_arm = 1 - np.random.binomial(1, arm_means[arm])
#         # if arm == best_arm:
#         #     if probability < 0.8:
#         #         loss_of_arm += 0
#         #     else:
#         #         loss_of_arm += 1
#         # else:
#         #     if probability < 0.2:
#         #         loss_of_arm += 0
#         #     else:
#         #         loss_of_arm += 1
#         losses_for_each_round[arm] = loss_of_arm
#     losses_for_all_rounds[round_played] = losses_for_each_round
# print(np.random.binomial(1, arm_means[arm]))
# optimal_arm = np.argmin(arm_means)

# best_arms_in_each_round = []
# for loss_vector in range(len(losses_for_all_rounds)):
#     best_arm_in_this_round = np.argmin(losses_for_all_rounds[loss_vector])
#     best_arms_in_each_round.append(best_arm_in_this_round)
# frequency_of_each_best_arm = np.bincount(best_arms_in_each_round)
# optimal_arm = np.argmax(frequency_of_each_best_arm)

# cumulative_optimal_mean = 0
# cumulative_actual_mean = 0
# regrets = []

# for round_loss_vector in range(len(losses_for_all_rounds)):
#     this_rounds_loss_vector = losses_for_all_rounds[round_loss_vector]
#     chosen_arm = stochasticOMDEnvironment.selectArm()
#     stochasticOMDEnvironment.updateLossVector(chosen_arm, this_rounds_loss_vector)
#     optimal_loss = this_rounds_loss_vector[optimal_arm]
#     actual_loss = this_rounds_loss_vector[chosen_arm]
#     cumulative_optimal_mean += optimal_loss
#     cumulative_actual_mean += actual_loss
#     regret_for_this_round = cumulative_actual_mean - cumulative_optimal_mean
#     regrets.append(regret_for_this_round)

# plt.plot(regrets, label='Cumulative Regret')
# plt.xlabel('Round')
# plt.ylabel('Cumulative Regret')
# plt.title("OMD Stochastic Cumulative Regret")
# plt.legend()
# plt.grid()
# plt.show()