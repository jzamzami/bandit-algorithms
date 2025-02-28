import random

class BernoulliBandit:
    # accepts a list of K >= 2 floats , each lying in [0, 1]
    def __init__(self, means):
        self.means = means # means here helps us know the number the arms
        # means is a list of the reward means of each arm
        # (how likely youre gonna get a reward from a specifc arm)
        # between 0 to 1 because that's how probability works lol
        # because this is a bernoulli bandit you either get the reward or not
        # so knowing the success (getting reward) means is beneficial
        self.totalActionsTaken = [] #added list to track total actions taken in order to find
        #accumulated regret
        if (any(x < 0 or x > 1 for x in means)): #taken from tensorflow agents documentation
            raise ValueError('All parameters should be floats in [0, 1].')
        if (len(means) < 2):
            raise ValueError("list of means should consist of k >= 2 floats")
        # two if statements to ensure that our input meets stated conditions
    
    def __str__(self): #to string function needed so print statements are readable
        return f"{self.means}" #forgot f strings existed lol
    
    # Function should return the number of arms
    def K(self):
        return len(self.means) #each mean is for a distinct arm so lenght of list is how many arms
    # list arms = []
    # for learner l
        # for action taken a
            # arms.append(a)
    #return len(arms)
    # this nested for loop method does not work since the learner can choose repeated arms
    # actions != arms, even if we use a set to make sure the arms are distinct there's the
    # possibility the learner might not choose a specific arm so there's that

    # Accepts a parameter 0 <= a <= K -1 and returns the
    # realisation of random variable X with P(X = 1) being
    # the mean of the (a+1) th arm .
    def pull(self, a):
        numberOfArms = self.K()
        # totalActionsTaken = [], if we define it here we can't use it later on so it has to be
        #a global variable
        if (a < 0 or a > numberOfArms - 1): #making sure a input is a valid action
            raise ValueError("invalid action")
        else: #if input is valid
            self.totalActionsTaken.append(a) #if the action is valid add it to the list of actions
            #taken by the learner
            actualProbabilityOfRewardAfterPullingArm = random.random() #variable name
            #self-explanatory, but generates value from 0 to 1 (probability of actually getting
            # a reward is completely random in the context of the learner)
            if (actualProbabilityOfRewardAfterPullingArm <= self.means[a]):
                return 1
            #i orginally had flipped logic where probability bigger than means meant success
            #but this logic is flawed since we would be outside of the success range
            #for example 60% success means i can success upto 60% of the time, but after outside
            #that range is when ill fail (60% success and 40% failure anything after 0.6 is failure)
            else:
                return 0 #dont succeed you get nothing :(

    # Returns the regret incurred so far .
    def regret(self):
        accumulatedRegret = 0 #start with 0
        maxMeanOfArms = max(self.means) # arm with best mean -> optimal arm
        for a in self.totalActionsTaken: #iterate over each action taken by the learner
            meanOfArmActuallyPlayed = self.means[a] #find out what the mean of the arm played
            regret = maxMeanOfArms - meanOfArmActuallyPlayed #find the regret as the difference
            #between optimal mean and the mean of the arm the learner actually chose
            accumulatedRegret += regret
            #add that regret to our accumulated regret so we know the total regret after each
            #action taken by the learner
        return accumulatedRegret #return that value after going through all actions

#test case
# means_invalid = BernoulliBandit([0.8, 0.6, 5])
# print(means_invalid), prints correct error!

# means_too_short = BernoulliBandit([0.8])
# print(means_too_short), prints correct error!

means_nothing_wrong = BernoulliBandit([0.8, 0.6, 0.5, 0.9, 0.4])
print(means_nothing_wrong)

number_of_arms = means_nothing_wrong.K()
print(number_of_arms)

action_taken_by_learner = means_nothing_wrong.pull(1)
print(action_taken_by_learner)

regret_for_learner = means_nothing_wrong.regret()
print(regret_for_learner)

new_action_taken_by_learner = means_nothing_wrong.pull(2)
print(new_action_taken_by_learner)

new_regret_for_learner = means_nothing_wrong.regret()
print(new_regret_for_learner)