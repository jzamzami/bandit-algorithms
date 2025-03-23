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
        """something here is returning arms out of index but like this is the same function i used in exp3 and it worked fine there so im confused as to why its
        a problem here smhh"""

class Adversarial_OMD_Environment: #adversarial omd class
    def __init__(self, learning_rate, number_of_arms): 
        """
        class constructor
        
        arguments:
        1) self = used to represent an instance (object) of a class (stolen from geeks4geeks)
        2) learning_rate = controls the explore-exploit rate when we get to newtons method
        
        variables initialized:
        1) self.learning_rate = user input
        2) self.normalization_factor = for now is temporarily 10 (i tried other values between 0 and 15, and 10 gave me the best 
        results for some reason)
        3) self.weights = empty list -> weights are initialized to be 1
        4) self.estimated_loss_vector = empty list -> losses are initialized to be 0
        5) self.agents_history = empty list -> history of each round is initialized to be 0
        6) self.best_arm = index of best arm that is randomly chosen from our list of arms
        
        note: regularizer was also an argument and it was just user input, but it didn't really do anything and im not fully sure
        if it's supposed to, because in the OMD algorithm definition the only time it's used is when we wanted to update the weights 
        of the arms but we do that using newton's method and it's not used in newton's method so i think thats why it wasnt doing anything
        
        updated note: idky i thought regularizer was like user input and not literally the method that we're using to find the updated weights so
        like so much makes sense now for why it "did nothing" (slow moment)
        """
        self.learning_rate = learning_rate
        # self.regularizer = regularizer -> realizing this now makes absolutely no sense 
        self.normalization_factor = 10
        self.weights = [1.0 for arm in range(number_of_arms)]
        self.estimated_loss_vector = [0.0 for arm in range(number_of_arms)]
        self.number_of_arms = number_of_arms
        # self.agents_history = [] -> no point in keeping track of the agents history since in this setting the losses are arbitrarily chosen
        self.best_arm = random.randint(0, number_of_arms - 1)
        
    # def initialize_arm_weights(self, number_of_arms):
    #     """this was so unbelievably slow when running the debugger"""
    #     for arm in range(number_of_arms):
    #         self.weights.append(1.0)
    #     return self.weights
            
    # def initialize_loss_vector(self, number_of_arms):
    #     """same thing"""
    #     for arm in range(number_of_arms):
    #         self.estimated_loss_vector.append(0.0)
    #     return self.estimated_loss_vector
    
    # def initialize_agents_history(self, number_of_arms):
    #     """no point in this existing"""
    #     for arm in range(number_of_arms):
    #         self.agents_history.append(0.0)
    #     return self.agents_history
    
    # def finding_probability_distributions(self): 
    #     n_arms = len(self.weights)
    #     total_weight = sum(self.weights)
    #     probs = []
    #     for arm in range(n_arms):
    #         update_rule_for_arm = (1 - self.learning_rate) * (self.weights[arm] / total_weight) + (self.learning_rate / n_arms)
    #         probs.append(update_rule_for_arm)
    #     return probs
    """probability distrubution finding function used in exp3"""
    
    def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate):
        """
        method for finding weights of arms (unfortunately so many bugs with this one)
        
        arguments:
        1) self = python core!
        2) normalization_factor = self.normalization_factor is also just 10, want the normalization factor
        to control how our weights are being calculated and also want to find the optimal normalization factor (until convergence, which
        is until the difference in normalization factors is less than epsilon)
        3) estimated_loss_vector = self.estimated loss vector (want this to be updated after our update method but also important
        since we want to know the loss of a specific arm at a given point so we can also use it for finding our arm weights)
        4) learning_rate = self.learning rate and is just whatever we decide it is (doesn't change and is constant)
        
        returns:
        1) weights_for_arms = want to return a list that contains the weights of each arm so we can then use the weights to sample an action
        2) updated_normalization_factor = want to update our normalization factor with time and return the updated/optimal one
        """
        # weights_for_arms = self.weights -> something still up with this and like i know why it doesnt work now
        """so problem here is that when it comes to like chosen arm im getting arms that are out of bounds like one time the chosen arm would be like
        19 or 17 which shouldn't happen so need to fix this issue from occuring because like how are we choosing arms that don't even exist???"""
        # weights_for_arms = [1.0 for arm in range(number_of_arms)], this causes the same index out of bounds error
        """another bug is that weights_for_arms is initialized as an empty list every time newton's method function is called
        instead and whats happening is that we break out of the for loop super early because the difference in normalization factors
        converges quick (which at this point is probably also smth im also doing wrong), so what i mean to say is that by the end of like
        one or two iterations our weights_for_arms only has like 1 or 2 elements instead of having number of elements == number of arms, but 
        when i set weights_for_arms to be equal to self.weights i get an index out of bounds error, associated with line 104:
        self.history[chosen_arm] += 1 -> IndexError: index 16 is out of bounds for axis 0 with size 10, i kind of have a vague idea of what to fix
        because i dont think that self.history should ever reach an index 16 in the first place?? but yeah this is another thing that needs fixing
        (im so sorry again)"""
        weights_for_arms = [] #i also know this is wrong but its just for the code to run :(
        epsilon = 0.000001
        sum_of_weights = 0
        for arm in range(len(estimated_loss_vector)):
            inner_product = (learning_rate * (estimated_loss_vector[arm] - normalization_factor))
            exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2)
            weight_of_arm = 4 * exponent_of_inner_product
            weights_for_arms.append(weight_of_arm)
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
    """weights stop getting added when we reach "optimal" normaliztion factor but im not even sure if we're fr finding the optimal normalization
    factor, i think with this part im misundertsanding the logic of the algorithm which is causing the bug and the length of the weights list 
    not being equivalent to the number of the arms -> also like i realized that i was making a pretty dumb mistake with initializing the weights inside
    the method instead of using the self.weights thing or even just initializing it to be equal to the number of arms but fixing that causes an index error
    when it gets to the get loss method so there's definitely smth i need to fix there, definetely double check the logic of how the normalization factors
    are being calculated w like see if i fr know whats supposed to be going on (sorry this is an unserious remark)"""
    
    """in previous code/function i literally wasn't returning the updated normalization factor lol
    but also even when i fixed it, the normalization factor had no effect on the graph. Here the normalization
    factor does affect the graph, the regret seems to be less when the normalization factor is initially a value 
    like 10 instead of when it starts off as non-zero"""

    def selectArm(self):
        """
        selectArm method for selecting an arm based on our new arm weights
        
        arguments:
        1) self = python core!
        
        returns:
        1) action_chosen = index of arm chosen
        """
        self.weights, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        probabilites_of_arms = self.weights
        action_chosen = drawArm(probabilites_of_arms)
        return action_chosen
    
    def getLoss(self, chosen_arm):
        """
        get Loss method for getting the loss of the action we chose to take 
        
        arguments:
        1) self = python core!
        2) chosen_arm = index of arm we chose to play
        
        returns:
        1) nothing but should return the loss we got from that action so that the loss can be used to update
        our loss estimates
        
        note: something wrong agents_history and the way it is getting updated history (Ht−1 =(A1, X1, . . . , At−1, Xt−1)), 
        could be something wrong with the way the history is being stored
        """
        # self.agents_history[chosen_arm] += 1 #problem line smhh
        """no need to keep track of agents history (at least for now because it doesn't really impact how rewards
        /losses are generated since they're randomly generated so we don't really care what the agent chose in the past,
        this fixes one issue but there's still another one lol)"""
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
    """smth here needs to get fixed so i can git rid of the index out of bounds error and the weights list
    in newtons method function whatever can have the right number of elements"""
    
    #how does github work? 
    
    def ihate(self):
        pass
    
    def updateWeights(self, chosen_arm, loss):
        """
        update method for updating arm weights
        
        arguments:
        1) self = python core!
        2) chosen_arm = index of arm we played in round
        3) loss = loss observed from that action
        
        returns: nothing hehe
        """
        self.weights, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        probabilites_of_arms = self.weights
        if probabilites_of_arms[chosen_arm] > 0:
            """chosen arm is like index 19 which shouldnt be possible so something wrong with how we're getting 
            our chosen arm because we need to make sure that it's within the range of our actual arms because if its not then that makes like literally
            no sense"""
            new_loss_estimate = loss / probabilites_of_arms[chosen_arm]
            # self.estimated_loss_vector[chosen_arm] += new_loss_estimate
            """old problem(no longer an issue): other problem here is that adding the new loss estimate causes a ValueError: math domain error which has
            to do with line 30: exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2), this line
            i think is already problematic since it's like giving weirdly huge numbers which is also giving us huge
            weights which is smth that needs to be fixed, what's interesting or confusing i guess is that this error only 
            occurs when the normalization factor is 10 (when its 0 it just takes a long time to run but the graph it produces
            is unbelievably weird so its not like thats the solution, it probably would cause an error with other non-zero
            normalization factors which is not ideal), but i think this should be the correct way of updating the loss estimates
            unless im really missing smth
            
            new problem(is an issue right now): this results in either completely linear regret (with super high cumulative regrets
            like 40k) or the weirdest graph ever (its like super spikey and all over the place but super tiny cumulative regret like 150) so
            it could be either something with the way we're finding the loss estimates, the losses, or if the condition for the if statement is
            even correct -> weird graph is in figures folder under 'regret graph when adding loss estimates' lol"""
            self.estimated_loss_vector[chosen_arm] += loss #this line works but i know is logically incorrect and is only here so the code can run normally :(
        else:
            new_loss_estimate = 0
            self.estimated_loss_vector[chosen_arm] += new_loss_estimate

learning_rate = 0.01
number_of_arms = 10
T = 100000
# regularizer = 10 (should not have included this)
simulations = 30

for simulation in range(simulations): #i think this is how to do like many simulations
    omd_adversarial = Adversarial_OMD_Environment(learning_rate, number_of_arms)
    """got rid of separate adversarial class and just combined it with the omd class thinking it
    would magically get rid of the index out of bounds error lol"""
    # omd_adversarial.initialize_arm_weights(number_of_arms)
    # omd_adversarial.initialize_loss_vector(number_of_arms)
    # omd_adversarial.initialize_agents_history(number_of_arms)
    regrets = []
    cumulative_loss = 0

    for t in range(T):
        chosen_arm = omd_adversarial.selectArm()
        loss = omd_adversarial.getLoss(chosen_arm)
        cumulative_loss += loss
        omd_adversarial.updateWeights(chosen_arm, loss) #new problem line lol (i forgot what problem this is causing)
        optimal_loss = (t + 1) * 0.3
        regrets.append(cumulative_loss - optimal_loss)

plt.plot(regrets, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Class Adversarial Environment Cumulative Regret Over Time")
plt.legend()
plt.grid()
plt.show()