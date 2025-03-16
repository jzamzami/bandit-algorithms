import numpy as np
import math
import matplotlib.pyplot as plt
import random

def drawArm(probabilities_of_choosing_arms): #helper function for choosing arm based off arm weights
    choice = random.uniform(0, sum(probabilities_of_choosing_arms))
    choiceIndex = 0
    for probability_of_arm in probabilities_of_choosing_arms:
        choice -= probability_of_arm
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1 

class Adversarial_OMD_Environment:
    def __init__(self, learning_rate, regularizer):
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.weights = []
        self.normalization_factor = 10
        self.estimated_loss_vector = []
        self.agents_history = []
        self.best_arm = random.randint(0, number_of_arms - 1)
        
    def initialize_arm_weights(self, number_of_arms):
        for arm in range(number_of_arms):
            self.weights.append(1.0)
        return self.weights
            
    def initialize_loss_vector(self, number_of_arms):
        for arm in range(number_of_arms):
            self.estimated_loss_vector.append(0.0)
        return self.estimated_loss_vector
    
    def initialize_agents_history(self, number_of_arms):
        for arm in range(number_of_arms):
            self.agents_history.append(0.0)
        return self.agents_history
        
    def newtons_approximation_for_arm_weights(self, normalization_factor, estimated_loss_vector, learning_rate):
        # weights_for_arms = self.weights
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

    def select_arm(self): 
        self.weights, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        probabilites_of_arms = self.weights
        action_chosen = drawArm(probabilites_of_arms)
        return action_chosen
    
    def update(self, chosen_arm, loss):
        self.weights, self.normalization_factor = self.newtons_approximation_for_arm_weights(self.normalization_factor, self.estimated_loss_vector, self.learning_rate)
        if self.weights[chosen_arm] > 0:
            new_loss_estimate = loss / self.weights[chosen_arm]
            # self.estimated_loss_vector[chosen_arm] += new_loss_estimate
            """other problem here is that adding the new loss estimate causes a ValueError: math domain error which has
            to do with line 30: exponent_of_inner_product = math.pow(((inner_product + epsilon)), -2), this line
            i think is already problematic since it's like giving weirdly huge numbers which is also giving us huge
            weights which is smth that needs to be fixed, what's interesting or confusing i guess is that this error only 
            occurs when the normalization factor is 10 (when its 0 it just takes a long time to run but the graph it produces
            is unbelievably weird so its not like thats the solution, it probably would cause an error with other non-zero
            normalization factors which is not ideal), but i think this should be the correct way of updating the loss estimates
            unless im really missing smth"""
            self.estimated_loss_vector[chosen_arm] += loss #this line works but i know is logically incorrect and is only here so the code can run :(
        else:
            new_loss_estimate = 0
            self.estimated_loss_vector[chosen_arm] += new_loss_estimate
            
    def getLoss(self, chosen_arm):
        """smth here needs to get fixed so i can git rid of the index out of bounds error and the weights list 
        in newtons method function whatever can have the right number of elements"""
        self.agents_history[chosen_arm] += 1 #problem line smhh
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

learning_rate = 0.01
number_of_arms = 10
T = 100000
regularizer = 10
simulations = 30

for simulation in range(simulations):
    omd_adversarial = Adversarial_OMD_Environment(learning_rate, regularizer)
    """got rid of separate adversarial class and just combined it with the omd class thinking it
    would magically get rid of the index out of bounds error lol"""
    omd_adversarial.initialize_arm_weights(number_of_arms)
    omd_adversarial.initialize_loss_vector(number_of_arms)
    omd_adversarial.initialize_agents_history(number_of_arms)
    regrets = []
    cumulative_loss = 0

    for t in range(T):
        chosen_arm = omd_adversarial.select_arm()
        loss = omd_adversarial.getLoss(chosen_arm)
        cumulative_loss += loss
        omd_adversarial.update(chosen_arm, loss)
        optimal_loss = (t + 1) * 0.3
        regrets.append(cumulative_loss - optimal_loss)

plt.plot(regrets, label='Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.title("OMD Class Adversarial Environment Cumulative Regret Over Time")
plt.legend()
plt.grid()
plt.show()