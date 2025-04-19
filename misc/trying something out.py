import math
import matplotlib.pyplot as plt
import random
import numpy as np

class Adversarial_Exp3:
    def __init__(self, learning_rate, n_arms):
        self.learning_rate = learning_rate
        self.n_arms = n_arms
        self.weights = np.ones(n_arms)
        self.best_arm = random.randint(0, n_arms - 1)

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
    
    def select_arm(self):
        probs = self.finding_probability_distributions()
        action_chosen = np.random.choice(n_arms, p=probs)
        return action_chosen

    def update(self, chosen_arm, loss):
        probs = self.finding_probability_distributions()
        if probs[chosen_arm] > 0:
            loss_estimate = loss / probs[chosen_arm]
        else:
            loss_estimate = 0
        growth_factor = math.exp((self.learning_rate / n_arms) * loss_estimate)
        self.weights[chosen_arm] *= growth_factor
        
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

random.seed(1)
n_arms = 10
n_rounds = 100000
learning_rate = 0.01

adversarialExp3Environment = Adversarial_Exp3(learning_rate, n_arms)
regret = []
cumulative_loss = 0

for t in range(n_rounds):
    chosen_arm = adversarialExp3Environment.select_arm()
    loss = adversarialExp3Environment.assign_loss(chosen_arm)
    adversarialExp3Environment.update(chosen_arm, loss)
    cumulative_loss += loss
    optimal_loss = (t + 1) * 0.7
    regret.append(optimal_loss - cumulative_loss)

plt.figure(figsize=(10, 6))
plt.plot(regret, label="Cumulative Regret")
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Exp3 Adversarial Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.show()