#first attempt (kind of misunderstood that this had to be a greedy approach but its different)
def FollowTheLeader(bandit, n): #function declaration hehe
    for t in range(n): #for round in range of time horizon
        listOfArms = [] #empty list but intended to be list of arms
        listOfArmmean = [] #should actuall be empty list lol, stores meanOfEachArm of each arm
        for arm in listOfArms: #iterate through arms (don't know how to initially get this to work
            #given empty list)
            bandit.pull(arm) #pull each arm in the list at least once
            listOfArms.append(arm) #add to our list of arms
            arm += 1 #move on to next arm
            for mean in listOfArmmean: #now looking at meanOfEachArm (same issue with starting 
                #with empty list so initially won't work)
                mean = bandit.pull(arm)/count(arm) #attempt to find each arm's mean
                #don't know how to count the number of times an arm was used -> should be 1 maybe
                #since we're using greedy strategy but not sure
                listOfArmmean.append(mean) #add mean to list of meanOfEachArm so we can use for
                #comparison
    if mean[arm] > mean[arm - 1]: #if the mean of the current arm is higher than the 
        #previous then pull this arm
        bandit.pull(arm)
    else: #else pull the previous arm -> this process should terminate once looked at all arms
        #so this comparison should be done in one of the for loops probably?
        bandit.pull(arm - 1)

"""
general comments:
1) this isnt a traditional greedy problem where we update the best arms as we go, in this problem
we're exploring ALL of our options then finding the one with the best mean to exploit (we're
giving each arm one chance to find the best one)
-> misunderstanding the problem statement led to a wrong approach
2) assuming this function is implemented in a bandits class (like the bernoulli one for example)
then we can assume that we have some function that counts the number of arms -> this is highly
important as we cant just iterate through an empty list lol
3) using the number of arms we'd then have to keep track of how many times each arm is used, the
sum of rewards of each arm, and the mean for each arm -> initialize empty lists depending on the
number of arms available
4) we'll start by iterating through the number of arms and pulling each one while updating the 
three quantities? mentioned above
5) once we're done we'll find the arm with the best mean and ONLY pull that one, but since we can
have ties we're told that we have to randonly break the tie to figure out which one to use
6) this is done by initializing an empty list to store the arms with the best mean then we can
iterate through the averages found to see if they equal the max if yes then we can store them in 
the list -> use python random.choice function that chooses random element from list
7) then just choose that arm and update any values as necessary
"""

#2nd attempt
import random
def FollowTheLeader(bandit, n): #function declaration hehe
    listOfArms = bandit.find_num_arms #assuming we have a function that lets us find the number of arms
    #(like in our bernoulli class)
    frequencyOfEachArm = [0] * listOfArms #initialize how many times an arm has been pulled
    sumOfEachArm = [0.0] * listOfArms #initialize sum of rewards for each arm
    meanOfEachArm = [0.0] * listOfArms #initialize the meanOfEachArm for each arm
    
    for arm in listOfArms: #iterate through arms
        reward = bandit.pull(arm) #reward is whatever we got from pulling an arm
        frequencyOfEachArm[arm] += 1 #increment the count -> since we've now used an arm
        sumOfEachArm[arm] += reward #add reward to sum of rewards
        meanOfEachArm[arm] = sumOfEachArm[arm] / frequencyOfEachArm[arm] #find the mean reward of each arm
    
    for t in range(n): #now that we're done iterating through our list of arms, we can
        #do the actual greedy approach since we know each arm's mean
        max_mean = max(meanOfEachArm)
        #find the max of each avergae
        best_arms = [] #list of best arms -> need so we can break ties between them
        #(should have the same highest mean though)
        for arm in listOfArms: #finding if theres a tie between the arms
            if meanOfEachArm[arm] == max_mean: #comparing mean of chosen arm to the best mean
                best_arms.append(arm) #add arm to list of best arms
        chosen_arm = random.choice(best_arms) #randomly chose between arms with the best mean
        reward = bandit.pull(chosen_arm) #we'll continue to just choose this arm
        #exploit-only strategy (explored once then exploited best arm)