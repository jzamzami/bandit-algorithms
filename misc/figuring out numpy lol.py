import numpy as np

# print("this is array 1:", np.zeros(2))
# print("this is array 2:", np.zeros(3))
# print(np.zeros([2,3]))
# print(np.zeros([3,2]))

time_horizon = 5
num_iterations = 1
regret = np.zeros([time_horizon, num_iterations])
print(regret)