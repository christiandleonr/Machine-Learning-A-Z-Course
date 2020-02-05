# Upper Confidence Bound

# Importing the libraries

import pandas as pd
import matplotlib.pyplot as plt
import math

# Importing the data-set

data = pd.read_csv('C:/Users/chris/PycharmProjects/MachineLearningA-ZCourse/Part 6 - Reinforcement Learning/'
                   'Section 27 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv')

# Implementing UCB
N = 10000
d = 10
ads_selected = []
numbers_of_selection = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selection[i] > 0:
            average_reward = sums_of_rewards[i]/numbers_of_selection[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 10e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selection[ad] += 1
    reward = data.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

# Visualising the results

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of time each ad was selected')
plt.show()