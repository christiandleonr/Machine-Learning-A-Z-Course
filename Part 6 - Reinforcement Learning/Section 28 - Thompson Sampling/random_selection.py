# Random Selection

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the data-set

data = pd.read_csv('C:/Users/chris/PycharmProjects/MachineLearningA-ZCourse/Part 6 - Reinforcement Learning/'
                   'Section 27 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv')

# Implementing Random Selection

N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = data.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()