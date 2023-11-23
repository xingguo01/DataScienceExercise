import pandas as pd
import numpy as np # Only necessary for this example if you don't use it no poblem
import matplotlib.pyplot as plt
#Auto = pd.read_csv('College.csv')
#print(Auto)
college2 = pd.read_csv('College.csv')
print(college2)
college3 = college2.rename({'Unnamed: 0': 'College'},axis=1)
print(college3)
college3 = college3.set_index('College')
print(college3)
college=college3
print(college[['Enroll','Accept']].describe())
#fig, ax = plt(figsize=(8, 8))
fig=pd.plotting.scatter_matrix(college[['Top10perc','Apps','Enroll']])
plt.savefig(r"scatter_college.pdf")
fig=college.boxplot('Outstate',by='Private')
plt.savefig(r"boxplot_college.pdf")
college['Elite'] = pd.cut(college['Top10perc'],[0,0.5,1],labels=['No', 'Yes'])
print(college['Elite'].value_counts())
fig=college.boxplot('Outstate',by='Elite')
plt.savefig(r"boxplot_elite_college.pdf")

from matplotlib.pyplot import subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.hist(college['Top10perc'],color='r')
ax1.set_xlabel('Top10perc')
ax2.hist(college['Apps'],color='g')
ax2.set_xlabel('Apps')
ax3.hist(college['Accept'],color='b')
ax3.set_xlabel('Accept')
ax4.hist(college['Outstate'],color='m')
ax4.set_xlabel('Outstate')
plt.savefig(r"hist.pdf")
