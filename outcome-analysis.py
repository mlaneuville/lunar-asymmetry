import pandas as pd
import json

with open('data.txt') as data_file:    
    data = json.load(data_file)

df = pd.DataFrame(data)
print(df.info())

import matplotlib.pyplot as plt

cm = plt.cm.get_cmap('viridis')
ax = plt.scatter(df['F1'], df['F0'], c=df['nearside_mean'], cmap=cm)
plt.colorbar(ax)
plt.show()
