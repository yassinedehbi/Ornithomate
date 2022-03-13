import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

predictions = pd.read_csv('/Users/akbya/Downloads/Ornithomate/data.txt', sep=",", header=None)
predictions.columns = ["path", "xmin", "ymin", "xmax", "ymax", "actual_label"]

d = {'SITTOR': 0,'PINARB': 0,'TOUTUR': 0,'ROUGOR': 0,'MESCHA': 0,'MOIDOM': 0,'MESNON': 0,'VEREUR': 0,'ACCMOU': 0,'MESBLE': 0,
    'ECUROU': 0, 'PIEBAV': 0, 'MULGRI': 0, 'CAMPAG': 0, 'MESNOI': 0, 'MESHUP': 0, 'BERGRI': 0}

labels = ['SITTOR','PINARB','TOUTUR','ROUGOR','MESCHA','MOIDOM','MESNON','VEREUR','ACCMOU','MESBLE', 'ECUROU', 'PIEBAV', 'MULGRI',
        'CAMPAG', 'MESNOI', 'MESHUP', 'BERGRI']

data = []
L=[0, 0, 0, 0, 0, 0, 0]

for index, row in predictions.iterrows():
    d[labels[row["actual_label"]]] += 1
    if data.count(row['path'])>=1:
        L[data.count(row['path'])]+=1
        L[data.count(row['path']) -1]-=1
    else:
        L[data.count(row['path'])]+=1
    data.append(row['path'])


keys = d.keys()
values = d.values()
print(L)
#plt.bar(keys, values)
#plt.show()

plt.bar([str(i) for i in range(1, len(L)+1)], L)
plt.show()