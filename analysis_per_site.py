import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

predictions = pd.read_csv('/Users/akbya/Downloads/Ornithomate/4gen_predicted.txt', sep=" ", header=None)
predictions.columns = ["path", "xmin", "ymin", "xmax", "ymax", "actual_label", "predicted_label", "confidence"]

sites = {'Francon': ["task_2021-03-01_09", "task_2021-03-01_10"],
        'UPS': ["task_20210526_UPS"], 'Lab': ["task_20210611_Lab", "task_20210612_1_Lab"],
        'balacet': ["task_20210705-07_balacet"], 'Orlu': ["task_20211204_Orlu"], 'Gajan': ["task_21-01-2021"]}

accuracy = {'Francon': 0, 'UPS': 0, 'Lab': 0, 'balacet': 0, 'Orlu': 0, 'Gajan': 0}
num_images = {'Francon': 0, 'UPS': 0, 'Lab': 0, 'balacet': 0, 'Orlu': 0, 'Gajan': 0}
for index, row in predictions.iterrows():
    task = os.path.basename(os.path.dirname(row["path"]))
    if task in sites['Francon']:
        accuracy["Francon"] += 1 if row["predicted_label"]==row["actual_label"] else 0
        num_images["Francon"] +=1
    elif task in sites['UPS']:
        accuracy["UPS"] += 1 if row["predicted_label"]==row["actual_label"] else 0
        num_images["UPS"] +=1
    elif task in sites['Lab']:
        accuracy["Lab"] += 1 if row["predicted_label"]==row["actual_label"] else 0
        num_images["Lab"] +=1
    elif task in sites['balacet']:
        accuracy["balacet"] += 1 if row["predicted_label"]==row["actual_label"] else 0
        num_images["balacet"] +=1
    elif task in sites['Orlu']:
        accuracy["Orlu"] += 1 if row["predicted_label"]==row["actual_label"] else 0
        num_images["Orlu"] +=1
    elif task in sites['Gajan']:
        accuracy["Gajan"] += 1 if row["predicted_label"]==row["actual_label"] else 0
        num_images["Gajan"] +=1

d = {x:float(accuracy[x])/num_images[x] for x in num_images}

keys = d.keys()
values = d.values()

plt.bar(keys, values)
plt.show()