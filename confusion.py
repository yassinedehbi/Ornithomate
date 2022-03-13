import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix

test_predictions = pd.read_csv('/Users/akbya/Downloads/Ornithomate/4gen_predicted.txt', sep=" ", header=None)
test_predictions.columns = ["path", "xmin", "ymin", "xmax", "ymax", "actual_label", "predicted_label", "confidence"]
actual_label = []
predicted_label = []
for index, row in test_predictions.iterrows():
    actual_label.append(row["actual_label"])
    predicted_label.append(row["predicted_label"])
plt.scatter([i for i in range(17)], [sum(1 for x in actual_label if x==i) for i in range(17)])
plt.xticks(np.arange(17), ['SITTOR','PINARB','TOUTUR','ROUGOR','MESCHA','MOIDOM','MESNON','VEREUR','ACCMOU','MESBLE', 'ECUROU', 'PIEBAV', 'MULGRI',
        'CAMPAG', 'MESNOI', 'MESHUP', 'BERGRI'], rotation=90)
plt.show()
print([sum(1 for x in actual_label if x==i) for i in range(17)])
print([sum(1 for x in predicted_label if x==i) for i in range(18)])
print(sum(1 for x,y in zip(actual_label,predicted_label) if x == y) / len(actual_label))

cm = confusion_matrix(actual_label, predicted_label)
print(cm.shape)

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('/Users/akbya/Downloads/Ornithomate/confusion.png')
    plt.show()

plot_confusion_matrix(cm, ['SITTOR','PINARB','TOUTUR','ROUGOR','MESCHA','MOIDOM','MESNON','VEREUR','ACCMOU','MESBLE', 'ECUROU', 'PIEBAV', 'MULGRI',
        'CAMPAG', 'MESNOI'], normalize=False)