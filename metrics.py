from re import I
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
from collections import Counter


classes = ['SITTOR','PINARB','TOUTUR','ROUGOR','MESCHA','MOIDOM','MESNON','VEREUR','ACCMOU','MESBLE', 'ECUROU', 'PIEBAV', 'MULGRI',
        'CAMPAG', 'MESNOI', 'MESHUP', 'BERGRI']

figure, axis = plt.subplots(3, 3)
col, lin = 0, 0
result = []
for c in classes:
    tests = pd.read_csv('/Users/akbya/Downloads/Ornithomate/test.txt', sep=",", header=None)
    tests.columns = ["path", "xmin", "ymin", "xmax", "ymax", "actual_label"]
    predictions = pd.read_csv('/Users/akbya/Downloads/Ornithomate/4gen_predicted.txt', sep=" ", header=None)
    predictions.columns = ["path", "xmin", "ymin", "xmax", "ymax", "actual_label", "predicted_label", "confidence"]
    # Get only detection of class c
    dects = []
    for index, row in predictions.iterrows():
        if row["predicted_label"] != 17 and classes[row["predicted_label"]] == c:
            dects.append([row["path"], row["predicted_label"], row["confidence"], [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]])
    # Get only ground truths of class c
    gts = []
    for index, row in tests.iterrows():
        if classes[row["actual_label"]] == c:
            gts.append([row["path"], row["actual_label"], 1, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]])
    npos = len(gts)
    # sort detections by decreasing confidence
    dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
    TP = np.zeros(len(dects))
    FP = np.zeros(len(dects))
    # create dictionary with amount of gts for each image
    det = Counter([cc[0] for cc in gts])
    for key, val in det.items():
        det[key] = np.zeros(val)
    # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
    # Loop through detections
    for d in range(len(dects)):
        # print('dect %s => %s' % (dects[d][0], dects[d][3],))
        # Find ground truth image
        gt = [gt for gt in gts if gt[0] == dects[d][0]]
        iouMax = sys.float_info.min
        for j in range(len(gt)):
            # print('Ground truth gt => %s' % (gt[j][3],))
            def iou(boxA, boxB):
                # if boxes dont intersect
                def boxesIntersect(boxA, boxB):
                    if boxA[0] > boxB[2]:
                        return False  # boxA is right of boxB
                    if boxB[0] > boxA[2]:
                        return False  # boxA is left of boxB
                    if boxA[3] < boxB[1]:
                        return False  # boxA is above boxB
                    if boxA[1] > boxB[3]:
                        return False  # boxA is below boxB
                    return True
                if boxesIntersect(boxA, boxB) is False:
                    return 0
                def getIntersectionArea(boxA, boxB):
                    xA = max(boxA[0], boxB[0])
                    yA = max(boxA[1], boxB[1])
                    xB = min(boxA[2], boxB[2])
                    yB = min(boxA[3], boxB[3])
                    # intersection area
                    return (xB - xA + 1) * (yB - yA + 1)
                interArea = getIntersectionArea(boxA, boxB)
                def getUnionAreas(boxA, boxB, interArea=None):
                    def getArea(box):
                        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
                    area_A = getArea(boxA)
                    area_B = getArea(boxB)
                    if interArea is None:
                        interArea = getIntersectionArea(boxA, boxB)
                    return float(area_A + area_B - interArea)
                union = getUnionAreas(boxA, boxB, interArea=interArea)
                # intersection over union
                iou = interArea / union
                assert iou >= 0
                return iou
            iou = iou(dects[d][3], gt[j][3])
            if iou > iouMax:
                iouMax = iou
                jmax = j
        # Assign detection as true positive/don't care/false positive
        if iouMax >= 0.45:
            if det[dects[d][0]][jmax] == 0:
                TP[d] = 1  # count as true positive
                # print("TP")
            det[dects[d][0]][jmax] = 1  # flag as already 'seen'
        # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
        else:
            FP[d] = 1  # count as false positive
            # print("FP")
    # compute precision, recall and average precision
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]
    # Depending on the method, call the right implementation
    [ap, mpre, mrec, ii] = CalculateAveragePrecision(rec, prec)
    
    r = {
        'class': c,
        #'precision': prec,
        #'recall': rec,
        'AP': ap,
        'total positives': npos,
        'total TP': np.sum(TP),
        'total FP': np.sum(FP)
    }
    if len(prec) != 0:
        axis[col, lin].plot(rec, prec, label=c)
        axis[col, lin].set_xlabel('recall')
        axis[col, lin].set_ylabel('precision')
        ap_str = "{0:.2f}%".format(ap * 100)
        axis[col, lin].legend(shadow=True)
        axis[col, lin].grid()
        if (lin<2):
            lin+=1
        elif (lin==2):
            lin=0
            col+=1
    result.append(r)
    print(r)
print("MAP : ", sum(res['AP'] for res in result)/sum(1 for res in result if res['total positives'] > 0))
plt.savefig('/Users/akbya/Downloads/Ornithomate/precision_recall.png')
plt.show()
