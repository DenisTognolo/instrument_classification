from instrument_classificator import *

conf_mat = np.asarray([[5, 1, 0],[2, 5, 1],[0, 1, 7]])

n, m = conf_mat.shape

accuracy = np.zeros(n)
precision = np.zeros(n)
recall = np.zeros(n)
fscore = np.zeros(n)

for i in range(n):

    TP = conf_mat[i,i]
    FP = sum(conf_mat[:,i]) - TP
    FN = sum(conf_mat[i,:]) - TP
    TN = sum(sum(conf_mat)) - TP -  FP - FN

    accuracy[i] = (TN+TP) / (TN+FP+FN+TP)
    precision[i]  = TP / (TP+FP)
    recall[i] = TP / (TP+FN)
    fscore[i] = 2*(precision[i]*recall[i])/(precision[i]+recall[i])

