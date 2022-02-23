import numpy as np
import math
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def MSE(pred, label):
    assert len(pred) == len(label), "length not match"
    mse = 0
    for i in range(len(label)):
        mse += (pred[i] - label[i]) * (pred[i] - label[i])
    mse /= len(label)
    return mse

def MAD(pred, label):
    assert len(pred) == len(label), "length not match"
    err = 0
    for i in range(len(label)):
        err += abs(pred[i] - label[i])
    err /= len(label)
    return err

def R2(pred, label):
    return r2_score(label, pred)

def ave_err(pred, label):
    err = 0
    for i in range(len(label)):
        err += pred[i] - label[i]
    err /= len(label)
    return err

def hist_err(pred, label):
    #  matplotlib.axes.Axes.hist() 方法的接口
    err = pred - label
    lb = np.min(err)
    ub = np.max(err)
    plt.hist(x=err, bins=100, color='#0504aa'
                                )
    plt.xlabel('residuals')
    plt.ylabel('count')
    plt.xlim(lb, ub)
    plt.show()

if __name__ == "__main__":
    # pred = np.load("result/Kriging_pred.npy")
    # label = np.load("result/Kriging_test.npy")
    # print("mse", MSE(pred, label))
    # print("MAD", MAD(pred, label))
    # print("R2", R2(pred, label))
    # print("ave_err", ave_err(pred, label))
    # hist_err(pred, label)

    # pred = np.loadtxt("result_gcn_9/gcn_pred_5e-3_0.5.txt")
    # label = np.loadtxt("result_gcn_9/gcn_meas_5e-3_0.5.txt")
    pred = np.loadtxt("result\\gcn_testpred.txt")
    label = np.loadtxt("result\\gcn_testmeas.txt")
    print("mse", MSE(pred, label))
    print("MAD", MAD(pred, label))
    print("R2", R2(pred, label))
    print("ave_err", ave_err(pred, label))
    hist_err(pred, label)


