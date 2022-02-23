import numpy as np
import pykrige.kriging_tools as kt
from pykrige.uk import UniversalKriging
import math
from error_calculation import MSE, MAD, ave_err, hist_err
from sklearn.metrics import mean_squared_error, r2_score
import pickle


test_y = np.load(r".\data_neg\gcn\dist_0.10\1\test_GCN_X.npy")
label = np.load(r".\data_neg\gcn\dist_0.10\1\test_GCN_y.npy")

testy_hat = []
test_err = []
test_gt = []
trainy_hat = []
train_err = []
train_gt = []

for i in range(len(test_y)):
    sample = test_y[i,:,:].copy()
    del_rows = []
    for rowd in range(1, len(sample)): ##del empty rows
        if (sample[rowd,4] == 0.0) and (sample[rowd,5] == 0.0) and (sample[rowd,6] == 0.0):
            del_rows.append(rowd)
    newsample = np.delete(sample, del_rows, axis = 0)
    x = newsample[1:,0]
    y = newsample[1:,1]
    d = newsample[1:,4]
    lbl = newsample[1:,-1]
    drift = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], d[:, np.newaxis]), axis = 1)
    # if len(lbl) <= 6:
    #     continue
    if np.sum(lbl) == 0.0: ## if all training lbl = 0, output 0
        test_gt.append(newsample[0, -1])
        testy_hat.append(0)
        for row in range(1,len(newsample)):
            trainy_hat.append(0)
            train_gt.append(newsample[row,-1])
    else:
        UK = UniversalKriging(
            x,
            y,
            lbl,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
            nlags = 6,
            point_drift= drift
        )

        tx = newsample[0,0]
        ty = newsample[0,1]
        minx = np.min(newsample[:, 0])
        miny = np.min(newsample[:, 1])
        maxx = np.max(newsample[:, 0])
        maxy = np.max(newsample[:, 1])
        gridx = np.arange(minx - 0.1, maxx + 0.1, 0.05)
        gridy = np.arange(miny - 0.1, maxy + 0.1, 0.05)
        z, ss = UK.execute("grid", gridx, gridy)
        tmp = z[math.floor((ty - miny) // 0.05), math.floor((tx - minx) // 0.05)]
        err = tmp - newsample[0,-1]
        test_err.append(err)
        testy_hat.append(tmp)
        test_gt.append(newsample[0,-1])
        for row in range(1,len(newsample)):
            trainx = newsample[row,0:2]
            tmp = z[math.floor((trainx[1] - miny)// 0.05), math.floor((trainx[0] - minx) // 0.05)]
            err = tmp - newsample[row,-1]
            train_err.append(err)
            trainy_hat.append(tmp)
            train_gt.append(newsample[row, -1])
        # print("r2_score", r2_score(train_gt, trainy_hat))

np.save("trainy_hat.npy", trainy_hat)
np.save("testy_hat.npy", testy_hat)

print("train:\n")
print("mse", MSE(trainy_hat, train_gt))
print("mad", MAD(trainy_hat, train_gt))
print("ave", ave_err(trainy_hat, train_gt))
print("r2_score", r2_score(train_gt, trainy_hat))
print("test:\n")
print("mse", MSE(testy_hat, test_gt))
print("mad", MAD(testy_hat, test_gt))
print("ave", ave_err(testy_hat, test_gt))
print("r2_score", r2_score(test_gt, testy_hat))
