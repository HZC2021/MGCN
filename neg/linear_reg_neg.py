import numpy as np
from sklearn.cluster import KMeans
# import folium
import math
from sklearn.metrics import mean_squared_error, r2_score
from error_calculation import MSE, MAD, ave_err, hist_err

from sklearn.linear_model import LassoCV, Lasso
import numpy as np

rad = 0.10
r = 0.50
TrainX = np.load('./data_neg/gcn/dist_%.2f/ext_%.2f/1/train_GCN_X.npy'%(rad, r))
TestX = np.load('./data_neg/gcn/dist_%.2f/1/test_GCN_X.npy'%rad)
TrainY = np.load("./data_neg/gcn/dist_%.2f/ext_%.2f/train_GCN_y.npy" % (rad, r))
TestY = np.load("./data_neg/gcn/dist_%.2f/1/test_GCN_y.npy" % (rad))


trainx = TrainX[:,:,0:7]
shape = trainx.shape
trainx = trainx.reshape((shape[0], shape[1]*shape[2]))
shape = len(TrainY)
TrainY.resize((shape,))

testx = TestX[:,:,0:7]
shape = testx.shape
testx = testx.reshape((shape[0], shape[1]*shape[2]))
shape = len(TestY)
TestY.resize((shape,))

model = LassoCV()
model.fit(trainx, TrainY)

pred = model.predict(testx)

pred_train = model.predict(trainx)

## print error
print(r2_score(TestY, pred))
print("mse:", MSE(pred, TestY))
print("mad:", MAD(pred, TestY))
print("ave:", ave_err(pred, TestY))
print("r2_score:", r2_score(TrainY, pred_train))
print("hist_err:")












