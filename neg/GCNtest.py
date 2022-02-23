
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from CusGCN import GCN
from spektral.transforms import LayerPreprocess

import os
import pickle
from sklearn.metrics import r2_score,mean_squared_error
from error_calculation import MAD,ave_err


def predict(TestX, test_y, gcnnodes = 4, features = 5, fcnnodes = 64, dropout = 0., lr = 1e-3, name = "", l1 = 0.):
    model = GCN(fcnnodes = fcnnodes, channels = gcnnodes, n_input_channels=features, dropout_rate= 0., l1_reg=0.)

    model.compile(
        optimizer=Adam(lr),
        loss='mean_squared_error'
    )
    checkpoint_path = "GCN_record_neg_dist%.2f_2/training_mgcn_%0.2f_%s_%s_hd%d_fc%d/cp-150.ckpt" \
                      % (rad, r, name, dropout, gcnnodes, fcnnodes)
    #val_loss = np.load("val_epoch_%d_%d_%d.npy" % (0, fcnnodes, gcnnodes))
    # checkpoint_path = "GCN_record_neg_dist%.2f_2/training_mgcn_%0.2f_%s_%s_hd%d_fc%d/cp-%02d.ckpt"% (rad, r, name, dropout, gcnnodes, fcnnodes, val_loss[0])
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model.load_weights(checkpoint_path)

    y_pred = model.predict(TestX)
    R2 = r2_score(test_y,y_pred)
    MSE = mean_squared_error(test_y,y_pred)
    return R2, MSE, y_pred

if __name__ == "__main__":
    r = 0.50
    rad = 0.10
    
    train_GCN_X = np.load('./data_neg/gcn/dist_%.2f/ext_0.50/1/train_GCN_X.npy' % rad)
    train_GCN_A = np.load('./data_neg/gcn/dist_%.2f/ext_0.50/1/train_GCN_A.npy' % rad)
    test_GCN_X = np.load('./data_neg/gcn/dist_%.2f/1/test_GCN_X.npy' % rad)
    test_GCN_A = np.load('./data_neg/gcn/dist_%.2f/1/test_GCN_A.npy' % rad)
    train_y = np.load("./data_neg/gcn/dist_%.2f/ext_0.50/train_GCN_y.npy" % rad)
    test_y = np.load("./data_neg/gcn/dist_%.2f/1/test_GCN_y.npy" % rad)
    TrainX = []
    TrainX.append(train_GCN_X[:, :, [2,3,4,5, 6]])
    TrainX.append(train_GCN_A)
    TestX = []
    TestX.append(test_GCN_X[:, :, [2,3,4,5, 6]])
    TestX.append(test_GCN_A)

    l1_arr = 5e-8
    l1_arr_name = "5e-8"
    fnodes = 96
    gnodes = 12
    r2, mse, y_hat = predict(TestX, test_y, gcnnodes=gnodes, features=5, fcnnodes=fnodes, dropout=0.0,
          lr=1e-3, name = l1_arr_name, l1=l1_arr)
    print("mse:", mse)
    print("test R2:", r2)
    print("MAD:", MAD(y_hat, test_y))
    print("ave:", ave_err(y_hat, test_y))
    r2, mse, y_hat = predict(TrainX, train_y, gcnnodes=gnodes, features=5, fcnnodes=fnodes, dropout=0.0,
          lr=1e-3, name = l1_arr_name, l1=l1_arr)
    print("train R2:", r2)

