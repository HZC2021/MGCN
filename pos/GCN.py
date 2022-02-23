from spektral.models import GCN
"""
This example implements the experiments on citation networks from the paper:
Semi-Supervised Classification with Graph Convolutional Networks (https://arxiv.org/abs/1609.02907)
Thomas N. Kipf, Max Welling
"""
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
import random



def scheduler(epoch, lr):
  if epoch < 150:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def train(TrainX, train_y, TestX, test_y, gcnnodes = 4, features = 5, fcnnodes = 64, dropout = 0., lr = 1e-3, name = "", l1 = 2.5e-4, epochs = 150):
    model = GCN(fcnnodes = fcnnodes, channels = gcnnodes, n_input_channels=features, dropout_rate= dropout, l1_reg=l1)

    model.compile(
        optimizer=Adam(lr),
        loss='mean_squared_error'
    )
    checkpoint_path = "GCN_record_pos_dist%.2f_2/training_mgcn_%0.2f_%s_%s_hd%d_fc%d/cp-{epoch:02d}.ckpt" \
                      % (rad, r, name, dropout, gcnnodes, fcnnodes)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(TrainX,
                              np.reshape(train_y, (-1,1)),
                              epochs=epochs,
                              batch_size=32,  #origin batch-size = 32
                              validation_data=(TestX, np.reshape(test_y, (-1,1))),
                              shuffle=False,
                              callbacks=[cp_callback, lr_callback])


    return model, history

if __name__ == "__main__":

    r = 0.50
    rad = 0.10

    train_GCN_X = np.load('./data_pos/gcn/dist_%.2f/ext_0.50/1/train_GCN_X.npy' % rad)
    train_GCN_A = np.load('./data_pos/gcn/dist_%.2f/ext_0.50/1/train_GCN_A.npy' % rad)
    test_GCN_X = np.load('./data_pos/gcn/dist_%.2f/1/test_GCN_X.npy' % rad)
    test_GCN_A = np.load('./data_pos/gcn/dist_%.2f/1/test_GCN_A.npy' % rad)
    train_y = np.load("./data_pos/gcn/dist_%.2f/ext_0.50/train_GCN_y.npy" % rad)
    test_y = np.load("./data_pos/gcn/dist_%.2f/1/test_GCN_y.npy" % rad)

    num = len(train_GCN_X)
    total = list(np.arange(num))
    trainlen = round(num * 0.9)
    trainid = list(np.random.choice(total, trainlen, replace=False))
    for i in trainid:
        total.remove(i)
    np.save("data_pos/dist_%.2f/ext_0.50/validx_gcn.npy" % rad, np.array(total))

    TrainX = []
    ValX = []
    tmp = train_GCN_X[trainid,:,:]
    TrainX.append(tmp[:, :, [2,3,4,5, 6]])
    TrainX.append(train_GCN_A[trainid,:,:])
    tmp2 = train_GCN_X[total, :, :]
    ValX.append(tmp2[:, :, [2,3,4,5, 6]])
    ValX.append(train_GCN_A[total, :, :])
    TrainY = train_y[trainid]
    ValY = train_y[total]

    TestX = []
    TestX.append(test_GCN_X[:, :, [2,3,4,5, 6]])
    # TestX.append(test_X_dict[key][:, :, [4]])
    TestX.append(test_GCN_A)

    epochs = 150
    l1_arr = [5e-8, 5e-7, 5e-6]
    l1_arr_name = [ "5e-8", "5e-7", "5e-6"]
    rate_arr = [0.0]
    fcnodes = [96]
    gcnnodes = [12]
    for i in range(len(l1_arr)):
        print(i)
        for drop in rate_arr:
            for fnodes in fcnodes:
                for gnodes in gcnnodes:
                    _,his = train(TrainX, TrainY, ValX, ValY, gcnnodes=gnodes, features=5, fcnnodes=fnodes,
                          dropout=drop, lr=1e-3, l1 = l1_arr[i], name = l1_arr_name[i],epochs=epochs)
                    val_loss=[np.argmin(his.history["val_loss"])+1, np.min(his.history["val_loss"])]
                    np.save("val_epoch_%d_%d_%d.npy"%(i,fnodes,gnodes),np.array(val_loss))