from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import folium
import numpy as np
import random
import pickle
from math import radians, cos, sin, asin
import math
import scipy

def GetDist(pos1, pos2):
    dist = math.sqrt(math.pow(pos1[0] - pos2[0], 2) + math.pow(pos1[1] - pos2[1], 2))
    return dist

def CalAdjMat(pos,woro):
    A = np.zeros(shape = (len(pos), len(pos)))
    for i in range(len(pos)):
        for j in range(i, len(pos)):
            dist = math.sqrt(math.pow(pos[i, 0] - pos[j, 0], 2) + math.pow(pos[i, 1] - pos[j, 1], 2))
            wdiff = scipy.linalg.norm(woro[i] - woro[j],2)
            if dist != 0:
                A[i, j] = 1.0 / (dist + wdiff)
                A[j, i] = 1.0 / (dist + wdiff)
    return A

def norm_adjacency_matrix(A, n):
    I = np.eye(A.shape[0])
    A_hat = A + n * I
    D_inv = np.diag(np.float_power(np.sum(A_hat, axis = 0), -0.5))
    A_hat = D_inv @ A_hat @ D_inv
    return A_hat

df = pd.read_csv("Sample_WestCoast.txt", sep=',')
t0 = df.iloc[0:1071]
nonan_df = t0.dropna(axis=0, how='any')
raw = nonan_df.to_numpy()

raw = raw[:,[0,1,5,2,3,4,6]] # move woro to the 3rd column
raw = np.concatenate((raw, raw[:,0:4]), axis = 1) #concatenate lon,lat, woro at the end
#split the dataset into woro+ and woro-
woro_neg = raw[raw[:,2]<0].copy()
woro_pos = raw[raw[:,2 ]>0].copy()

coord_pos = woro_pos[:,0:2].copy()
coord_neg = woro_neg[:,0:2].copy()

# find the largest distance
largest = 0.0
for i in range(len(raw)):
    for j in range(i, len(raw)):
        dist = math.sqrt(
            (raw[i, 0] - raw[j, 0]) * (raw[i, 0] - raw[j, 0]) + (raw[i, 1] - raw[j, 1]) * (raw[i, 1] - raw[j, 1]))
        if dist > largest:
            largest = dist

#standardize
raw_pos = raw[raw[:,2]>0]
u_pos = np.mean(raw_pos, axis = 0)
sigma_pos = np.zeros(len(raw_pos[0]))
for i in range(len(raw_pos[0])):
    sigma_pos[i] = np.std(raw_pos[:,i])

raw_neg = raw[raw[:,2]<0]
u_neg = np.mean(raw_neg, axis = 0)
sigma_neg = np.zeros(len(raw_neg[0]))
for i in range(len(raw_neg[0])):
    sigma_neg[i] = np.std(raw_neg[:,i])

shape = woro_pos.shape
for j in range(shape[0]):
    for k in range(2, shape[1]-1):
        woro_pos[j,k] = (woro_pos[j,k] - u_pos[k]) / sigma_pos[k]
shape = woro_neg.shape
for j in range(shape[0]):
    for k in range(2, shape[1]-1):
        woro_neg[j,k] = (woro_neg[j,k] - u_neg[k]) / sigma_neg[k]

heatmap = folium.Map(location=[34, -103], zoom_start=4)
color = ["red", "blue"]
for j in range(len(coord_pos)):
    folium.Circle(
        location=[coord_pos[j, 0], coord_pos[j, 1]],
        radius=2,
        color=color[0],
        fill=True
    ).add_to(heatmap)
for j in range(len(coord_neg)):
    folium.Circle(
        location=[coord_neg[j, 0], coord_neg[j, 1]],
        radius=2,
        color=color[1],
        fill=True
    ).add_to(heatmap)
heatmap.save('Woro.html')

sil_scores = []
groups_pos = {}
for regions in range(3,15):
    estimator = KMeans(n_clusters = regions, random_state = 1)
    loc = coord_pos
    estimator.fit(loc)
    labels = estimator.labels_
    score = silhouette_score(loc, labels, metric='euclidean')
    sil_scores.append(score)
    # print(score)
    if regions == 10:
        heatmap = folium.Map(location=[34, -103], zoom_start=4)
        color = ["red", "green", "blue", "black", "orange", "yellow", "pink", "purple", "violet", "brown", "gray",
                 "cyan"]
        for i in range(regions):
            group = woro_pos[labels == i]
            for j in range(len(group)):
                folium.Circle(
                    location=[group[j, 0], group[j, 1]],
                    radius=2,
                    color=color[i % 12],
                    fill=True
                ).add_to(heatmap)
            np.save(r".\data_pos\pos_regions%d_group%d.npy" % (regions, i), group)
            groups_pos[i] = group
        heatmap.save('Kmeans_pos_%d.html'%regions)
np.savetxt("data_pos\\sil_scores_pos.txt", sil_scores)

sil_scores = []
groups_neg = {}
for regions in range(3,15):
    estimator = KMeans(n_clusters = regions, random_state = 1)
    loc = coord_neg
    estimator.fit(loc)
    labels = estimator.labels_
    score = silhouette_score(loc, labels, metric='euclidean')
    sil_scores.append(score)
    print(score)
    if regions == 8:
        heatmap = folium.Map(location=[34, -103], zoom_start=4)
        color = ["red", "green", "blue", "black", "orange", "yellow", "pink", "purple", "violet", "brown", "gray",
                 "cyan"]
        for i in range(regions):
            group = woro_neg[labels == i]
            for j in range(len(group)):
                folium.Circle(
                    location=[group[j, 0], group[j, 1]],
                    radius=2,
                    color=color[i % 12],
                    fill=True
                ).add_to(heatmap)
            #np.save(r".\data\neg_regions%d_group%d.npy" % (regions, i), group)
            groups_neg[i] = group
        heatmap.save('Kmeans_neg_%d.html'%regions)
#np.savetxt("data\\sil_scores_neg.txt", sil_scores)

# split training and testing index
cnt = 0
# aggregate woro+ and woro- clusters
groups = {}
#for key in groups_neg:
#    groups[cnt] = groups_neg[key]
#    cnt += 1
for key in groups_pos:
    groups[cnt] = groups_pos[key]
    cnt += 1

train_idx = {}
test_idx = {}
for key in groups:
    num = len(groups[key])
    total = np.arange(num)
    trainid = total.copy()
    if num >= 30:  ## make sure enough stations for partition
        testid = np.random.choice(num, round(num / 3), replace=False)
        test_idx[key] = testid
        trainid = np.delete(total, testid)
        test_idx[key] = testid


    train_idx[key] = trainid
# save index
f = open("./data_pos/test_idx.pkl", "wb")
pickle.dump(test_idx, f)
f.close()
f = open("./data_pos/train_idx.pkl", "wb")
pickle.dump(train_idx, f)
f.close()

# build training set
ratios = [0.5]
radius_ratios = [0.5, 0.25, 0.1]
for rad in radius_ratios:
    radius = rad * largest
    for r in range(len(ratios)):
        train_X = {}
        train_A = {}
        train_idx_ext = {}
        train_label = []
        for key in train_idx: ##pick training data index
            train_len = round(len(train_idx[key]) * ratios[r])
            tmp = np.random.choice(len(train_idx[key]), train_len, replace = False)
            train_each_group_idx = train_idx[key][tmp]
            train_idx_ext[key] = train_each_group_idx
            for i in range(len(train_each_group_idx)): ## pick training target
                idx = train_each_group_idx[i]
                target = groups[key][idx].copy()
                train_label.append(target[-1])
                target[3] = 0.
                for group_i in groups:  ## pick the input from every cluster
                    tmp = groups[group_i].copy()
                    if group_i == key: ## set target 0 when target is in the same cluster
                        tmp[idx, 2:7] = 0.0
                    eachgroup = np.concatenate(([target], tmp), axis=0) # put the target at the head
                    loc = eachgroup[:,7:9]
                    woro = eachgroup[:,9:-1]
                    A = norm_adjacency_matrix(CalAdjMat(loc, woro), 1)
                    if group_i == key:  ## set the line and row in adjacency matrix 0 when target is in the same cluster
                        A[idx + 1,:] = 0.
                        A[:, idx + 1] = 0.
                    for row in range(len(eachgroup)): ## check distance for every row in the cluster
                        if GetDist(eachgroup[row,0:2], target[0:2]) > radius: ## if the data point is out of range, set it 0
                            eachgroup[row, 2:7] = 0.0
                            A[row,:] = 0.
                            A[:,row] = 0.
                    ## put A and X into the training set
                    if group_i not in train_A:
                        train_A[group_i] = [A]
                    else:
                        train_A[group_i] = np.concatenate((train_A[group_i], [A]), axis=0)
                    if group_i not in train_X:
                        train_X[group_i] = [eachgroup]
                    else:
                        train_X[group_i] = np.concatenate((train_X[group_i], [eachgroup]), axis = 0)
        f = open("./data_pos/dist_%.2f/ext_%.2f/1/train_X.pkl"%(rad, ratios[r]), "wb")
        pickle.dump(train_X, f)
        f.close()
        f = open("./data_pos/dist_%.2f/ext_%.2f/1/train_idx.pkl"%(rad, ratios[r]), "wb")
        pickle.dump(train_idx_ext, f)
        f.close()
        f = open("./data_pos/dist_%.2f/ext_%.2f/1/train_A.pkl" %(rad, ratios[r]), "wb")
        pickle.dump(train_A, f)
        f.close()
        np.save("./data_pos/dist_%.2f/ext_%.2f/train_y.npy" %(rad, ratios[r]), train_label)

        ## aggregate to GCN data
        train_GCN_X = []
        train_GCN_A = []
        train_GCN_label = []
        for key in train_idx_ext:  ##pick training data index
            train_len = len(train_idx_ext[key])
            for i in range(train_len):  ## pick training target
                idx = train_idx_ext[key][i]
                target = groups[key][idx].copy()
                eachX = np.empty(shape=(0, 11))
                train_GCN_label.append(target[-1])
                target[3] = 0.
                for groups_k in groups:
                    eachX = np.concatenate((eachX, groups[groups_k]), axis=0)
                eachX = np.concatenate(([target], eachX), axis=0)
                eachX[idx + 1, 2:7] = 0.0
                loc = eachX[:, 7:9]
                woro = eachX[:, 9:-1]
                A = norm_adjacency_matrix(CalAdjMat(loc, woro), 1)
                A[idx + 1, :] = 0.
                A[:, idx + 1] = 0.
                for row in range(len(eachX)):
                    if GetDist(eachX[row, 0:2], target[0:2]) > radius:
                        eachX[row, 2:7] = 0.
                        A[row, :] = 0.
                        A[:, row] = 0.
                ## put A and X into the training set
                train_GCN_A.append(A)
                train_GCN_X.append(eachX)
        np.save("./data_pos/gcn/dist_%.2f/ext_%.2f/1/train_GCN_X.npy" % (rad, ratios[r]), train_GCN_X)
        np.save("./data_pos/gcn/dist_%.2f/ext_%.2f/1/train_GCN_A.npy" % (rad, ratios[r]), train_GCN_A)
        np.save("./data_pos/gcn/dist_%.2f/ext_%.2f/train_GCN_y.npy" % (rad, ratios[r]), train_GCN_label)

    ## test
    ## MGCN test
    test_X = {}
    test_A = {}
    test_label = []
    for key in test_idx:
        for i in range(len(test_idx[key])):
            idx = test_idx[key][i]
            target = groups[key][idx].copy()
            target[3] = 0.
            test_label.append(target[-1])
            for group_i in groups:
                tmp = groups[group_i].copy()
                if group_i == key:
                    tmp[idx, 2:7] = 0.0
                eachgroup = np.concatenate(([target], tmp), axis=0)
                loc = eachgroup[:, 7:9]
                woro = eachgroup[:, 9:-1]
                A = norm_adjacency_matrix(CalAdjMat(loc, woro), 1)
                if group_i == key:
                    A[idx + 1, :] = 0.
                    A[:, idx + 1] = 0.
                for row in range(len(eachgroup)):  ## check distance for every row in the cluster
                    if GetDist(eachgroup[row, 0:2],
                               target[0:2]) > radius:  ## if the data point is out of range, set it 0
                        eachgroup[row, 2:7] = 0.0
                        A[row, :] = 0.
                        A[:, row] = 0.
                if group_i not in test_A:
                    test_A[group_i] = [A]
                else:
                    test_A[group_i] = np.concatenate((test_A[group_i], [A]), axis=0)
                if group_i not in test_X:
                    test_X[group_i] = [eachgroup]
                else:
                    test_X[group_i] = np.concatenate((test_X[group_i], [eachgroup]), axis=0)
    f = open("./data_pos/dist_%.2f/1/test_X.pkl"%(rad), "wb")
    pickle.dump(test_X, f)
    f.close()
    f = open("./data_pos/dist_%.2f/1/test_A.pkl"%(rad), "wb")
    pickle.dump(test_A, f)
    f.close()
    np.save("./data_pos/dist_%.2f/1/test_y.npy"%(rad), test_label)

    ## GCN test
    test_GCN_X = []
    test_GCN_A = []
    test_GCN_label = []
    for key in test_idx:
        for i in range(len(test_idx[key])):
            idx = test_idx[key][i]
            target = groups[key][idx].copy()
            target[3] = 0.
            test_GCN_label.append(target[-1])
            eachX = np.empty(shape=(0, 11))
            for group_i in groups:
                eachX = np.concatenate((eachX, groups[group_i]), axis = 0)
            eachX = np.concatenate(([target], eachX), axis = 0)
            loc = eachX[:, 7:9]
            woro = eachX[:, 9:-1]
            A = norm_adjacency_matrix(CalAdjMat(loc, woro), 1)
            A[idx,:] = 0.
            A[:,idx] = 0.
            for row in range(len(eachX)):
                if GetDist(eachX[row,:2], target[:2]) > radius:
                    eachX[row,2:7] = 0.
                    A[row,:] = 0.
                    A[:,row] = 0.
            test_GCN_A.append(A)
            test_GCN_X.append(eachX)
    np.save("./data_pos/gcn/dist_%.2f/1/test_GCN_X.npy" % (rad), test_GCN_X)
    np.save("./data_pos/gcn/dist_%.2f/1/test_GCN_A.npy" % (rad), test_GCN_A)
    np.save("./data_pos/gcn/dist_%.2f/1/test_GCN_y.npy" % (rad), test_GCN_label)
