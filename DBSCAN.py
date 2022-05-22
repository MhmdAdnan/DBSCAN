# -*- coding: utf-8 -*-

import numpy
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import math


#for making density data 
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)
#------------------------------------------------------------------
def MyDBSCAN(D, eps, MinPts):
    c = 0
    lables = []
    for x in range(len(D)):
        lables.append(0)
    for p in range(len(D)):

      if(lables[p]==0):
         spherepoint = regionQuery(D,p,eps)
         if len(spherepoint) < MinPts:
             lables[p]=-1
         else:
             c =c+1
             growCluster(D,lables,p,spherepoint,c,eps,MinPts)
    return lables


def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    labels[P] = C
    i = 0
    while i < len(NeighborPts):
        p_point = NeighborPts[i]
        if (labels[p_point] == 0):
            labels[p_point] = C
            SpherepointsP_POINT = regionQuery(D,p_point,eps)
            if(len(SpherepointsP_POINT)>=MinPts):
                NeighborPts = NeighborPts + SpherepointsP_POINT
        if(labels[p_point]==-1):
            labels[p_point]=C
        i+=1
    return NeighborPts

def regionQuery(D, P, eps):
    avalabile_points = []
    for point in range(len(D)):

        dist = math.sqrt(numpy.sum((D[P]-D[point])**2))
        if dist <= eps:
            avalabile_points.append(point)

    return avalabile_points

my_labels = MyDBSCAN(X, eps=0.3, MinPts=10)
#print(my_labels)

print("==========================================")
# built in DBSCAN Function
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
skl_labels = db.labels_
#print(skl_labels)

# Scikit learn uses -1 to for NOISE, and starts cluster labeling at 0. I start
# numbering at 1, so increment the skl cluster numbers by 1.
for i in range(0, len(skl_labels)):
    if not skl_labels[i] == -1:
        skl_labels[i] += 1


num_disagree = 0
#---------------------------------
#compare built in and custom made dbsan function
# Go through each label and make sure they match (print the labels if they 
# don't)
for i in range(0, len(skl_labels)):
    if not skl_labels[i] == my_labels[i]:
        print ('Scikit learn:', skl_labels[i], 'mine:', my_labels[i])
        num_disagree += 1

if num_disagree == 0:
    print ('PASS - All labels match!')
else:
    print ('FAIL -', num_disagree, 'labels don\'t match.')
