#https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

#region k means
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
import matplotlib.pyplot as plt
#endregion

#region hierarchical clustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt

from sklearn.cluster import AgglomerativeClustering
#endregion

import numpy as np

class agglomerativeclustering(object):
    def __init__(self, X, cv2, frame):
        self.X = X
        self.cv2 = cv2
        self.frame = frame

    def calculate(self):

        if len(self.X) == 0:
            print ("No entries")
            return

        K = None        
        centers = []
        print ( self.X )
        K = range(0,len(self.X))
        i = 0
        for k in K:
            centers.append([self.X[i][0], self.X[i][1]])
            i=i+1

        #https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
        points = np.array(centers)

        # create blobs
        #data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)
        # create np array for data points
        #points = data[0]

        linked = linkage(self.X, 'ward')
        labelList = range(0, len(points))

        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance (Ward)')

        dendogram = sch.dendrogram(linked,
            orientation='top', 
            labels=labelList, 
            distance_sort='descending', 
            show_leaf_counts=True)        
        
        # create dendrogram
        dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))

        # create clusters
        #hc = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'ward')
        # save clusters for chart
        #y_hc = hc.fit_predict(points)
        #print (y_hc) 
        
        #text = "Cluster"
        #self.cv2.putText(self.frame, text, (int(c[0]) - 10, int(c[1]) - 10),
        #    self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #self.cv2.circle(self.frame, (int(c[0]), int(c[1])), 14, (0, 255, 0), -1)
