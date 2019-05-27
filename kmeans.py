#region k means
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from kneed import KneeLocator
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
#endregion

import numpy as np
import collections

class kmeans(object):
    def __init__(self, X, cv2, frame):
        self.X = X        
        self.cv2 = cv2
        self.frame = frame

    def calculate(self):

        K = None
        kn = None
        distortions = []
        centers = []

        #print (self.X)

        if len(self.X) == 0:
            print ("No entries")
            #text = "Center"
            #self.cv2.circle(self.frame, (int(self.X[0]), int(self.X[1])), 14, (0, 255, 0), -1)
            return

        if (len(self.X) < 4):
            K = range(1,len(self.X))
            i = 0
            for k in K:
                center = np.array([self.X[i][0], self.X[i][1]])            
                centers.append(center.astype("int"))
                i=i+1

                for c in centers:
                    text = "Cluster"
                    self.cv2.putText(self.frame, text, (int(c[0]) - 10, int(c[1]) - 10),
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    self.cv2.circle(self.frame, (int(c[0]), int(c[1])), 14, (0, 255, 0), -1)

        if len(self.X) >= 4:
            print ("many entries")

            K = range(1,len(self.X))
            for k in K:
                kmeanModel = KMeans(n_clusters=k).fit(self.X)
                kmeanModel.fit(self.X)
                distortions.append(sum(np.min(cdist(self.X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / len(self.X))
                #distortions.append(kmeanModel.inertia_)

            #Determine how many clusters
            kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')

            print ("distortions knee")
            print (distortions)
            print(kn.knee)

            kmeans = KMeans(n_clusters=kn.knee)
            kmeans.fit(self.X)
            y_kmeans = kmeans.fit_predict(self.X)
            centers = kmeans.cluster_centers_
            
            #print ("kmeans.labels_")
            #print (kmeans.labels_)

            counter=collections.Counter(y_kmeans)
            
            print("Cluster frequency")
            print(counter.most_common)

            print ("Largest cluster")
            print (counter.most_common(1)[0][0])

            i = 0
            for c in centers:
                #print ("Center")
                #print (c)
                    if (i == counter.most_common(1)[0][0]):
                        text = "Largest Cluster"
                        self.cv2.putText(self.frame, text, (int(c[0]) - 10, int(c[1]) - 10),
                            self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        self.cv2.circle(self.frame, (int(c[0]), int(c[1])), 14, (0, 255, 0), -1)
                        i=i+1
