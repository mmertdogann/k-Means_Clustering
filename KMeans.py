#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:58:57 2019

@author: mert
"""
import numpy as np
from random import randint

class KMeans:
    
    iterations = 0
    objc_value = []
    
    def __init__(self, k = 2, tolerance = 0.0001, max_iterations = 100):
        
        #initial values
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, data):

        #initializations
        self.centroids = {}
        self.randoms = {}
        
        #initialize the centroids, random 'k' elements in the dataset will be our initial centroids
        for i in range(self.k):
            self.rnd = randint(0, len(data))
            self.centroids[i] = data[self.rnd]
            self.randoms[i] = self.centroids[i]

        #begin iterations
        for i in range(self.max_iterations):
            self.iterations+=1
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            #The distance between the point and cluster; prefer to choose the nearest centroid
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            #previous centroid points
            previous = dict(self.centroids)

            #average the cluster datapoints to re-calculate the centroids
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis = 0)

            #stop criteria
            isValid = True

            #check the Stopping criterion
            for centroid in self.centroids:

                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    isValid = False

            #break out of the main loop if the results are optimal (smaller than 0.00001)
            if isValid:
                break
