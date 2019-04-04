#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:29:50 2019
@author: mert
"""

#import necessary libraries
from KMeans import KMeans
import matplotlib.pyplot as plt
import numpy as np


def handle_data(input_file):
    data = []
    handle = open(input_file)
    for line in handle:
        line = line.strip() # remove new lines
        word = line.split(',') #split from commas
        data.extend([[word[0],word[1]]])
    
    data = np.asarray(data, dtype=np.float32)
    return data

def main():
    
    #load data
    X = handle_data('data2.txt')
    km = KMeans(5)
    km.fit(X)

    #Plotting
    colors = 10*['gold', 'mediumseagreen', 'orangered', 'lightpink', 'coral', 'mediumslateblue', 'violet', 'magenta']
    plt.figure(figsize=(10,10))

    #plotting each feature by using corresponding color
    for classification in km.classes:
        color = colors[classification]
        
        #features
        for features in km.classes[classification]:
            plt.scatter(features[0], features[1], color = color,s = 10)
            
        #plt.scatter(np.mean(features[0]), np.mean(features[1]), marker='*', c = 'k',s = 150)
        
    #Centroid centers   
    for centroid in km.centroids:
        plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], c='k', s = 100, marker = "x")

    #random inital points
    for l in range(km.k):
        plt.scatter(km.randoms[l][0], km.randoms[l][1], marker='*', c = 'k',s = 100)
        
    #plot attributes
    plt.legend(['* = Initial random points','X = Final cluster centers'])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('k-Means');
    plt.show()
    print('\t\t\tIteration:',km.iterations)
    print('\n\t\t\tk value: ',km.k)
 
    
main()