# Machine-Learning-kNearestNeighbor
using Java and Weka-Framework

### Introduction
k-Nearest Neigbor is an lazy learning and instance based Machine Learning algorithm, which classifies new unseen examples based on the training examples, but without training the model beforehand. When a new example has to be classified, the algorithm looks for the distance of each of its attribute values to that of other examples in the training data and classifies depending on the k instances, which has the lowest distance to it and summs up every observed class and votes for the one which occurs most often. This implementation supports weighted voting, that means, that every one of the k nearest instances are voted while having different weights. The weight is dependent of the inverse distance: 1/d(v1,v2)

### Implementation
The Java-Class NearestNeighbor can be used in the Weka-Explorer as a plugin for the kNN-Algorithms. The methods which are 
already implemented is voting and the weighted and unweighted Euclidian and Manhattan Distance Metrics.

### TODO 
Normalization of the values of the attributes. The method already exists unfinished in class NearestNeighbor.
