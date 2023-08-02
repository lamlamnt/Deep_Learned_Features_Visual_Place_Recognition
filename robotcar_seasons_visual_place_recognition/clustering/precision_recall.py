import numpy as np

def get_precision_recall(similarity):
    #Precision
    #Threshold for cross-entropy. Lowest: 0.6. Highest: 2.5 
    threshold_list = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    average_precision_list = []
    for threshold in threshold_list:
        precision_list = []
        #Ierate over each column (each query result)
        for idx,column in enumerate(similarity.T):
            precision_list.append(np.sum(column < threshold))
        average_precision_list.append(1/(sum(precision_list)/len(precision_list)))
    
    #Recall - need gps data

def plot_precision_recall(precision,recall):
    pass

#ref query query
#ref 

#For one map run and one query run only
similarity = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/similarity_matrix.txt")
get_precision_recall(similarity)
