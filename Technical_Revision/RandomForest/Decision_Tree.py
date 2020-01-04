# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 22:42:17 2020

@author: dangt
"""

def gini_index(groups, classes):
    # cout all samples at split point 
    n_instances = float(sum([len(group) for group in groups]))
    
    # sum weight Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            # score the group based on the score for each class
            p = [row[-1] for row in group].count(class_val)/size
            score += p*p
        # weight the group score by its relative size 
        gini += (1.0 - score) * (size/n_instances)
    
    return gini

gini_index([[[1,1], [1,0]], [[1,1], [1,0]]], [1,0])
    

    