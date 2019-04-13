#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 23:07:44 2019

Reference: Data Science from Scratch

@author: tungutokyo
"""

from collections import Counter, defaultdict
from functools import partial 
import math, random

def entropy (class_probabilities):
    """" Given a list of class probabilities, compute the entropy """
    return sum(-p * math.log(p, 2)
               for p in class_probabilities
               if p)                         # ignore zero probabilities

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):
    """ Find the entropy from this partition of data into subsets
        subsets is a list of lists of labeled data """
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)
    

