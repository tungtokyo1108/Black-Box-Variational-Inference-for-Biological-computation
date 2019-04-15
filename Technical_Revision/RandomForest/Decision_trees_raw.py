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
    
def group_by(items, key_fn):
    """ Returns a defautdict(list), where each input item is in the list 
        whose key is key_fn(item) """
    groups = defaultdict(list)
    for item in items:
        # get the value of the specified attribute
        key = key_fn(item)
        # then add this input to the correct list 
        groups[key].append(item)
    return groups

def partition_by(inputs, attribute):
    """ Returns a dict of inputs partitioned by the attribute 
        each input is a pair (attribute_dict, label)"""
    return group_by(inputs, lambda x: x[0][attribute])

def partition_entropy_by(inputs, attribute):
    """ computes the entropy corresponding to the given partition """
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

def classify(tree, inputs):
    """ Computes the entropy corresponding to the given partition """
    if tree in [True, False]:
        return tree 
    
    # otherwise find the correct subtree
    attribute, subtree_dict = tree
    
    # None if input is missing attribute 
    subtree_key = inputs.get(attribute)
    
    if subtree_key not in subtree_dict:
        subtree_key = None
    
    # choose the appropriate subtree and use it to classify the input
    subtree = subtree_dict[subtree_key]
    return classify(subtree, inputs)

def build_tree_id3(inputs, split_candidates=None):
    # if this is our first pass, all keys of the first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
        
    # Count Trues and Falses in the inputs
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues
    
    if num_trues == 0:
        return False
    
    if num_falses == 0:
        return True
    
    if not split_candidates:
        return num_trues >= num_falses
    
    best_attribute = min(split_candidates, 
                         key=partial(partition_entropy_by, inputs))
    
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates if a != best_attribute]
    
    # recursively build the subtrees 
    subtrees = {attribute : build_tree_id3(subset, new_candidates)
                for attribute, subset in partitions.items()}
    
    subtrees[None] = num_trues > num_falses 
    return (best_attribute, subtrees)

def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]

###############################################################################

inputs = [
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'no'},   False),
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'yes'},  False),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'no'},     True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'no'},  True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'yes'},    False),
        ({'level':'Mid','lang':'R','tweets':'yes','phd':'yes'},        True),
        ({'level':'Senior','lang':'Python','tweets':'no','phd':'no'}, False),
        ({'level':'Senior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'yes','phd':'no'}, True),
        ({'level':'Senior','lang':'Python','tweets':'yes','phd':'yes'},True),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'yes'},    True),
        ({'level':'Mid','lang':'Java','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'yes'},False)
]

for key in ['level', 'lang', 'tweets', 'phd']:
    print(key, partition_entropy_by(inputs, key))
print()

senior_inputs = [(input, label)
                    for input, label in inputs if input["level"] == "Senior"]

for key in ['lang', 'tweets', 'phd']:
    print(key, partition_entropy_by(senior_inputs, key))
print()

print("Building the tree")
tree = build_tree_id3(inputs)
print(tree)

print("Junior / Java / tweets / no phd", classify(tree,
        { "level" : "Junior",
          "lang" : "Java",
          "tweets" : "yes",
          "phd" : "no"} ))
