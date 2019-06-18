# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:06:55 2019

@author: Tung1108
"""

from __future__ import print_function, division, absolute_import 

import copy
import json
import numpy as np
from numpy.testing import assert_equal
import jsonctmctree.interface 

s_mtmam = """\
 32                                                                         
  2   4                                                                     
 11   0 864                                                                 
  0 186   0   0                                                             
  0 246   8  49   0                                                         
  0   0   0 569   0 274                                                     
 78  18  47  79   0   0  22                                                 
  8 232 458  11 305 550  22   0                                             
 75   0  19   0  41   0   0   0   0                                         
 21   6   0   0  27  20   0   0  26 232                                     
  0  50 408   0   0 242 215   0   0   6   4                                 
 76   0  21   0   0  22   0   0   0 378 609  59                             
  0   0   6   5   7   0   0   0   0  57 246   0  11                         
 53   9  33   2   0  51   0   0  53   5  43  18   0  17                     
342   3 446  16 347  30  21 112  20   0  74  65  47  90 202                 
681   0 110   0 114   0   4   0   1 360  34  50 691   8  78 614             
  5  16   6   0  65   0   0   0   0   0  12   0  13   0   7  17   0         
  0   0 156   0 530  54   0   1 1525 16  25  67   0 682   8 107   0  14    
398   0   0  10   0  33  20   5   0 2220 100  0 832   6   0   0 237   0   0\
"""

s_distn = """
0.0692 0.0184 0.0400 0.0186 0.0065 0.0238 0.0236 0.0557 0.0277 0.0905
0.1675 0.0221 0.0561 0.0611 0.0536 0.0725 0.0870 0.0293 0.0340 0.0428
"""

s_aas = 'ARNDCQEGHILKMFPSTWYV'

def main():
    nstates = len(s_aas)
    assert_equal(nstates, 20)
    d = {a : i for i, a in enumerate(s_aas)}
    distn = [float(x) for x in s_distn.strip().split()]
    assert_equal(len(distn), nstates)
    lines = s_mtmam.splitlines()
    assert_equal(len(lines), nstates-1)
    rate_matrix = np.zeros((nstates, nstates), dtype=int)
    for i, line in enumerate(lines):
        row_index = i + 1
        row = [int(x) for x in line.strip().split()]
        assert_equal(len(row), row_index)
        rate_matrix[row_index, :row_index] = row
    rate_matrix = np.multiply(rate_matrix + rate_matrix.T, distn)
    exit_rates = rate_matrix.sum(axis=1)
    
    scene = {
            "node_count" : 12,
            "process_count": 1,
            "state_space_shape": [20],
            "tree": {
                    "row_nodes": [
                            0, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11],
                    "column_nodes": [
                            8, 1, 2, 7, 9, 3, 10, 6, 11, 4, 5],
                    "edge_rate_scaling_factors": [
                            0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                            0.001, 0.001, 0.001, 0.001, 0.001],
                    "edge_processes": [
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    }
            }
            
    scene['root_prior'] = {
            "states": [[i] for i in range(nstates)],
            "probabilities": distn}
    
    # The process definition 
    triples = []
    for i in range(nstates):
        for j in range(nstates):
            r = rate_matrix[i, j]
            if i != j and r:
                triples.append((i, j, r))
    row_states, col_states, rates = zip(*triples)
    scene['process_definitions'] = [{
            "row_states": [[s] for s in row_states],
            "column_states": [[s] for s in col_states],
            "transition_rates": rates}]
    
    # The observed data 
    sequences = []
    with open('mtCDNApri.aa') as fin:
        lines = fin.readlines()
        header = lines[0]
        for line in lines[1:]:
            name, sequence = line.strip().split()
            sequences.append([d[x] for x in sequence])
    columns = [list(x) for x in zip(*sequences)]
    nsites = len(columns)
    scene['observed_data'] = {
            "nodes": [0,1,2,3,4,5,6],
            "variables": [0, 0, 0, 0, 0, 0, 0],
            "iid_observations": columns}
    
    # These include the log likelihood,
    # some dwell time expectation, and some transition count expectations
    requests = [
            {"property": "SNNLOGL"},
            {"property": "SDWDWEL",
             "state_reduction": {
                     "states": [[i] for i in range(nstates)],
                     "weights": exit_rates.tolist()
                     }},
            {"property": "SDNTRAN",
             "transition_reduction": {
                     "row_states": [[s] for s in row_states], 
                     "column_states": [[s] for s in col_states], 
                     "weights": [1] * len(triples)}}
            ]
    
    j_in = {
            "scene": scene,
            "requests": requests}
    
    arr = []
    j_out = None
    for i in range(7):
        if j_out is None:
            j_out = jsonctmctree.interface.process_json_in(j_in)
        else:
            dwells = j_out['responses'][1]
            transitions = j_out['responses'][2]
            scaling_factors = [t/d for t, d in zip(transitions, dwells)]
            j_in['scene']['tree']['edge_rate_scaling_factors'] = scaling_factors
            j_out = jsonctmctree.interface.process_json_in(j_in)
        arr.append(copy.deepcopy(j_out))
    
    print(json.dumps(arr, indent=4))
    
main()
        
