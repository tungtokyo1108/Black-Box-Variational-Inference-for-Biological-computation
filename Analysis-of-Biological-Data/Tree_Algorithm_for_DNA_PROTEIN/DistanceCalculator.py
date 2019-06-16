# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:30:49 2019

Reference: https://github.com/biopython/biopython/tree/master/Bio/Phylo

@author: Tung1108
"""

import itertools
import copy 
from Bio.Phylo import BaseTree
from Bio.Phylo import TreeConstruction
from Bio.Align import MultipleSeqAlignment
from Bio.SubsMat import MatrixInfo
from Bio import _py3k
from Bio._py3k import zip, range

class DistanceCalculator(object):
    dna_alphabet = ['A', 'T', 'C', 'G']
    blastn = [[5],
              [-4, 5],
              [-4, -4, 5],
              [-4, -4, -4, 5]]
    # transition/transversion scoring matrix
    # A -> G, A <- G; C -> T, C <- T: transition 
    trans = [[6],
             [-5, 6],
             [-5, -1, 6],
             [-1, -5, -5, 6]]
    
    protein_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y',
                        'Z']
    
    dna_matrices = {'blastn': blastn, 'trans': trans}
    protein_models = MatrixInfo.available_matrices
    protein_matrices = {name: getattr(MatrixInfo, name)
                        for name in protein_models}
    
    dna_models = list(dna_matrices.keys())
    models = ['identity'] + dna_models + protein_models
    
    def __init__(self, model='identity', skip_letters=None):
        if skip_letters:
            self.skip_letters = skip_letters
        elif model == 'identity':
            self.skip_letters = ()
        else:
            self.skip_letters = ('-', '*')
            
        if model == 'identity':
            self.scoring_matrix = None
        elif model in self.dna_models:
            self.scoring_matrix = TreeConstruction._Matrix(self.dna_alphabet,
                                                           self.dna_matrices[model])
        elif model in self.protein_models:
            self.scoring_matrix = self._build_protein_matrix(self.protein_matrices[model])
        else:
            raise(ValueError("Model not supported. Availabel models: " + 
                             ",".join(self.models)))
    
    def _pairwise(self, seq1, seq2):
        """ Calculate pairwise distance from two sequences """
        
        score = 0
        max_score = 0
        if self.scoring_matrix:
            max_score1 = 0
            max_score2 = 0
            for i in range(0, len(seq1)):
                l1 = seq1[i]
                l2 = seq2[i]
                if l1 in self.skip_letters or l2 in self.skip_letters:
                    continue 
                if l1 not in self.scoring_matrix.names:
                    raise ValueError("Bad alphabet '%s' in sequence '%s' at position '%s'"
                                     % (l1, seq1.id, i))
                if l2 not in self.scoring_matrix.names:
                    raise ValueError("Bad alphabet '%s' in sequence '%s' at position '%s'"
                                     % (l2, seq2.id, i))
                max_score1 += self.scoring_matrix[l1, l1]
                max_score2 += self.scoring_matrix[l2, l2]
                score += self.scoring_matrix[l1, l2]
            max_score = max(max_score1, max_score2)
        else:
            score = sum(l1 == l2 
                        for l1, l2 in zip(seq1, seq2)
                        if l1 not in self.skip_letters and l2 not in self.skip_letters)
            max_score = len(seq1)
        
        if max_score == 0:
            return 1
        
        return 1 - (score * 1.0 / max_score)
    
    def get_distance(self, msa):
        """ Return a Distance Matrix for MSA object
        
        msa: Multiple Sequence Aligment DNA or Protein multiple sequence alignment 
        """
        if not isinstance(msa, MultipleSeqAlignment):
            raise TypeError("Must provide a MultipleSeqAlignment object")
            
        names = [s.id for s in msa]
        dm = TreeConstruction.DistanceMatrix(names)
        for seq1, seq2 in itertools.combinations(msa,2):
            dm[seq1.id, seq2.id] = self._pairwise(seq1, seq2)
        return dm
    
    def _build_protein_matrix(self, subsmat):
        protein_matrix = TreeConstruction._Matrix(self.protein_alphabet)
        for k, v in subsmat.items():
            aa1, aa2 = k
            protein_matrix[aa1, aa2] = v
        return protein_matrix
    
""" Test Code """ 
from Bio import AlignIO
aln = AlignIO.read(open('msa.phy'), 'phylip')
print(aln)    

calculator = DistanceCalculator('identity')
dm = calculator.get_distance(aln)    
print(dm)

calculator_trans = DistanceCalculator('trans')
dm_trans = calculator_trans.get_distance(aln)
print(dm_trans)    
    
