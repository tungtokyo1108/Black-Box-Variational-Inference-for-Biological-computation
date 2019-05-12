#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:48:07 2019

@author: tungutokyo
"""

import numpy as np
import numpy.linalg as la
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
############################### Wigner Matrix #################################
###############################################################################

def plot_matrix(X, d, title, ax, color='Greens'):
    T = np.linspace(-2,2,d)
    ax.hist(X, bins=50, alpha=0.4, density=1)
    ax.plot(T, np.sqrt(4-T**2)/(2*np.pi), '--', color="Red", alpha=1)
    ax.set_title(title)
    ax.set_xlabel('Eigenvalue', fontsize=10)
    ax.set_ylabel('Frequencey', fontsize=10)
    return ax

# Semicircle Law (Random symmetric matrix eigenvalues)
def Wigner_Real(N, t=1):
    a = np.triu(np.random.normal(scale=1, size=(N,N)))
    s = (a + a.T)
    s[range(N), range(N)] /= 2
    return s

N = [1000, 4000, 8000, 10000]
eigs_norm = []
fig, ax = plt.subplots(1, len(N), figsize=(20,5))
ind = 0
for i in N:
    A = Wigner_Real(i)
    eigs = la.eigvalsh(A)
    eigs = eigs/np.sqrt(i)
    ax[ind] = plot_matrix(eigs, i,'Size of Maxtrix: {}'.format(i), ax[ind])
    ind += 1
plt.tight_layout()
plt.show()

###############################################################################
##################### Gaussian Orthogonal Ensemble ############################
############################################################################### 

# The numerical histogram of the eigenvalue density 
def eig_den(N, sample, beta):
    while beta != 1 and beta != 2 and beta != 4:
        print("Error: beta has to equal 1, 2 and 4")
    
    x = []
    if beta == 1:
        # Gaussian Orthogonal Ensemble
        for s in range(sample):
            H = np.random.randn(N,N)
            H = (H + H.T) / 2
            x = np.append(x, la.eigvalsh(H))
    elif beta == 2:
        # Gaussian Unitary Ensemble
        for s in range(sample):
            H = np.random.randn(N,N)
            H = np.mat(H)
            H = (H + H.T) / 2
            x = np.append(x, la.eigvalsh(H))
    
    return x


def GOE(N):
    H1 = np.triu(np.random.normal(scale=1, size=(N,N)))
    H = H1 + H1.T
    H[range(N), range(N)] /= np.sqrt(2)
    return H

def space(H):
    N = H.shape[0]
    eigs = np.real(la.eigvalsh(H))
    eigs.sort()
    space_H = eigs[1:] - eigs[:-1]
    return space_H

def multi_space(N, sample = 1000):
    m_space = np.zeros([sample, N-1])
    for i in range(sample):
        m_space[i] = space(GOE(N))    
    return m_space

# Size of Matrix = 2 and Number of sample = 10000
N = 2 
sample = 10000
multiR = multi_space(N, sample)
meanR = np.sum(multiR, axis=0)/sample
n_multiR = multiR/meanR

fig, ax = plt.subplots(figsize=(12,12))
ax.hist(n_multiR, bins=50, alpha=0.5, density=1)
T = np.linspace(n_multiR.min(), n_multiR.max(), 101)
ax.plot(T, (np.pi/2) * T * np.exp((-T**2/4)*np.pi), '--', alpha=1, color="Red")
ax.set_xticks(np.linspace(n_multiR.min(), n_multiR.max(), 11))
