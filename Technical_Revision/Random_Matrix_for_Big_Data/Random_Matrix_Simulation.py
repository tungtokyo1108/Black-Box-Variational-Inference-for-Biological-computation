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

