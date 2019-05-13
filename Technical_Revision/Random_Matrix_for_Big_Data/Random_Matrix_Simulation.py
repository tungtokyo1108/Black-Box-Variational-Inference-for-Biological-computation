#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:48:07 2019

Reference: http://math.mit.edu/~edelman/publications/random_matrix_theory_innovative.pdf

@author: tungutokyo
"""

import numpy as np
import numpy.linalg as la
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random as rd

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

def plot_eig_GOE(X, sample, title, ax, color='Greens'):
    T = np.linspace(-np.sqrt(2), np.sqrt(2), sample)
    ax.hist(X, bins=50, alpha=0.5, density=1)
    # Definition of the semicircle distribution function
    ax.plot(T, np.sqrt(2-T**2)/np.pi, '--', alpha=1)
    ax.set_title(title)
    ax.set_xlabel('Eigenvalue', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    return ax

# The numerical histogram of the eigenvalue density 
def eig_den(N, sample, beta, rescaled=True):
    while beta != 1 and beta != 2 and beta != 4:
        print("Error: beta has to equal 1, 2 and 4")
    
    x = []
    if beta == 1:
        # Gaussian Orthogonal Ensemble
        for s in range(sample):
            if rescaled == True:
                H = np.random.randn(N,N) / np.sqrt(N)
                H = (H + H.T) / 2
            else:
                H = np.random.randn(N,N)
                H = (H + H.T) / 2
            x = np.append(x, la.eigvalsh(H))
    elif beta == 2:
        # Gaussian Unitary Ensemble
        for s in range(sample):
            #if normalize == True:
            #    M = np.random.randn(N,N) + 1j*np.random.randn(N,N)
            #    M = (M + M.T) / (2 * np.sqrt(4*N))
            if rescaled == True:
                M = (np.random.randn(N,N) + 1j * np.random.randn(N,N)) / np.sqrt(2 * N)
                M = np.mat(M)
                M = (M * M.H) / 2
            else:
                M = np.random.randn(N,N)
                M = np.mat(M)
                M = (M + M.H) / 2
            x = np.append(x, la.eigvalsh(M))
    else:
        # Gaussian Symplectic Ensemble
        A = np.random.randn(N,N) + 1j * np.random.randn(N,N)
        B = np.random.randn(N,N) + 1j * np.random.randn(N,N)
        H1 = np.hstack((A, B))
        H2 = np.hstack((-np.conjugate(B), np.conjugate(A)))
        M = np.vstack((H1,H2))
        M = np.mat(M)
        M = (M + M.H) / 2
        x = np.append(x, np.unique(la.eigvalsh(M)))
    
    return x

rand_mat = eig_den(1000, 1000, 1, rescaled=True)
# Normalized eigenvalue histogram
n, bins = np.histogram(rand_mat, bins=50, normed=1)
bin_new = []
for i in range(len(bins) - 1):
    bin_new.append((bins[i] + bins[i+1])/2)

fig, ax = plt.subplots(figsize=(12,12))
ax.hist(rand_mat, density=1, alpha=0.5, bins=50)
# Definition of the semicircle distribution function
T = np.linspace(-np.sqrt(2),np.sqrt(2), 10000)
ax.plot(T, np.sqrt(2-T**2)/np.pi, '--', alpha=1)
# ax.plot(bin_new, n*np.pi/2, 'o-', alpha=0.5, color='Red')
ax.set_xlabel("Eigenvalue", fontsize=15)
ax.set_ylabel(r'$rho(x)$', fontsize=15)

Size_matrix = [10, 50, 100, 500]
fig, ax = plt.subplots(1, len(Size_matrix), figsize=(20,5))
ind = 0
for i in Size_matrix:
    GOE_eig = eig_den(i, 1000, 1, rescaled = True)
    ax[ind] = plot_eig_GOE(GOE_eig, 1000, 'Size of Matrix: {}'.format(i), ax[ind])
    ind += 1
plt.tight_layout()
plt.show()


# Computing the level density (GUE) using Christoffol-Darboux
T = np.linspace(-1,1,10000)
def ChrisDarb(N):
    x = np.linspace(-1,1,10000)*np.sqrt(2*N)*1.3
    pold = 0*x
    p = 1+0*x
    k=p
    for j in range(N):
        pnew = (np.sqrt(2)*x*p-np.sqrt(j+1-1)*pold)/np.sqrt(j+1)
        pold = p
        p = pnew
    pnew = (np.sqrt(2)*x*p-np.sqrt(N)*pold)/np.sqrt(N+1)
    k = N*p**2 - np.sqrt(N*(N+1))*pnew*pold
    k = k * np.exp(-x**2)/np.sqrt(np.pi)
    return k

d = ChrisDarb(10)

fig, ax = plt.subplots(figsize=(12,12))
# ax.hist(rand_mat, density=1, alpha=0.5, bins=50)
# T = np.linspace(-1.5,1.5,10000)
ax.plot(T/np.sqrt(2*10), d*np.pi/np.sqrt(2*10), '--')
ax.plot(bin_new, n*np.pi/2, 'o-', alpha=0.5, color='Red')
ax.set_xlabel("Eigenvalue", fontsize=15)
ax.set_ylabel(r'$rho(x)$', fontsize=15)

# The density of spacing 
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

def plot_space_eig(X, sample, title, ax, color='Greens'):
    T = np.linspace(X.min(), X.max(), sample)
    ax.hist(X, bins=50, alpha=0.5, density=1)
    ax.plot(T, (np.pi/2)*T*np.exp((-T**2/4)*np.pi), '--', alpha=1, color='Red')
    ax.set_title(title)
    ax.set_xlabel('Eigen Space', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    return ax

# Size of Matrix = 2 and Number of sample = 10000
N = 2 
sample = 100000
multiR = multi_space(N, sample)
meanR = np.sum(multiR, axis=0)/sample
n_multiR = multiR/meanR

fig, ax = plt.subplots(figsize=(12,12))
ax.hist(n_multiR, bins=50, alpha=0.5, density=1)
T = np.linspace(n_multiR.min(), n_multiR.max(), 101)
ax.plot(T, (np.pi/2) * T * np.exp((-T**2/4)*np.pi), '--', alpha=1, color="Red")
ax.set_xticks(np.linspace(n_multiR.min(), n_multiR.max(), 11))

N = 2 
samples = [1000, 10000, 50000, 100000]
fig, ax = plt.subplots(1, len(samples), figsize=(20,5))
ind = 0
for sample in samples:
    multiR = multi_space(N, sample)
    meanR = np.sum(multiR, axis=0)/sample
    n_multiR = multiR/meanR
    ax[ind] = plot_space_eig(n_multiR, sample, "Size of Sample: {}".format(sample), ax[ind])
    ind += 1
