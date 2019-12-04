import random

import matplotlib.pyplot as plt
import numpy as np

class IsingLattice():
    def __init__(self, length, dims, compute_xi=False):
        assert length % 2 == 0
        
        self.length = length
        self.dims = dims
        self.lattice = np.ones([length]*dims, dtype=np.int8)
        self.shape = self.lattice.shape
        
        if compute_xi:
            self.r2s = self._compute_r2_spin_pair()

    def _compute_r2_spin_pair(self):
        """
        Gives r^2 between all spin pairs as a 2D array,
        respecting periodic boundary conditions.
        """
        halflength = self.length // 2
        ind_arr = np.array([ind for ind, val in np.ndenumerate(self.lattice)])
        coorddeltas = ind_arr[np.newaxis] - ind_arr[:, np.newaxis]
        sep_vecs = halflength - np.abs(np.abs(coorddeltas) - halflength)
        
        return np.sum(sep_vecs**2, axis=2)
    
    def spin_val(self, index):
        return self.lattice[tuple(index)]
        
    def neighbors_of(self, index):
        for dim in range(len(self.shape)):
            for sign in [+1, -1]:
                new_index = np.copy(index)
                new_index[dim] = (index[dim] + sign) % self.shape[dim]
                yield new_index
    
    def boltzmann_exp(self, k, h):
        """
        The value

        -beta * H = (K sum_{ij} s_i s_j + h sum_i s_i)
        """
        
        coupling_term = k * np.sum([
            np.sum(self.lattice * np.roll(self.lattice, 1, axis=axis))
            for axis in range(len(self.shape))
        ])
        bias_term = h * np.sum(self.lattice)
        
        return coupling_term + bias_term
    
    def total_magnetization(self):
        return np.sum(self.lattice)
    
    def magnetization(self):
        return np.mean(self.lattice)
    
    def imshow(self):
        if len(self.shape) != 2:
            raise RuntimeError('Lattice does not have dimension 2, cannot imshow')
        return plt.imshow(self.lattice, cmap='Greys')
        
    def __str__(self):
        return str(self.lattice)
    
    def wolff_cluster(self, K):
        """
        K : number, > 0
            dimensionless coupling constant
            
        Returns
        -------
        cluster_membership : ndarray
            ndarray of same shape as lattice, with values -/+ 1;
            cluster members have -1
        """
        bond_prob = 1 - np.exp(-2*K)

        # cluster membership array; -1 for membership, +1 else
        cluster_membership = np.ones(self.shape, dtype=np.int8)

        def in_cluster(index):
            return cluster_membership[tuple(index)] == -1

        def add_to_cluster(index):
            cluster_membership[tuple(index)] = -1

        cluster_seed = np.array([random.randrange(size) for size in self.shape])
        cluster_spin_val = self.spin_val(cluster_seed)
        add_to_cluster(cluster_seed)

        consideration_stack = [cluster_seed]
        while consideration_stack:
            cur_site = consideration_stack.pop()
            for site in self.neighbors_of(cur_site):
                if self.spin_val(site) != cluster_spin_val:
                    continue
                if in_cluster(site) or random.random() >= bond_prob:
                    continue
                consideration_stack.append(site)
                add_to_cluster(site)

        return cluster_membership
    
    def wolff_cluster_flip(self, K):
        self.lattice *= self.wolff_cluster(K)
        
    def rirjsisj(self):
        """
        Compute the spin-pair sum
        
        \sum_{ij} (r_i - r_j)^2 s_i s_j
        
        which is useful in estimating correlation length.
        """
        
        flattened_spin_arr = self.lattice.flatten()
        sisj_arr = np.outer(flattened_spin_arr, flattened_spin_arr)
        rirjsisj_arr = self.r2s * sisj_arr
        rirjsisj_sum = np.sum(rirjsisj_arr)
        
        return rirjsisj_sum
    
    def sisiprime(self):
        """
        The sum
        
        ..math:: \sum_i s_i s_{i'}
        
        where ..math::`i'` is the antipodal index to ..math::`i`.
        """
        L = self.length
        rolled = np.roll(self.lattice, L//2, axis=tuple(range(len(self.shape))))
        return np.sum(self.lattice * rolled)
