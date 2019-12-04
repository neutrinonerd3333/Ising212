import numpy as np
import astropy.stats

from ising import IsingLattice

class IsingAnalysis:
    def __init__(self, burnin):
        self.burnin = burnin
        self.bootnum = 200

    def magnetization_per_spin(self, data_arr, nspins):
        mag_arr = data_arr['total_mag'][self.burnin:]

        m = np.abs(mag_arr).mean() / nspins
        dm = np.abs(astropy.stats.bootstrap(mag_arr, bootnum=self.bootnum)).mean(axis=1).std(ddof=1) / nspins
        return m, dm
    
    def specific_heat(self, data_arr, nspins):
        minus_beta_h_arr = data_arr['minus_beta_h_over_n'][self.burnin:]
        
        cv = np.var(minus_beta_h_arr) / nspins
        d_cv = (np.var(astropy.stats.bootstrap(minus_beta_h_arr, bootnum=self.bootnum), axis=1) / nspins).std(ddof=1)
        return cv, d_cv

    def magnetic_susceptibility(self, data_arr, nspins):
        mag_arr = data_arr['total_mag'][self.burnin:]
        
        chi = np.var(mag_arr) / nspins
        dchi = (np.var(astropy.stats.bootstrap(mag_arr, bootnum=self.bootnum), axis=1) / nspins).std(ddof=1)
        return chi, dchi
    
    def _corr_len_sq(self, data_arr, length, dims, r2sum):
        nspins = length ** dims
        cl_times_nspins = data_arr['sisip'].mean()
        cl = cl_times_nspins / nspins

        denom = np.mean(data_arr['total_mag']**2) - cl_times_nspins
        numer = np.mean(data_arr['rirjsisj']) - r2sum * cl
        xi2 = numer / denom / 2 / dims
        
        return xi2
    
    def corr_len_sq(self, data_arr, length, dims):
        fake_lattice = IsingLattice(length, dims)
        r2sum = fake_lattice.r2s.sum()
        
        burnt_in_data_arr = data_arr[self.burnin:]
        
        xi2 = self._corr_len_sq(burnt_in_data_arr, length, dims, r2sum)
        
        dxi2 = np.std([
            self._corr_len_sq(burnt_in_data_arr[index_resample.astype(int)], length, dims, r2sum)
            for index_resample in astropy.stats.bootstrap(np.arange(len(burnt_in_data_arr), dtype=int), bootnum=self.bootnum)
        ], ddof=1)
        
        return xi2, dxi2
