import argparse
import os
import os.path

import numpy as np

from ising import IsingLattice

parser = argparse.ArgumentParser(description='Run Ising simulation!')

parser.add_argument('--k', type=float, help='Coupling constant k', default=0.4)
parser.add_argument('--n-steps', type=int, help='Number of time steps to run Monte Carlo', default=65536)
parser.add_argument('--dims', type=int, help='Number of dimensions', default=2)
parser.add_argument('--lattice-size', type=int, help='Side length of lattice', default=8)
parser.add_argument('--output-prefix', type=str, default='out', help='prefix of filenames')
parser.add_argument('--output-dir', type=str, default='ising-out', help='prefix of filenames')

args = parser.parse_args()

coupling_k = args.k
nsteps = args.n_steps
side_length = args.lattice_size
dims = args.dims
prefix = args.output_prefix
output_dir = args.output_dir

run_infostring = '{}^{} spins, k = {}'.format(side_length, dims, coupling_k)
print('Starting Ising simulation with {}, for {} steps'.format(run_infostring, nsteps))

lattice = IsingLattice(side_length, dims)

fname = '{}/{}-d{}-side{}-k{}-n{}.dat'.format(output_dir, prefix, dims, side_length, int(coupling_k*1000), nsteps)
header = 'total_mag minus_beta_h rirjsisj sisip'

os.makedirs(os.path.dirname(fname), exist_ok=True)

with open(fname, 'w') as f:
    f.write('# {}\n'.format(header))
    for step in range(nsteps):
        if step % 256 == 0:
            print('Step {} of {}, {}'.format(step, nsteps, run_infostring))

        lattice.wolff_cluster_flip(coupling_k)
        
        total_mag = lattice.total_magnetization()
        minus_beta_h = lattice.boltzmann_exp(coupling_k, 0)
        
        # rirjsisj = lattice.rirjsisj()
        # sisip = lattice.sisiprime()
        
        statistics = (total_mag, minus_beta_h)
        f.write('{} {}\n'.format(*statistics))
        # statistics = (total_mag, minus_beta_h, rirjsisj, sisip)
        # f.write('{} {} {} {}\n'.format(*statistics))

