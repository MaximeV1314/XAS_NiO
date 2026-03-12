#! /usr/bin python3

# from triqs-dft_tools
from triqs_dft_tools.sumk_dft import *

# from triqs
from triqs.gf import *
from triqs.operators.util import *
import triqs.utility.mpi as mpi
#from triqs.gf.tools import fit_legendre
import triqs.gf.gf_factories
from triqs.gf.dlr_crm_dyson_solver import minimize_dyson

# from triqs-cthyb
from triqs_cthyb import *
#from triqs_cthyb.tail_fit import tail_fit

# others
import numpy as np
from h5 import HDFArchive
from datetime import datetime
import sys

# Parameters

## input/output parameters
hdf_filename = 'NiO_fixed.h5'
out_filename = 'dmft.h5'
inpdmft_filename = out_filename
start_iter = 11
inpdmft_group = "iteration-{}".format(start_iter)

## model parameters
U = 7.0
J = 1.1
beta = 40

## DMFT parameters
loops = 2
prec_mu = 0.0001
# using the double counting correction of Hariki 2017
dc_name = "Hariki2017"
# double-counting correction to the energy levels of the correlated bands, in eV
# in SumKDFT.calc_dc, one can specify the DC correction for the self-energy
# this argument will be adjusted at each iteration
dc_value = 52.
mix = 0.5

## Solver parameters
p = {}
p['n_warmup_cycles'] = int(5e+4) # converge_CT-HYB OK
p['length_cycle'] = 1000
p['n_cycles'] = int(5e+5)
p['random_seed'] = int(30345 + datetime.now().timestamp() * (mpi.rank+1))
### Legendre expansion
n_l = 40
p['measure_G_l'] = False #starting from iteration 32
### Tail fitting
p['measure_density_matrix'] = True #needed to measure the moments of the self-energy
p['use_norm_as_weight'] = True
# DLR coefficients
wmax = 15.
eps_DLR = 1e-12


## Save
# output
if mpi.is_master_node():
	with HDFArchive(out_filename, 'a') as ar:
                if 'parameters' not in ar.keys():
                        ar.create_group('parameters')
#                if 'iteration-{}'.format(start_iter+1) in ar['parameters'].keys():
#                        print("ERROR: The Group 'parameters/iteration-{}' already exists - ABORT".format(start_iter+1))
#                        sys.exit(1)
#                else:
                if True:
                        key_gp = "parameters/iteration-{}".format(start_iter+1)
                        ar.create_group(key_gp)
                        for key, val in p.items():
                                 ar[key_gp][key] = val
                        ar[key_gp]['beta'] = beta
                        ar[key_gp]['dc_type'] = dc_name
                        ar[key_gp]['dc_value'] = dc_value
                        ar[key_gp]['mixing'] = mix
                        ar[key_gp]['n_legendre'] = n_l
                        ar[key_gp]["DLR wmax"] = wmax
                        ar[key_gp]["DLR eps"] = eps_DLR

# Initialize

## Lattice Green functions
mesh_iw = MeshImFreq(beta, 'Fermion')
SK = SumkDFT(hdf_file=hdf_filename, mesh=mesh_iw, use_dft_blocks=True)

n_orb = SK.corr_shells[0]['dim']
spin_names = ["up", "down"]

## interaction Hamiltonian
#Umat, Upmat = U_matrix_kanamori(n_orb=n_orb, U_int=U, J_hund=J)
rot_basis = spherical_to_cubic(2, convention='wien2k')
Uijkl = U_matrix_slater(2, U_int=U, J_hund=J, basis='other', T=rot_basis)
Umat, Upmat = reduce_4index_to_2index(Uijkl)
h_int = h_int_density(spin_names, n_orb, U=Umat, Uprime=Upmat, map_operator_structure=SK.sumk_to_solver[0])

## cthyb solver
S = Solver(beta=beta, gf_struct=SK.gf_struct_solver_list[0], n_l=n_l)


## init from previous dmft calculation
if inpdmft_filename is not None:
        if mpi.is_master_node():
                print("Setting up self-energy and DC correction from {}/{}".format(inpdmft_filename, inpdmft_group))
        with HDFArchive(inpdmft_filename, 'r') as ar:
                SK.set_dc(ar[inpdmft_group]['dc_imp'], ar[inpdmft_group]['dc_energ'])
                SK.set_mu(ar[inpdmft_group]["chemical_potential"])

        if mpi.is_master_node():
                with HDFArchive(inpdmft_filename, 'r') as ar:

                         ### Discrete Lehmann Representation and Constrained Residual Method
                         G_tau = ar[inpdmft_group]['G_tau']
                         G_dlr = fit_gf_dlr(G_tau, wmax, eps_DLR)

                         G0_iw = ar[inpdmft_group]['G_0']
                         G0_tau = make_gf_from_fourier(G0_iw)
                         G0_dlr = fit_gf_dlr(G0_tau, wmax, eps_DLR)

                         Sigma_dlr_iw, S_H, res = minimize_dyson(G0_dlr, G_dlr, ar[inpdmft_group]['Sigma-moments'])
                         S.Sigma_iw << make_gf_imfreq(Sigma_dlr_iw, n_iw=mesh_iw.n_iw)
                         # add the Hartree shift to the iw_n self-energy, not the DLR one
                         for key in S_H.keys():
                                 S.Sigma_iw[key] += S_H[key]

                         ## symmetrize the self-energy wrt spins
                         for i in range(n_orb):
                                 S.Sigma_iw['up_{}'.format(i)] += S.Sigma_iw['down_{}'.format(i)]
                                 S.Sigma_iw['up_{}'.format(i)] *= 0.5
                                 S.Sigma_iw['down_{}'.format(i)] << S.Sigma_iw['up_{}'.format(i)]

        S.Sigma_iw << mpi.bcast(S.Sigma_iw)

# DMFT loop
for n in range(start_iter+1, start_iter+loops+1):
	if mpi.is_master_node():
		print("\n\n====> Iteration = ", n)


	## symmetrize the self-energy wrt irreps T_2g
	#S.Sigma_iw['up_2'] += S.Sigma_iw['up_3'] + S.Sigma_iw['up_4']
	#S.Sigma_iw['up_2'] /= 3
	#S.Sigma_iw['down_2'] << S.Sigma_iw['up_2']
	#for i in range(3, n_orb):
	#	S.Sigma_iw['up_{}'.format(i)] << S.Sigma_iw['up_2']
	#	S.Sigma_iw['down_{}'.format(i)] << S.Sigma_iw['down_2']

	SK.set_Sigma([S.Sigma_iw])

	# chemical potential
	chemical_potential = SK.calc_mu(precision=prec_mu)

	# set the local Green function
	S.G_iw << SK.extract_G_loc()[0]

	# In the first loop only, PASS otherwise
	if n == 1 and inpdmft_filename is None:
		# Double-counting correction
		dm = S.G_iw.density()
		#dc_value = dc_energy / S.G_iw.total_density().real
		SK.calc_dc(dm, orb=0, use_dc_value=dc_value)
		# init the real part of the self-energy:
		# real part set to Hartree term
		S.Sigma_iw << SK.dc_imp[0]['up'][0,0]

	# Calculate new G0_iw to input into the solver:
	S.G0_iw << S.Sigma_iw + inverse(S.G_iw)
	S.G0_iw << inverse(S.G0_iw)

	# Monitor the total density of the impurity
	if mpi.is_master_node():
		mpi.report("Filling of the impurity: ", S.G_iw.total_density())

	# Solve the impurity problem:
	S.solve(h_int=h_int, **p)


	## Post-processing

	### Double-counting correction
	dm = S.G_iw.density()
	#dc_value = dc_energy / S.G_iw.total_density().real
	SK.calc_dc(dm, orb=0, use_dc_value=dc_value)

	## Input/output
	if mpi.is_master_node():

		# Monitor the total density of the impurity
		mpi.report("Filling of the impurity: ", S.G_iw.total_density())

		# Write the final Sigma and G to the hdf5 archive:
		with HDFArchive(out_filename) as ar:
			ar.create_group('iteration-{}'.format(n))
			arIter = ar['iteration-{}'.format(n)]
			arIter['G_0'] = S.G0_iw
			arIter['G_tau'] = S.G_tau
			arIter['G_iw'] = S.G_iw
			arIter['Sigma_iw'] = S.Sigma_iw
			if p['measure_G_l']:
				arIter['G_l'] = S.G_l
			arIter['average_order'] = S.average_order
			arIter['auto_corr_time'] = S.auto_corr_time
			arIter['chemical_potential'] = chemical_potential
			if p['measure_density_matrix']:
				arIter['G-moments'] = S.G_moments
				arIter['Sigma-moments'] = S.Sigma_moments
				arIter['density-mat'] = S.density_matrix
				arIter['hloc_diag'] = S.h_loc_diagonalization
			arIter['dc_imp'] = SK.dc_imp
			arIter['dc_energ'] = SK.dc_energ


		### Discrete Lehmann Representation and Constrained Residual Minimisation
		G_dlr = fit_gf_dlr(S.G_tau, wmax, eps_DLR)
		G_iw_from_dlr = make_gf_imfreq(G_dlr, n_iw=mesh_iw.n_iw)
		G_tau_from_dlr = make_gf_imtime(G_dlr, n_tau=len(S.G_tau.mesh))

		G0_tau = make_gf_from_fourier(S.G0_iw)
		G0_dlr = fit_gf_dlr(G0_tau, wmax, eps_DLR)

		Sigma_dlr_iw, S_H, res = minimize_dyson(G0_dlr, G_dlr, S.Sigma_moments)
		S.Sigma_iw << make_gf_imfreq(Sigma_dlr_iw, n_iw=mesh_iw.n_iw)
		# add the Hartree shift to the iw_n self-energy, not the DLR one
		for key in S_H.keys():
			S.Sigma_iw[key] += S_H[key]

		## symmetrize the self-energy wrt spins
		for i in range(n_orb):
			S.Sigma_iw['up_{}'.format(i)] += S.Sigma_iw['down_{}'.format(i)]
			S.Sigma_iw['up_{}'.format(i)] *= 0.5
			S.Sigma_iw['down_{}'.format(i)] << S.Sigma_iw['up_{}'.format(i)]

	S.Sigma_iw << mpi.bcast(S.Sigma_iw)

if mpi.is_master_node(): print("\n Done!")

