import pyten as ptn
from toolkit import *
import numpy as np
from scipy.linalg import hessenberg

import matplotlib.pyplot as plt
import glob, os, shutil
import warnings


#######################################################
####################   FONCTIONS   ####################
#######################################################




def dmrg_run(init_state, lat, calcdir):
    """Run a DMRG optimization using a PyTen matrix product state.

    This function saves the lattice and initial MPS state, configures a three-stage
    DMRG workflow, executes the optimization across all stages, and saves both
    intermediate and final state files.

    Parameters:
        init_state: initial MPS state used by the DMRG solver.
        lat: PyTen lattice object containing the Hamiltonian and site operators.
        calcdir: output directory for state files and log files.

    Returns:
        final_state: the optimized MPS after the final DMRG stage.
    """

    print("\n *** Run DMRG ***")
    print("Directory: ", calcdir)
    os.makedirs(calcdir, exist_ok=True)
    lat.save(calcdir + "lattice.lat")
    init_state.save(calcdir + "init_state.mps")

    # Initialize the DMRG configuration and stage definitions
    configuration = ptn.dmrg.DMRGConfig()
    stages = []

    # Stage 0: loose truncation for fast initial convergence
    stages.append(ptn.dmrg.DMRGStage())
    stages[0].trunc.maxStates = 512
    stages[0].trunc.threshold = 1e-4
    stages[0].convergenceMaxSweeps = 5
    stages[0].convMinEnergyDiff = 1e-6

    # Stage 1: tighter truncation to improve the state accuracy
    stages.append(ptn.dmrg.DMRGStage())
    stages[1].trunc.maxStates = 512
    stages[1].trunc.threshold = 1e-8
    stages[1].convergenceMaxSweeps = 5
    stages[1].convMinEnergyDiff = 1e-8

    # Stage 2: final refinement with a larger bond dimension
    stages.append(ptn.dmrg.DMRGStage())
    stages[2].trunc.maxStates = 1024
    stages[2].trunc.threshold = 1e-10
    stages[2].convergenceMinSweeps = 2
    stages[2].convergenceMaxSweeps = 20
    stages[2].convMinEnergyDiff = 1e-8

    configuration.stages = stages

    # Create the DMRG object and associate it with the Hamiltonian operator
    dmrg = ptn.mp.dmrg.PDMRG(init_state, [lat.get("H")], configuration)

    print("Run DMRG")
    for m in range(len(configuration.stages)):
        dmrg.run()
        dmrg.save(calcdir + 'stage{}.dmrg'.format(m))
        state = dmrg.get_psi()
        state.save(calcdir + "stage{}.mps".format(m))

    print("Done! ")
    final_state = dmrg.get_psi()
    final_state.save(calcdir + 'final_state.mps')

    # Move generated log files into the calculation directory
    log_name = [".tmlog", ".t1log", ".i1log"]
    k = 0
    for f in glob.glob("*log"):
        shutil.move(f, calcdir + "pyten_dmrg" + log_name[k % 3])
        k += 1

    return final_state
        




def hamiltonian_AIM(lat, U, mu, ed, V_l, e_l, imp_index):
    """Build the Anderson impurity model Hamiltonian in star form.

    The impurity is located at the site with index imp_index in the MPS chain.
    Hybridization amplitudes V_l connect the impurity to each bath site and e_l
    contains the bath energy levels. The impurity energy ed is shifted by the
    chemical potential mu from DMFT.

    Parameters:
        lat: PyTen lattice object where operators are defined.
        U: Hubbard interaction on the impurity site.
        mu: chemical potential from DMFT.
        ed: impurity on-site energy.
        V_l: list of hybridization amplitudes between impurity and bath.
        e_l: list of bath energy levels.
        imp_index: index of the impurity site in the MPS chain.
    """

    L = lat.size()
    list_op = []

    # Bath energy terms for all non-impurity sites
    j = 0
    for i in range(L):
        if i == imp_index:
            continue
        list_op.append(e_l[j] * lat.get("cu", i) * lat.get("chu", i))
        list_op.append(e_l[j] * lat.get("cd", i) * lat.get("chd", i))
        j += 1

    # Hybridization from bath to impurity
    j = 0
    for i in range(L):
        if i == imp_index:
            continue
        list_op.append(V_l[j] * lat.get("cu", i) * lat.get("chu", imp_index))
        list_op.append(V_l[j] * lat.get("cd", i) * lat.get("chd", imp_index))
        j += 1

    # Hybridization from impurity to bath (adjoint)
    j = 0
    for i in range(L):
        if i == imp_index:
            continue
        list_op.append(V_l[j] * lat.get("cu", imp_index) * lat.get("chu", i))
        list_op.append(V_l[j] * lat.get("cd", imp_index) * lat.get("chd", i))
        j += 1

    # Local interaction term on the impurity site
    list_op.append(
        U
        * lat.get("cu", imp_index)
        * lat.get("chu", imp_index)
        * lat.get("cd", imp_index)
        * lat.get("chd", imp_index)
    )

    # Impurity on-site energy, shifted by chemical potential
    list_op.append((ed - mu) * lat.get("cu", imp_index) * lat.get("chu", imp_index))
    list_op.append((ed - mu) * lat.get("cd", imp_index) * lat.get("chd", imp_index))

    H = ptn.mp.addLog(list_op)
    lat.add("H", "Hamiltonian", H, True)





def hamiltonian_CH(lat, U, mu, ed, V_l, e_l):
    """Transform the star Anderson impurity Hamiltonian to chain representation.

    This function performs a unitary transformation from the star representation
    to the chain representation using the Lanczos algorithm. The transformation
    orthogonalizes the bath degrees of freedom while preserving the impurity dynamics.

    Parameters:
        lat: PyTen lattice object (u1u1 fermionic Hubbard lattice).
        U: Hubbard interaction on the impurity site.
        mu: chemical potential from DMFT.
        ed: impurity on-site energy.
        V_l: array of hybridization amplitudes (star representation).
        e_l: array of bath energy levels (star representation).

    The Lanczos algorithm constructs an orthonormal basis of bath states,
    transforming the original star bath into a linear chain. The output is the
    Hamiltonian in this chain representation with renormalized hybridization
    and on-site energy parameters.
    """

    L = lat.size()
    N_bain = L - 1
    imp_index = 0

    # Calculate the imp-1st bath site hybridization strength
    Vch_imp = np.sqrt(np.sum(V_l**2))
    
    # creation and annihilation operators of the first bath in chain representation
    bu_ch_1  = ptn.mp.addLog([V_l[i]/Vch_imp * lat.get("cu", i+1) for i in range(N_bain)])
    bd_ch_1  = ptn.mp.addLog([V_l[i]/Vch_imp * lat.get("cd", i+1) for i in range(N_bain)])
    bhu_ch_1 = ptn.mp.addLog([V_l[i]/Vch_imp * lat.get("chu", i+1) for i in range(N_bain)])
    bhd_ch_1 = ptn.mp.addLog([V_l[i]/Vch_imp * lat.get("chd", i+1) for i in range(N_bain)])

    # Impurity-bath hybridization Hamiltonian in chain language
    H_hyb_ch = ptn.mp.addLog([Vch_imp * bu_ch_1*lat.get("chu", imp_index), Vch_imp * bd_ch_1*lat.get("chd", imp_index), \
                           Vch_imp * lat.get("cu", imp_index)*bhu_ch_1, Vch_imp * lat.get("cd", imp_index)*bhd_ch_1])

    # Bath Hamiltonian of the original star representation
    H_bath = ptn.mp.addLog([e_l[i] * lat.get("cu", i+1)*lat.get("chu", i+1) for i in range(N_bain)] + 
                             [e_l[i] * lat.get("cd", i+1)*lat.get("chd", i+1) for i in range(N_bain)])
    
    # Lanczos algorithm to orthogonalize the bath basis and compute the chain parameters
    def lanczos_CH(init_state, H_bath):
        """Apply the Lanczos algorithm to construct the chain bath representation.

        This produces an orthonormal basis {|bl>} that diagonalizes the bath
        Hamiltonian within the subspace reachable via the impurity coupling.

        Parameters:
            init_state: initial normalized state.
            H_bath: the star representation bath Hamiltonian.

        Returns:
            ech_l: array of diagonal bath energies in the chain representation.
            Vch_l: array of hybridization couplings between chain sites.
        """
        
        statel = init_state.copy()
        statel.normalise()
        statel_previous = statel.copy()*0.

        ech_l = np.zeros(N_bain, dtype=np.complex64)
        Vch_l = np.zeros(N_bain, dtype=np.complex64)

        basis = [init_state]
        
        for l in range(N_bain-1):

            # Compute the diagonal energy element <bl|H_bath|bl>
            ech_l[l] = ptn.mp.expectation(statel, H_bath)
            
            # Apply the bath Hamiltonian and truncate the MPS
            rl_state  = statel * H_bath
            rl_state.truncateKeepNorm(ptn.Truncation(threshold=1e-10, maxStates=1024))
            
            # Subtract the diagonal energy to orthogonalize: (H_bath - ech_l[l]) |bl>
            rl_state += statel * (-ech_l[l])
            rl_state.truncateKeepNorm(ptn.Truncation(threshold=1e-10, maxStates=1024))
            
            # Subtract the previous state contribution: subtract Vch_l[l] |bl-1>
            rl_state += statel_previous * (-Vch_l[l])
            rl_state.truncateKeepNorm(ptn.Truncation(threshold=1e-10, maxStates=1024))

            # Compute the norm, which gives the off-diagonal coupling strength
            Vch_l[l+1] = np.sqrt(ptn.mp.overlap(rl_state, rl_state))
            
            # Normalize to obtain the next basis state
            statel_previous = statel.copy()
            statel = rl_state * (1/(Vch_l[l+1]))

            basis.append(statel.copy())
            
            # Check orthogonality: verify that new state is orthogonal to all previous states
            for n in range(len(basis)-1):
                overlap_basis = np.abs(ptn.mp.overlap(basis[n], statel))
                if overlap_basis > 1e-6:
                    warnings.warn("<b%d|b%d> = %.2E > 1e-6"%(n+1, l+1, overlap_basis))
            
            # Check normalization: verify that new basis state is normalized
            if abs(np.abs(ptn.mp.overlap(statel, statel)) - 1) > 1e-6:
                warnings.warn("| <b%d|b%d> - 1 | = %.2E > 1e-6"%(l+1, l+1, overlap_basis))

            # Check orthogonality to the impurity-bath coupling: |H_hyb|bl> must be zero
            trystate = statel * H_hyb_ch
            trystate.truncateKeepNorm(ptn.Truncation(threshold=1e-10, maxStates=1024))
            trystate_norm = np.abs(ptn.mp.overlap(trystate, trystate))
            if trystate_norm > 1e-6:
                raise TypeError("|| H_hyb|bl> || = %.2E > 1e-6 !!!"%trystate_norm)

        # Compute the final diagonal energy element
        ech_l[-1] = ptn.mp.expectation(statel, H_bath)
        return ech_l, Vch_l

    # Run Lanczos algorithm for the spin-up bath states
    init_state_up = ptn.mp.generateNearVacuumState(lat) * bhu_ch_1
    ech_l_up, Vch_l_up = lanczos_CH(init_state_up, H_bath)
    Vch_l_up[0] = Vch_imp  # Set the impurity-bath hybridization

    # Run Lanczos algorithm for the spin-down bath states
    init_state_down = ptn.mp.generateNearVacuumState(lat) * bhd_ch_1
    ech_l_dn, Vch_l_dn = lanczos_CH(init_state_down, H_bath)
    Vch_l_dn[0] = Vch_imp  # Set the impurity-bath hybridization

    # Construct the chain representation Hamiltonian with renormalized parameters
    list_op = []

    # On-site energies for the chain bath sites (diagonal terms)
    for i in range(N_bain):
        list_op.append(ech_l_up[i] * lat.get("cu", i+1) * lat.get("chu", i+1))
        list_op.append(ech_l_dn[i] * lat.get("cd", i+1) * lat.get("chd", i+1))
    
    # Hybridization terms from site i+1 to site i (moving up the chain)
    for i in range(N_bain):
        list_op.append(Vch_l_up[i] * lat.get("cu", i+1) * lat.get("chu", i))
        list_op.append(Vch_l_dn[i] * lat.get("cd", i+1) * lat.get("chd", i))

    # Hybridization terms from site i to site i+1 (adjoint, moving down the chain)
    for i in range(N_bain):
        list_op.append(Vch_l_up[i].conj() * lat.get("cu", i) * lat.get("chu", i+1))
        list_op.append(Vch_l_dn[i].conj() * lat.get("cd", i) * lat.get("chd", i+1))

    # Impurity Hubbard interaction
    list_op.append(
        U
        * lat.get("cu", imp_index)
        * lat.get("chu", imp_index)
        * lat.get("cd", imp_index)
        * lat.get("chd", imp_index)
    )

    # Impurity on-site energy, shifted by chemical potential
    list_op.append((ed - mu) * lat.get("cu", imp_index) * lat.get("chu", imp_index))
    list_op.append((ed - mu) * lat.get("cd", imp_index) * lat.get("chd", imp_index))

    # Construct the final chain-representation Hamiltonian and store it in the lattice
    H_ch = ptn.mp.addLog(list_op)
    lat.add("H", "Hamiltonian in chaine rep", H_ch, True)






def hamiltonian_CH_v2(lat, U, mu, ed, V_l, e_l):
    """Transform the star Anderson impurity Hamiltonian to chain representation.

    This function performs a unitary transformation from the star representation
    to the chain representation using the Lanczos algorithm. The transformation
    orthogonalizes the bath degrees of freedom while preserving the impurity dynamics.
    All calculations are done in the one-body basis, so the computation of the hamiltonian
    is faster than the first version using MPS on many-body basis.

    Parameters:
        lat: PyTen lattice object (u1u1 fermionic Hubbard lattice).
        U: Hubbard interaction on the impurity site.
        mu: chemical potential from DMFT.
        ed: impurity on-site energy.
        V_l: array of hybridization amplitudes (star representation).
        e_l: array of bath energy levels (star representation).

    The Lanczos algorithm constructs an orthonormal basis of bath states,
    transforming the original star bath into a linear chain. The output is the
    Hamiltonian in this chain representation with renormalized hybridization
    and on-site energy parameters.
    """
        
    Vch_imp = np.sqrt(np.sum(V_l**2))
    N_bain  = len(V_l)
    imp_index = 0

    state1_up  = np.zeros(2*(N_bain + 1))  # spin * (# bath site + # imp site)
    state1_down  = np.zeros(2*(N_bain + 1))  # spin * (# bath site + # imp site)
    state1_up[2::2] = np.ones(N_bain) * V_l/Vch_imp
    state1_down[3::2] = np.ones(N_bain) * V_l/Vch_imp

    state_imp_up = np.zeros(2*(N_bain + 1))
    state_imp_down = np.zeros(2*(N_bain + 1))
    state_imp_up[0] = 1
    state_imp_down[1] = 1

    H_hyb = Vch_imp * (np.outer(state1_up, state_imp_up) + np.outer(state_imp_up, state1_up)
        + np.outer(state1_down, state_imp_down) + np.outer(state_imp_down, state1_down))
    # print(H_hyb, H_hyb.shape)

    H_bath = np.zeros(2*(N_bain+1))
    H_bath[2::2] = e_l
    H_bath[3::2] = e_l
    H_bath = np.diag(H_bath)
    # print(H_bath)

    def lanczos_CH_v2(init_state, H_bath):
        """Apply the Lanczos algorithm to construct the chain bath representation.

        This produces an orthonormal basis {|bl>} that diagonalizes the bath
        Hamiltonian within the subspace reachable via the impurity coupling.

        Parameters:
            init_state: initial normalized state.
            H_bath: the star representation bath Hamiltonian.

        Returns:
            ech_l: array of diagonal bath energies in the chain representation.
            Vch_l: array of hybridization couplings between chain sites.
        """

        statel = init_state.copy()
        statel /= np.linalg.norm(statel)
        statel_previous = statel.copy()*0.

        ech_l = np.zeros(N_bain, dtype=np.complex64)
        Vch_l = np.zeros(N_bain, dtype=np.complex64)

        basis = [init_state]

        for l in range(N_bain-1):

            # Compute the diagonal energy element <bl|H_bath|bl>
            ech_l[l] = statel.T @ (H_bath @ statel)

            # Apply the bath Hamiltonian
            rl_state  = H_bath @ statel - ech_l[l] * statel - Vch_l[l] * statel_previous

            # Compute the norm, which gives the off-diagonal coupling strength
            Vch_l[l+1] = np.sqrt(np.abs(rl_state.T @ rl_state))
            
            # Normalize to obtain the next basis state
            statel_previous = statel.copy()
            statel = rl_state * (1/(Vch_l[l+1]))

            basis.append(statel.copy())

            # Check orthogonality: verify that new state is orthogonal to all previous states
            for n in range(len(basis)-1):
                overlap_basis = np.abs(np.dot(basis[n], statel))
                if overlap_basis > 1e-6:
                    warnings.warn("<b%d|b%d> = %.2E > 1e-6"%(n+1, l+1, overlap_basis))
            
            # Check normalization: verify that new basis state is normalized
            if abs(np.abs(np.linalg.norm(statel)) - 1) > 1e-6:
                warnings.warn("| <b%d|b%d> - 1 | = %.2E > 1e-6"%(l+1, l+1, overlap_basis))

            # Check orthogonality to the impurity-bath coupling: |H_hyb|bl> must be zero
            trystate = H_hyb @ statel
            trystate_norm = np.abs(np.linalg.norm(trystate))
            if trystate_norm > 1e-6:
                warnings.warn("|| H_hyb|bl> || = %.2E > 1e-6 !!!"%trystate_norm)

        # Compute the final diagonal energy element
        ech_l[-1] = statel.T @ (H_bath @ statel)
        return ech_l, Vch_l

    # Run Lanczos algorithm for the spin-up bath states
    ech_l_up, Vch_l_up = lanczos_CH_v2(state1_up, H_bath)
    Vch_l_up[0] = Vch_imp  # Set the impurity-bath hybridization

    # Run Lanczos algorithm for the spin-down bath states
    ech_l_dn, Vch_l_dn = lanczos_CH_v2(state1_down, H_bath)
    Vch_l_dn[0] = Vch_imp  # Set the impurity-bath hybridization

    # Construct the chain representation Hamiltonian with renormalized parameters
    list_op = []

    # On-site energies for the chain bath sites (diagonal terms)
    for i in range(N_bain):
        list_op.append(ech_l_up[i] * lat.get("cu", i+1) * lat.get("chu", i+1))
        list_op.append(ech_l_dn[i] * lat.get("cd", i+1) * lat.get("chd", i+1))
    
    # Hybridization terms from site i+1 to site i (moving up the chain)
    for i in range(N_bain):
        list_op.append(Vch_l_up[i] * lat.get("cu", i+1) * lat.get("chu", i))
        list_op.append(Vch_l_dn[i] * lat.get("cd", i+1) * lat.get("chd", i))

    # Hybridization terms from site i to site i+1 (adjoint, moving down the chain)
    for i in range(N_bain):
        list_op.append(Vch_l_up[i].conj() * lat.get("cu", i) * lat.get("chu", i+1))
        list_op.append(Vch_l_dn[i].conj() * lat.get("cd", i) * lat.get("chd", i+1))

    # Impurity Hubbard interaction
    list_op.append(
        U
        * lat.get("cu", imp_index)
        * lat.get("chu", imp_index)
        * lat.get("cd", imp_index)
        * lat.get("chd", imp_index)
    )

    # Impurity on-site energy, shifted by chemical potential
    list_op.append((ed - mu) * lat.get("cu", imp_index) * lat.get("chu", imp_index))
    list_op.append((ed - mu) * lat.get("cd", imp_index) * lat.get("chd", imp_index))

    # Construct the final chain-representation Hamiltonian and store it in the lattice
    H_ch = ptn.mp.addLog(list_op)
    lat.add("H", "Hamiltonian in chaine rep", H_ch, True)





def hamiltonian_NO(lat, U, mu, ed, V_l, e_l, dm, imp_index):

    """Transform the star Anderson impurity Hamiltonian to chain representation.
    This function performs a unitary transformation from the star representation
    to the natural orbital representation using the Lanczos algorithm. We follow
    the notes of Jason Kaye.

    Parameters:
        lat: PyTen lattice object (u1u1 fermionic Hubbard lattice).
        U: Hubbard interaction on the impurity site.
        mu: chemical potential from DMFT.
        ed: impurity on-site energy.
        V_l: array of hybridization amplitudes (star representation).
        e_l: array of bath energy levels (star representation).
        dm : density_matrix of the system (calculated previously using DMRG)
        imp_index : index of the impurity in the dm (be careful, it is 2*imp_index because of spin)
    """

    verbosity = True

    ###########################################################
    # Check real part of density matrix and put imp at site 0 #
    ###########################################################

    if np.max(dm.imag) > 1e-8 :
        print("Warning : density_matrix has imaginary part greater than 1e-8. We discard them anyway.")
    dm = dm.real

    dm = np.roll(np.roll(dm, -2*imp_index, axis=0), -2*imp_index, axis=1) # put imp in first index

    ###################################################################
    # exctract spin up an down part of the dm + exctract the bath dm  #
    ###################################################################

    ########## up ###########

    rho_up = dm[::2, ::2]       # spin up dm
    rho_bath_up = rho_up[1:, 1:]    # spin up bath dm
    nu_up, P_bath_up = np.linalg.eigh(rho_bath_up)  # diag spin up bath dm

    ######### dn ############

    rho_dn = dm[1::2, 1::2]     # spin down dm
    rho_bath_dn = rho_dn[1:, 1:]    # spin down bath dm
    nu_dn, P_bath_dn = np.linalg.eigh(rho_bath_dn)  # diag spin down bath dm

    #########################

    N_bath = len(nu_up)

    ###########################################################
    # Construction of the 1-body part of the hamiltonian      #
    ###########################################################

    H_1B = np.zeros((N_bath+1, N_bath+1))
    H_1B[0,0] = ed - mu
    H_1B[0, 1:] = np.roll(V_l, -imp_index)  # roll to put again the imp in first site
    H_1B[1:, 0] = np.roll(V_l, -imp_index)
    for i in range(1, N_bath+1) : H_1B[i, i] = np.roll(e_l, -imp_index)[i-1]

    ###################################################################
    # find the b active site, such that Tr(rho_imp) + Tr(rho_b) = 1   #
    ###################################################################

    ########## up ###########

    error_up = np.zeros(N_bath)
    for i in range(N_bath): 
        error_up[i] = np.abs(rho_up[0,0] + nu_up[i] - 1)    # erreur = |Tr(rho_imp) + Tr(rho_b) - 1|

    bup_idx = np.argmin(error_up)           # active site = the one that minimize the error
    state_bup = np.zeros(N_bath+1)
    state_bup[1:] = P_bath_up[:, bup_idx]   # state of the active site in the full basis (imp+bath)
    rho_bup = nu_up[bup_idx]                # occupation of the active site (already ev of rho_bath)

    ########## dn ###########

    error_dn = np.zeros(N_bath)
    for i in range(N_bath): 
        error_dn[i] = np.abs(rho_dn[0,0] + nu_dn[i] - 1)    # erreur = |Tr(rho_imp) + Tr(rho_b) - 1|

    bdn_idx = np.argmin(error_dn)           # active site = the one that minimize the error
    state_bdn = np.zeros(N_bath+1)
    state_bdn[1:] = P_bath_dn[:, bdn_idx]   # state of the active site in the full basis (imp+bath)
    rho_bdn = nu_dn[bdn_idx]                # occupation of the active site (already ev of rho_bath)

    ######################################################################
    #                   define imp vector basis                          #
    ######################################################################

    state_imp = np.zeros(N_bath+1)
    state_imp[0] = 1.

    ######################################################################
    # OUTDATED (not used anymore) : basis and imp + b basis. Then, calculate rho_ib  #
    ######################################################################

    ########## up ###########

    # ibup_basis = np.column_stack([state_imp, state_bup])
    # rho_ib_up = ibup_basis.T @ rho_up @ ibup_basis          # density matrix of rho_ib

    ########## dn ###########

    # ibdn_basis = np.column_stack([state_imp, state_bdn])
    # rho_ib_dn = ibdn_basis.T @ rho_dn @ ibdn_basis

    ###########################################################
    # OUTDATED (not used anymore) : construct AB and B state, st rho_B = 0 and rho_AB = 1   #
    ###########################################################

    ########## up ###########

    # Diagonalize rho_ib --> give the anti-bonding and bonding change of basis P_ib
    # nu_ibup, P_ibup = np.linalg.eigh(rho_ib_up)
    # state_AB_up = ibup_basis @ P_ibup[:, 0]   # # come back to initial basis. occupation ~ 0
    # state_B_up  = ibup_basis @ P_ibup[:, 1]   # occupation ~ 1

    ########## dn ###########

    # nu_ibdn, P_ibdn = np.linalg.eigh(rho_ib_dn)
    # state_AB_dn = ibdn_basis @ P_ibdn[:, 0]   # occupation ~ 0
    # state_B_dn  = ibdn_basis @ P_ibdn[:, 1]   # occupation ~ 1

    # print("\nB occupation  =", state_B_up.T @ rho_up @ state_B_up)
    # print("AB occupation =", state_AB_up.T @ rho_up @ state_AB_up)
    # print("B norm  =", np.linalg.norm(state_B_up))
    # print("AB norm =", np.linalg.norm(state_AB_up))
    # print("<B|AB>  =", state_B_up.T @ state_AB_up)
    

    ############################################################################
    # construct the first two rotation basis (diag bath dm + AB/B) before Lanczos  #
    ############################################################################

    ########## up ###########

    P_bath_up_upfold = np.zeros((N_bath+1, N_bath+1))   # rotation of bath diag in full space (imp+bath)
    P_bath_up_upfold[0,0]   = 1
    P_bath_up_upfold[1:,1:] = P_bath_up                 

    H_prime_up = P_bath_up_upfold.T @ H_1B @ P_bath_up_upfold   # first rotation on H_star

    C_up = np.zeros((N_bath+1, N_bath+1))    # second rotation to define bonding / anti-bonding state (see notes of Jason Kaye)
    C_up[0, bup_idx] = 1/np.sqrt(2)             
    C_up[bup_idx+1, bup_idx] = -1/np.sqrt(2)    
    C_up[0, bup_idx+1] = 1/np.sqrt(2)         
    C_up[bup_idx+1, bup_idx+1] = 1/np.sqrt(2) 

    # C_up[0, bup_idx] = P_ibup[0,0]
    # C_up[bup_idx+1, bup_idx] = P_ibup[1,0]  
    # C_up[0, bup_idx+1] = P_ibup[0,1]
    # C_up[bup_idx+1, bup_idx+1] =  P_ibup[1,1]

    for i in range(bup_idx): C_up[i+1, i] = 1
    for i in range(bup_idx+2, N_bath+1) : C_up[i, i] = 1

    state_AB_up = (P_bath_up_upfold @ C_up).T[bup_idx]       # anti-bonding state (only for debuguing)
    state_B_up  = (P_bath_up_upfold @ C_up).T[bup_idx+1]     # bonding state (only for debuguing)
    occ_AB_up = state_AB_up.T @ rho_up @ state_AB_up         # anti-bonding occupation (only for debuguing)
    occ_B_up  = state_B_up.T @ rho_up @ state_B_up           # bonding occupation (only for debuguing)
    occ_outdiag_up = state_AB_up.T @ rho_up @ state_B_up     # <AB|rho|B> occ

    H_prime_prime_up = C_up.T @ H_prime_up @ C_up   # second rotation

    ########## dn ###########

    P_bath_dn_upfold = np.zeros((N_bath+1, N_bath+1))   # rotation of bath diag in full space (imp+bath)
    P_bath_dn_upfold[0,0]   = 1
    P_bath_dn_upfold[1:,1:] = P_bath_dn

    H_prime_dn = P_bath_dn_upfold.T @ H_1B @ P_bath_dn_upfold   # first rotation on H_star

    C_dn = np.zeros((N_bath+1, N_bath+1))     # second rotation to define bonding / anti-bonding state (see notes of Jason Kaye)
    C_dn[0, bdn_idx] = 1/np.sqrt(2)             
    C_dn[bdn_idx+1, bdn_idx] = -1/np.sqrt(2)    
    C_dn[0, bdn_idx+1] = 1/np.sqrt(2)         
    C_dn[bdn_idx+1, bdn_idx+1] = 1/np.sqrt(2) 

    # C_dn[0, bup_idx] = P_ibdn[0,0]
    # C_dn[bup_idx+1, bup_idx] = P_ibdn[1,0]  
    # C_dn[0, bup_idx+1] = P_ibdn[0,1]
    # C_dn[bup_idx+1, bup_idx+1] =  P_ibdn[1,1]
    for i in range(bdn_idx): C_dn[i+1, i] = 1
    for i in range(bdn_idx+2, N_bath+1) : C_dn[i, i] = 1    

    state_AB_dn = (P_bath_up_upfold @ C_dn).T[bdn_idx]      # anti-bonding state (only for debuguing)
    state_B_dn  = (P_bath_up_upfold @ C_dn).T[bdn_idx+1]    # bonding state (only for debuguing)
    occ_AB_dn = state_AB_dn.T @ rho_dn @ state_AB_dn        # anti-bonding occupation (only for debuguing)
    occ_B_dn = state_B_dn.T @ rho_dn @ state_B_dn           # bonding occupation (only for debuguing)
    occ_outdiag_dn = state_AB_dn.T @ rho_dn @ state_B_dn

    H_prime_prime_dn = C_dn.T @ H_prime_dn @ C_dn   # second rotation


    ############################################################################
    #         Tridiag of the two block and go back so imp-active basis         #
    ############################################################################

    ########## up ###########

    H_empty_up = H_prime_prime_up[0:bup_idx+1, 0:bup_idx+1]     # first block (empty states)
    _, Q_empty_up = hessenberg(H_empty_up, calc_q=True)         # Q_empty = rotation to tridiag H_empty

    H_full_up  = H_prime_prime_up[bup_idx+1:, bup_idx+1:]       # first block (filledstates)
    _, Q_full_up = hessenberg(H_full_up, calc_q=True)           # Q_full = rotation to tridiag H_full

    Q = np.block([[Q_empty_up, np.zeros((len(Q_empty_up), len(Q_full_up)))], 
                  [np.zeros((len(Q_full_up), len(Q_empty_up))), Q_full_up]])    # tridiag rotation in the full space

    H_tri_up = C_up @ Q.T @ H_prime_prime_up @ Q @ C_up.T   #third and fourth rotation

    ########## dn ###########

    H_empty_dn = H_prime_prime_dn[0:bdn_idx+1, 0:bdn_idx+1]     # first block (empty states)
    _, Q_empty_dn = hessenberg(H_empty_dn, calc_q=True)         # Q_empty = rotation to tridiag H_empty

    H_full_dn  = H_prime_prime_dn[bdn_idx+1:, bdn_idx+1:]       # first block (filledstates)
    _, Q_full_dn = hessenberg(H_full_dn, calc_q=True)           # Q_full = rotation to tridiag H_full

    Q = np.block([[Q_empty_dn, np.zeros((len(Q_empty_dn), len(Q_full_dn)))], 
                  [np.zeros((len(Q_full_dn), len(Q_empty_dn))), Q_full_dn]])    # tridiag rotation in the full space

    H_tri_dn = C_dn @ Q.T @ H_prime_prime_dn @ Q @ C_dn.T   #third and fourth rotation

    ############################################################################
    #         Construction of the Hamiltonian (MPO)         #
    ############################################################################

    list_op = []

    # bath on site energy
    for i in range(1, N_bath+1):
        list_op.append(H_tri_up[i, i] * lat.get("cu", i) * lat.get("chu", i))
        list_op.append(H_tri_dn[i, i] * lat.get("cd", i) * lat.get("chd", i))

    # hopping between bath site (with the active site)
    for i in range(1, N_bath):
        list_op.append(H_tri_up[i+1, i] * lat.get("cu", i+1) * lat.get("chu", i))
        list_op.append(H_tri_up[i, i+1] * lat.get("cu", i) * lat.get("chu", i+1))
        list_op.append(H_tri_dn[i+1, i] * lat.get("cd", i+1) * lat.get("chd", i))
        list_op.append(H_tri_dn[i, i+1] * lat.get("cd", i) * lat.get("chd", i+1))

    # Impurity Hubbard interaction
    list_op.append(U * lat.get("cu", 0) * lat.get("chu", 0)
                    * lat.get("cd", 0) * lat.get("chd", 0))

    # Impurity on-site energy, shifted by chemical potential
    list_op.append((ed - mu) * lat.get("cu", 0) * lat.get("chu", 0))
    list_op.append((ed - mu) * lat.get("cd", 0) * lat.get("chd", 0))

    # hybridation between the impurity and -empty chain, -active site and -filled chain.
    for i in range(bup_idx, bup_idx+3):
        list_op.append(H_tri_up[0, i] * lat.get("cu", i) * lat.get("chu", 0))
        list_op.append(H_tri_up[i, 0] * lat.get("cu", 0) * lat.get("chu", i))
    for i in range(bdn_idx, bdn_idx+3):
        list_op.append(H_tri_dn[0, i] * lat.get("cd", i) * lat.get("chd", 0))
        list_op.append(H_tri_dn[i, 0] * lat.get("cd", 0) * lat.get("chd", i))

    # Construct the final chain-representation Hamiltonian and store it in the lattice
    H_NO = ptn.mp.addLog(list_op)
    lat.add("H", "Hamiltonian in natural orbital rep", H_NO, True)


    if verbosity :
        calcdir = "img/hamiltonian_NO/sebastian_dm/N%d_NO"%N_bath
        os.makedirs(calcdir, exist_ok=True)

        # plot initial hamiltonian (in * rep)
        dict0 = {imp_index:"imp"}
        hamiltonian_plot(np.roll(np.roll(H_1B, imp_index, axis=0), imp_index, axis=1), 
                         dict0, title=r"$H_{\star}^{1B}$", file=calcdir+"/star_rep_N%d.png"%N_bath)

        print("\n############# up #############\n")

        # diag of up dm
        print("Diagonal of the density matrix in ascending order : \n", np.sort(np.diag(rho_up)))
        print("Eigenvalues of the bath density matrix : \n", nu_up)
        print("Impurity occupation : \n", rho_up[0,0])

        # find b active site
        print("\nerror = | Tr(rho_imp) + Tr(rho_b) - 1 | = ", np.min(error_up))
        print("b_idx = ", bup_idx)
        print("bstate occupation : \n", rho_bup)
        print("bstate : \n", state_bup)

        # construction of the ibup basis
        # print("\nrho_ib =")
        # print(rho_ib_up)

        # bonding and anto-bonding states
        #print("\nEigenvalues of rho_ib:")
        #print(nu_ibup)
        #print("\nEigenvectors:")
        #print(P_ibup)
        print("\noccupation AB, B = ", occ_AB_up, occ_B_up)
        print("\n<AB|rho|B> =",  occ_outdiag_up)

        print("\nEigenvalue of H_star - H_NO=",  np.linalg.norm(np.linalg.eigh(H_1B)[0] - np.linalg.eigh(H_tri_up)[0]))

        dict1 = {0:"imp", (bup_idx+1):"active"}
        dict2 = {bup_idx:"AB", (bup_idx+1):"B"}

        hamiltonian_plot(H_prime_up, dict1, title=r"$H' = P^T H_{\star}^{1B} P$", 
                        file=calcdir+"/up_H1_prime_N%d.png"%N_bath) # first rotation (diag dm)
        hamiltonian_plot(H_prime_prime_up, dict2, title=r"$H'' = C^TP^T H_{\star} PC$", 
                        file=calcdir+"/up_H2_prime_prime_N%d.png"%N_bath) # second rotation (A / AB)
        hamiltonian_plot(H_tri_up, dict1, title=r"$H_{NO}^{1B} = CT^TC^TP^T H_{\star} PCTC^T$", 
                        file=calcdir+"/up_H3_tri_N%d.png"%N_bath)       # last rotation ((A / AB)^T @ tridiag)


        print("\n############# down #############\n")

        # diag of up dm
        print("Diagonal of the density matrix in ascending order : \n", np.sort(np.diag(rho_dn)))
        print("Eigenvalues of the bath density matrix : \n", nu_dn)
        print("Impurity occupation : \n", rho_dn[0,0])

        # find b active site
        print("\nerror = | Tr(rho_imp) + Tr(rho_b) - 1 | = ", np.min(error_dn))
        print("b_idx = ", bdn_idx)
        print("bstate occupation : \n", rho_bdn)
        print("bstate : \n", state_bdn)

        # construction of the ibup basis
        # print("\nrho_ib =")
        # print(rho_ib_dn)

        # bonding and anto-bonding states
        #print("\nEigenvalues of rho_ib:")
        #print(nu_ibdn)
        #print("\nEigenvectors:")
        #print(P_ibdn)
        print("\noccupation AB, B = ", occ_AB_dn, occ_B_dn)
        print("\n<AB|rho|B> =", occ_outdiag_dn)

        print("\nEigenvalue of H_star - H_NO=",  np.linalg.norm(np.linalg.eigh(H_1B)[0] - np.linalg.eigh(H_tri_dn)[0]))

        dict1 = {0:"imp", (bdn_idx+1):"active"}
        dict2 = {bdn_idx:"AB", (bdn_idx+1):"B"}
        hamiltonian_plot(H_prime_dn, dict1, title=r"$H' = P^T H_{\star}^{1B} P$", 
                         file=calcdir+"/dn_H1_prime_N%d.png"%N_bath)   # first rotation (diag dm)
        hamiltonian_plot(H_prime_prime_dn, dict2, title=r"$H'' = C^TP^T H_{\star} PC$", 
                         file=calcdir+"/dn_H2_prime_prime_N%d.png"%N_bath) # second rotation (A / AB)
        hamiltonian_plot(H_tri_dn, dict1, title=r"$H_{NO}^{1B} = CT^TC^TP^T H_{\star} PCTC^T$", 
                         file=calcdir+"/dn_H3_tri_N%d.png"%N_bath)       # last rotation ((A / AB)^T @ tridiag)




def init_state_naive(lat, e_l, imp_index, init_state_method="random", occ_deriv = 0, mag = 0.5):
    """Construct an initial MPS state for the AIM calculation.

    The bath valence sites are filled first depending on the bath energies. The
    impurity site is then initialized using the selected method.

    Parameters:
        lat: PyTen lattice object for state construction.
        e_l: list of bath energies used to determine valence filling.
        imp_index: index of the impurity site in the MPS chain.
        init_state_method: method for initializing the impurity state.
            Options: 'random', 'singlet', 'triplet', 'up', or 'down'.

    Returns:
        state: initial MPS state to be used by DMRG.
    """

    state = ptn.mp.generateNearVacuumState(lat)
    L = lat.size()

    # Fill the valence bath orbitals for negative energy levels
    j = 0

    while e_l[j] <= 0 : 
        if j != imp_index :
            state *= lat.get("chu", j)
            state *= lat.get("chd", j)
        j += 1

    if init_state_method == "singlet":
        print("singlet init state.")
        stateu = state * (1 / np.sqrt(2) * lat.get("chu", imp_index))
        stated = state * (-1 / np.sqrt(2) * lat.get("chd", imp_index))
        state = stateu + stated

    elif init_state_method == "triplet":
        print("triplet init state.")
        stateu = state * (1 / np.sqrt(2) * lat.get("chu", imp_index))
        stated = state * (1 / np.sqrt(2) * lat.get("chd", imp_index))
        state = stateu + stated

    elif init_state_method == "up":
        state *= lat.get("chu", imp_index)

    elif init_state_method == "down":
        state *= lat.get("chd", imp_index)

    elif init_state_method == "sebastian":
        state1 = state.copy()
        state1 *= lat.get("chu", imp_index)
        state1 *= 1/np.sqrt(2) * lat.get("cu", j-1)

        state2 = state.copy()
        state2 *= lat.get("chd", imp_index)
        state2 *= 1/np.sqrt(2) * lat.get("cd", j-1)

        state = state1 + state2

    else:
        print("random init state.")
        if L % 2 == 0:
            remp = L - 1
        else:
            remp = L
        state = ptn.mp.generateCompleteState(lat, "%d, %.1f" % (remp - occ_deriv, mag))

    return state

