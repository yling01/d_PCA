'''
Tim Ling

Last update: 2020.05.18
'''
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.lib.distances import calc_dihedrals

import numpy as np
from sklearn.decomposition import PCA
import os

def ThreeToOne(three_letter_code):
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    return d[three_letter_code]

'''
Arguments: 
    
    u: (mda object) protein trajectory

Returns: 
    
    dihedral_angle: (np array (n observations, m features)) dihedral angles

Does:
    
    Calculates all of the phi psi angles in the protein
'''
def calculate_phi_psi(u, start, stop):
    u_protein = u.select_atoms("protein")
    num_res = len(u_protein.residues)
    num_dihedral = num_res * 2
    ags_phi = [res.phi_selection() for res in u_protein.residues[1:]]
    ags_psi = [res.psi_selection() for res in u_protein.residues[0:-1]]
    R_phi = Dihedral(ags_phi).run(start=start, stop=stop)
    R_psi = Dihedral(ags_psi).run(start=start, stop=stop)
    last_psi_group = u_protein.select_atoms("resid " + str(num_res) +\
                                    " and name N", "resid " +\
                                    str(num_res) +\
                                    " and name CA", "resid " +\
                                    str(num_res) +\
                                    " and name C", "resid 1 and name N")

    first_phi_group = u_protein.select_atoms("resid " +\
                                     str(num_res) +\
                                     " and name C", "resid 1 and name N", 
                                     "resid 1 and name CA", "resid 1 and name C")

    first_phi = Dihedral([first_phi_group]).run(start=start, stop=stop)
    last_psi = Dihedral([last_psi_group]).run(start=start, stop=stop)
    phi = np.hstack((first_phi.angles, R_phi.angles))
    psi = np.hstack((R_psi.angles, last_psi.angles))

    dihedral_angle = np.hstack((phi, psi))
    return dihedral_angle

def calculate_phi_psi_system(trajectory_files, topology):
    dihedral = np.array([])
    u_dummy = mda.Universe(topology)
    num_res = len(u_dummy.select_atoms("protein").residues)
    for xtc in trajectory_files:
        u = mda.Universe(topology, xtc)
        stop = len(u.trajectory)
        start = stop - 50000
        assert start >= 0
        dihedral = np.append(dihedral, calculate_phi_psi(u, start, stop))
    return dihedral.reshape((-1, num_res * 2))

def get_residue_name(topology):
    u_dummy = mda.Universe(topology)
    res_name = list(u_dummy.select_atoms("protein").residues.resnames)
    return res_name

def get_trajectory_files(trajectory_dir):
    traj_files = os.listdir(trajectory_dir) 

    xtc = []

    for file in traj_files:
        if file[-3:] == "gro":
            topology = trajectory_dir + "/" + file
        elif file[0] == ".":
            continue
        else:
            xtc.append(trajectory_dir + "/" + file)

    return topology, xtc

