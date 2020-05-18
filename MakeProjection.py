'''
Tim Ling

Last update: 2020.05.18
'''
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.lib.distances import calc_dihedrals

import numpy as np
from sklearn.decomposition import PCA
'''
Arguments: 
    
    u: (mda object) protein trajectory

Returns: 
    
    dihedral_angle: (np array (n observations, m features)) dihedral angles

Does:
    
    Calculates all of the phi psi angles in the protein
'''
def calculate_phi_psi(u, num_trajectories):
    u_protein = u.select_atoms("protein")
    num_res = len(u_protein.residues)
    num_frame = len(u.trajectory)
    num_dihedral = num_res * 2
    ags_phi = [res.phi_selection() for res in u_protein.residues[1:]]
    ags_psi = [res.psi_selection() for res in u_protein.residues[0:-1]]
    R_phi = Dihedral(ags_phi).run()
    R_psi = Dihedral(ags_psi).run()
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

    first_phi = Dihedral([first_phi_group]).run()
    last_psi = Dihedral([last_psi_group]).run()
    phi = np.hstack((first_phi.angles, R_phi.angles))
    psi = np.hstack((R_psi.angles, last_psi.angles))
    phi_t = phi.T
    psi_t = psi.T
    assert len(phi) == len(psi)
    dihedral_angle = np.ones((1, num_frame))
    for index in range(num_res):
        dihedral_angle = np.vstack((dihedral_angle, phi_t[index]))
        dihedral_angle = np.vstack((dihedral_angle, psi_t[index]))
    dihedral_angle = dihedral_angle[1:].T
    dihedral_angle_kept = np.ones((1, num_res * 2))
    
    for j in range(1, num_trajectories + 1):
        start = (np.array(range(1, i * 2, 2))*50000)[j - 1] + j
        end = start + 50000
        
        dihedral_angle_kept = np.vstack((dihedral_angle_kept, dihedral_angle[start:end]))
        
    return dihedral_angle_kept[1:]

'''
Arguments:
    
    dataset: (np array (n observations, m features)) 
    
    variance_explained: (int or float) 
                        when less than one, is the percent variance explained
                        when greater than one, is the number of components

Returns: 

    projection: (np array (n observations, m features))
    
    pca: (pca model) pca model that performs dimensionality reduction

Does:

    Performed dimensionality reduction based on either the number of 
    components or percent variance explained
'''
def dimension_reduction(dataset, variance_explained):
    pca = PCA(n_components=variance_explained)
    projection = pca.fit_transform(dataset)
    return projection, pca

'''
Parameters:
    
    dihedral_trr: (np.array [n observations, m dihedral angles])
                  the matrix that contains all the dihedral angles

Returns:
    
    trig_trr: (np.array [n observations, m dihedral angles])
              the matrix that contains all the sin and cos values of 
              the dihedral angles

Does:
    
    Simply calculates the sin and cos values of the dihedral angles
'''
def convert_dihedral_to_trig(dihedral_trr):
    trig_trr = np.hstack((np.cos(np.deg2rad(dihedral_trr)), np.sin(np.deg2rad(dihedral_trr))))
    for i in range(len(trig_trr)):
        frame_trig = trig_trr[i]
        half_length = int(len(frame_trig) / 2)
        trig_trr[i] = np.dstack((frame_trig[:half_length], frame_trig[half_length:])).flatten()
    return trig_trr