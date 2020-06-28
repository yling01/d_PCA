'''
Tim Ling

Last update: 2020.05.18
'''
import numpy as np
from sklearn.decomposition import PCA

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