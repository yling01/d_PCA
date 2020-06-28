'''
Tim Ling

Last update: 2020.05.18
'''
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
'''
Parameters:

    h: (np.array) histogram(h[0]) and edges(h[1])

    cutoff: (float) the cell whose density below this threshold will be disregarded

Returns:

    density_coor_array: (n observations, m dimensions + 1) the array that 
                        has density written next to the coordinate

    density_coor_array_clean: (n observations, m dimensions + 1) the array that 
                              only has cells whose density is above the cutoff
'''

def combine_density_coor(h, cutoff):
    h_flatten = h[0].flatten()
    edges = h[1]
    dimension = len(edges)
    binSize = len(edges[0]) - 1
    coor_array = np.array([])
    density_coor_array = np.array([])
    
    for edge in edges:
        coor_array = np.append(coor_array, (edge[1:] + edge[:-1]) / 2)
    coor_array = coor_array.reshape((dimension, binSize))
    
    for index, density in enumerate(h_flatten):
        if density == 0.0:
            continue
        index_array = np.zeros(shape=dimension, dtype=int)
        index_array_dummy = []
        convert_index_to_coordinate(index, index_array_dummy, binSize)
        index_array_dummy = np.trim_zeros(np.array(index_array_dummy), 'f')
        index_array[dimension - len(index_array_dummy):] = index_array_dummy
        density_coor_array = np.append(density_coor_array, np.concatenate((get_location(coor_array, index_array), [density])))
    num_col = int(dimension + 1)
    density_coor_array = density_coor_array.reshape((-1, num_col))
    density_coor_array_clean = density_coor_array[np.around(density_coor_array[:,-1],16) > cutoff]
    return density_coor_array, density_coor_array_clean

'''
Arguments:
    
    projection1: (np array (n observations, m dimensions)) 
                 projection of population data on the eigen vector
    
    projection2: (np array (n observations, m dimensions))
                 projection of population data on the eigen vector
                 
    BINS: (int) number of bins to have on the histogram (default at 50)
    
Returns:

    s1_h: (np array, np array) tuple composed of the histrogram and the edges
    
    s2_h: (np array, np array) tuple composed of the histrogram and the edges

Does:

    Createes the histogram based on the two projections
'''
def create_histogram(projection1, projection2, BINS = 50):
    
    BINS = 50
    projection_reference = np.vstack((projection1, projection2))
    
    ref_h = np.histogramdd(projection_reference, bins=BINS, density=True)
    
    BINS_array = ref_h[1]
    
    s1_h = np.histogramdd(projection1, bins=BINS_array, density=True)
    s2_h = np.histogramdd(projection2, bins=BINS_array, density=True)
    
    return s1_h, s2_h

'''
Parameters:
    
    index: (int) the current index in the density array
    
    index_array (list) the actual index in the edge array
    
    binSize: (int) the size of the bin
    
Returns:
    
    None
    
Does:
    
    Finds the index of a cell in the edge list based on the 
    index of the cell in the density list

'''
def convert_index_to_coordinate(index, index_array, binSize):
    if index > 1:
        convert_index_to_coordinate(index // binSize, index_array, binSize)
    index_array.append(int(index % binSize))

'''
Arguments:
    
    edges: (np array) edges of the histogram

    index_array: (list) the location of the cell
                 in the histogram

Returns:

    location: (np array) the location of the element in the histogram

Does:

    Returns the location of the element in the histogram
'''
def get_location(edges, index_array):
    location = np.array([])
    for index, edge in enumerate(edges):
        location = np.append(location, edge[index_array[index]])
    return location 

'''
Parameters:

    density_coor: (np.array) the array that has density written next to the coordinates

Returns:

    distance_mtx: (np.array) the distance matrix in the form of 
                  index i, index j, distance, density i, density j

Does:

    Construct the distance matrix
'''
def calculate_distance_matrix(density_coor):
    distance_mtx = np.array([])
    
    for i in range(len(density_coor) - 1):
        for j in range(i + 1, len(density_coor)):
            distance = np.sqrt(np.sum((density_coor[i][:-1] - density_coor[j][:-1])**2))
            distance_mtx = np.concatenate((distance_mtx, [i+1, j+1, distance, density_coor[i][-1], density_coor[j][-1]]))
    distance_mtx = distance_mtx.reshape((-1, 5))
    return distance_mtx