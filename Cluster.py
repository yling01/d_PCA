'''
Tim Ling

Last update: 2020.05.18
'''
import numpy as np
from scipy.spatial.distance import pdist
import math
import matplotlib.pyplot as plt
import colorsys
from scipy.spatial import distance
'''
Parameters:

    projection: (np.array [n observations, m dimensions]) projection of data points

    radius: (float) cutoff for two data points to be neighbors

    min_sample_number: (int) the minimum number of data points that a cluster has to have

Returns:

    n_cluster_: (np.array) the cluster information of all the data points

    n_noise_: (np.array) the noise information of all the data points

Does:

    Using sklearn package to do cluster analysis

Note: 

    This function is currently not working, it needs to be tuned further for 
    proper function.
'''
def cluster(projection, radius=0.5, min_sample_number=100):
    db = DBSCAN(eps=radius, min_samples=min_sample_number).fit(projection)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    return n_clusters_, n_noise_


'''
Parameters:
    
    num_colors: (int) the number of color to return

Returns:
    
    colors: (np.array [n, 3]) the rgb values of colors

Does:
    
    Randomly generate n color rgb's

Note:

    This should be replaced with a more stable function where
    colors should be as distinct as 'possible'.

'''
def get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

'''
Parameters:
    
    i: (int) the ith index 

    j: (int) the jth index

    n: (int) the number of data points

Returns:
    
    index: (int) the index of the distance in the compact distance matrix

Does:
    
    Finds the index of the distance between data i and j in the 
    compact distance matrix

'''
def square_to_condensed(i, j, n):
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return int(n*j - j*(j+1)/2 + i - 1 - j)


'''
Parameters: 
    
    distance_mtx: (np.array) the compact distance mtx of data points

    density: (np.array) the density of all cells

    distance_cutoff: (float) distance_cutoff to calculate rho

Returns:
    
    rho: (np.array) the density of the data points 

    rho_order: (np.array) the array to sort rho

    nearest_neighbor: (np.array) the nearest neighbor for all data points

    delta: (np.array) the shortest distance to the point 
           with a higher rho value

Does:

    Calculates rho, delta, rho_order and nearest_neighbor

Notes:

    The point with the highest density has to be selected as the 
    cluster center because it does not have a nearest neighbor.
    It is a known issue that the program will fail if this point is 
    not selected as the cluster center.

Reference:
    
    Rodriguez, A., and A. Laio. “Clustering by Fast Search and Find of Density
        Peaks.” Science, vol. 344, no. 6191, 2014, pp.1492–1496., doi:10.1126/science.1242072.
'''
def calculate_rho_delta(distance_mtx, density, distance_cutoff):
    max_distance = np.amax(distance_mtx)
    rho = np.copy(density)
    num_datapoint = len(density)
    delta = np.full(num_datapoint,max_distance)
    nearest_neighbor = np.ones(num_datapoint, dtype=int) * -1
    for i in range(num_datapoint-1):
        for j in range(i + 1, num_datapoint):
            index_distance_mtx = square_to_condensed(i, j, num_datapoint)
            ij_distance = distance_mtx[index_distance_mtx]
            exponent = ij_distance / distance_cutoff
            adder = density[j] * math.exp(-exponent**2)
            rho[i] = rho[i] + adder
            rho[j] = rho[j] + adder

    rho_order = np.flip(np.argsort(rho))
    rho_sorted = rho[rho_order]
    delta[rho_order[0]] = -1.0
    for i in range(1, num_datapoint):
        for j in range(0, i):
            rho_orderI = rho_order[i]
            rho_orderJ = rho_order[j]
            index_distance_mtx = square_to_condensed(rho_orderI, rho_orderJ, num_datapoint)
            ij_distance = distance_mtx[index_distance_mtx]
            if ij_distance < delta[rho_orderI]:
                delta[rho_orderI] = ij_distance
                nearest_neighbor[rho_orderI] = rho_orderJ
    nearest_neighbor[rho_order[0]] = 0
    delta[rho_order[0]] = np.amax(delta)
    assert(not np.any(nearest_neighbor == -1))
    return rho, rho_order, nearest_neighbor, delta

'''
Parameters:
    
    event: a click event

Returns:
    
    None

Does:
    
    Enables the customer selection of cluster center.
    The cluster centers are labelled green.
'''
def onpick3(event):
    ind = event.ind
    current_color = col.get_facecolors()[ind]
    if current_color[0][0]: #selecting a new point
        col._facecolors[ind,:] = (0, 1, 0, 1) #plots the point green
    else: #deselect a old point
        col._facecolors[ind,:] = (1, 0, 0, 1) #plots the point green

    fig.canvas.draw()
    

'''
Parameters:
    
    density_clean: (np.array) the density mtx 

    distance_cutoff_percent: (float) the percent of data points to drop 

    delta_cutoff: (float) in automated cluster mode, the cutoff for delta

    interactive: (bool) True to select cluster center in the interactive mode

Returns:
    
    rho: (np.array) the density of the data points 

    delta: (np.array) the shortest distance to the point 
           with a higher rho value

    cluster_center_index: (list) stores the cluster center index
    
    distance_mtx_condensed: (np.array) compact distance list

    distance_cutoff: (float) distance_cutoff to calculate rho
    
    
Does:
    
    Implementation of the density peak based clustering algorithm

Reference:
    
    Rodriguez, A., and A. Laio. “Clustering by Fast Search and Find of Density
        Peaks.” Science, vol. 344, no. 6191, 2014, pp.1492–1496., doi:10.1126/science.1242072.
'''

def DB_cluster(density_clean, distance_cutoff_percent=0.02, delta_cutoff=0.5, interactive=False):
    distance_mtx_condensed = pdist(density_clean[:,0:-1])
    density = density_clean[:,-1]
    cluster_center_index = []
    num_datapoint = len(density)
    cluster = np.full(num_datapoint, -1)
    num_cluster = 0

    distance_cutoff_index = math.ceil(distance_cutoff_percent * len(distance_mtx_condensed))
    distance_cutoff = np.sort(distance_mtx_condensed)[distance_cutoff_index]    
    rho, rho_order, nearest_neighbor, delta = calculate_rho_delta(distance_mtx_condensed, density, distance_cutoff)

    if interactive:
        global fig, axis, col 
        
        fig, axis = plt.subplots(dpi=200)
        mask = delta > delta_cutoff

        color = np.array([1, 0, 0, 1] * num_datapoint).reshape(-1, 4) #original poitns: all red

        for index, decider in enumerate(mask):
            if decider:
                color[index] = [0, 1, 0, 1] #color those above threshold gree
            
        
        col = axis.scatter(rho, delta, c=color, marker='.', picker=True)

        axis.set_title("Decision Graph", fontsize='xx-large')
        axis.set_ylabel(r"$\delta$", fontsize='x-large')
        axis.set_xlabel(r"$\rho$", fontsize='x-large')
        
        fig.canvas.mpl_connect('pick_event', onpick3)
        
        plt.show()

        for index, point_color in enumerate(col.get_facecolors()):
            point_color = point_color.flatten()
            if not point_color[0]: #if green, meaning selected
                num_cluster += 1
                cluster[index] = num_cluster
                cluster_center_index.append(index)
        plt.close('all')

    else:
        for i in range(num_datapoint):
            if delta[i] >= delta_cutoff:
                num_cluster += 1
                cluster[i] = num_cluster
                cluster_center_index.append(i)
            
    for i in range(num_datapoint):
        index = rho_order[i]
        if cluster[index] == -1:
            cluster[index] = cluster[nearest_neighbor[index]]
            
    assert(not np.any(cluster == -1))
    
    return rho, delta, cluster, cluster_center_index, distance_mtx_condensed, distance_cutoff


'''
Parameters:
    
    cluster: (np.array) the cluster assignment of the data points

    distance_mtx_condensed: (np.array) compact distance list

    rho: (np.array) density of the data points on the decision graph

    distance_cutoff: (float) distance_cutoff to calculate rho

Returns:
    
    halo: (np.array) the halo of the data points 
    
Does:
    
    Calculates the halo of the data points
    
Reference:
    
    Rodriguez, A., and A. Laio. “Clustering by Fast Search and Find of Density
        Peaks.” Science, vol. 344, no. 6191, 2014, pp.1492–1496., doi:10.1126/science.1242072.
'''

def calculate_halo(cluster, distance_mtx_condensed, distance_cutoff, rho):
    num_cluster = len(np.unique(cluster))
    num_datapoint = len(cluster)
    halo = np.copy(cluster)
    
    if num_cluster > 1:
        bord_rho = np.zeros(num_cluster)
        for i in range(num_datapoint - 1):
            for j in range(i + 1, num_datapoint):
                index_distance_mtx = square_to_condensed(i, j, num_datapoint)
                if (cluster[i] != cluster[j]) and (distance_mtx_condensed[index_distance_mtx] < distance_cutoff):
                    rho_ave = 0.5 * (rho[i] + rho[j])
                    if rho_ave > bord_rho[cluster[i] - 1]:
                        bord_rho[cluster[i] - 1] = rho_ave
                    if rho_ave > bord_rho[cluster[j] - 1]:
                        bord_rho[cluster[j] - 1] = rho_ave
        for i in range(num_datapoint):
            if rho[i] < bord_rho[cluster[i] - 1]:
                halo[i] = 0
    for i in range(num_cluster):
        nc = 0
        nh = 0
        for j in range(num_datapoint):
            if cluster[j] == i:
                nc += 1
            if halo[j] == i:
                nh += 1
    return halo 

'''
Parameters:
    
    rho: (np.array) the density of the data points 

    delta: (np.array) the shortest distance to the point 
           with a higher rho value

    cluster_center_index: (list) stores the cluster center index

    file_name: (str) the file name of the decision graph

    dir_name: (str) the directory name to store the file

Returns:
        
    None
    
Does:
    
    Draws the decision grpah

'''
def draw_clustered_decision_graph(rho, delta, cluster_center_index, file_name, dir_name):
    file_name = dir_name + '/' +file_name
    assert (len(rho) == len(delta))
    fig, axis = plt.subplots(dpi=300)
    num_datapoint = len(rho)
    
    color = get_colors(len(cluster_center_index))
    
    mask = np.isin(np.arange(num_datapoint), cluster_center_index)
    
    axis.scatter(rho[~mask], delta[~mask], c="black", marker='.')
    axis.scatter(rho[mask], delta[mask], c=color, marker='.')

    axis.set_title("Clustered Decision Graph", fontsize='xx-large')
    axis.set_ylabel(r"$\delta$", fontsize='x-large')
    axis.set_xlabel(r"$\rho$", fontsize='x-large')
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(file_name, bbox_inches='tight')   

'''
Parameters:
    
    density_cube: (np.array) the density mtx along with the axis

    cluster_assignment: (np.array) the cluster assignment of the data points

Returns:

    population: (np.array) the population for all clusters
    
    
Does:
    
    Calculates the population for all clusters
'''
def calculate_population(density_cube, cluster_assignment):
    assert len(density_cube) == len(cluster_assignment)
    num_cluster = np.amax(cluster_assignment)
    num_dimension = density_cube.shape[1] - 1
    volume = 1.0
    cluster_population = np.zeros(num_cluster)
    for i in range(len(cluster_assignment)):
        c = cluster_assignment[i] - 1
        cluster_population[c] += density_cube[i][-1]
        
    for i in range(num_dimension):
        measurement = np.unique(density_cube[:,i])
        volume = volume * (measurement[1] - measurement[0])
        
    return cluster_population * volume 
    
'''
Parameters:
    
    projection: (np.array) projection of the data points

    cluster: (np.array) cluster assignment of the data points

    density_cube: (np.array) the density matrix along with the axis

Returns:
    
    projection_cluster_assignment: (np.array) the assignment of cluster of each projection
    
Does:
    
    Puts the data points into the clusters
    
Reference:
    
    Rodriguez, A., and A. Laio. “Clustering by Fast Search and Find of Density
        Peaks.” Science, vol. 344, no. 6191, 2014, pp.1492–1496., doi:10.1126/science.1242072.
'''
def assign_projection_cluster(projection, cluster, density_cube):
    assert len(cluster) == len(density_cube)
    cube_matrix = density_cube[:,0:-1]
    dimension = cube_matrix.shape[1]
    projection_cluster_assignment = np.zeros(len(projection), dtype=int)
    for projection_index, data in enumerate(projection):
        in_range = True
        min_dist_index = np.argmin(distance.cdist([data], cube_matrix, 'euclidean')[0])
        for i in range(dimension):
            measurement = np.unique(cube_matrix[:,i])
            if round(abs(data[i] - cube_matrix[min_dist_index][i]), 16) >= round(0.5 * (measurement[1] - measurement[0]), 16):
                in_range = False
                break
        if in_range:
            projection_cluster_assignment[projection_index] = cluster[min_dist_index]
                
    return projection_cluster_assignment

'''
Parameters:
    
    n_clusters: (int) the number of clusters to get

    dihedral: (np.array) the dihedral angles of the residues

    cluster_assignment: (np.array) cluster assignment of all data points

Returns:
    
    dihedral_clusters: (np.array) the original dihedral angles in the cluster

    zero_cluster_dihedral: (np.array) the dihedral angles that are not in any cluter
    
Does:
    
    Obtain the clusters on the original dihedral angles
'''

def get_top_clusters(n_clusters, dihedral, cluster_assignment):
    zero_cluster_indices = np.argwhere(cluster_assignment==0)
    zero_cluster_dihedral = dihedral[zero_cluster_indices]
    
    cluster_assignment = np.delete(cluster_assignment, zero_cluster_indices)
    dihedral = np.delete(dihedral, zero_cluster_indices, axis=0)
    
    clusters, cluster_point_count = np.unique(cluster_assignment, return_counts=True)
    
    assert n_clusters <= len(clusters)
    assert len(dihedral) == len(cluster_assignment)
    clusters_sorted = clusters[np.flip(np.argsort(cluster_point_count))]
    dihedral_clusters = []
    for i in range(n_clusters):
        cluster = clusters_sorted[i]
        point_indices = np.argwhere(cluster_assignment == cluster)
        dihedral_cluster = dihedral[point_indices.flatten()]
        dihedral_clusters.append(dihedral_cluster)
    return dihedral_clusters, zero_cluster_dihedral

'''
Parameters:
    
    density_clean: (np.array) the clean density mtx 

    projection: (np.array) projection of the data points

    file_name: (str) the file name of the decision graph

    dir_name: (str) the directory name to store the file

Returns:
    
    projection_cluster_assignment: (np.array) the assignment of cluster of each projection
    
Does:
    
    A wrapper that calls multiple functions to get the clustering results

'''
def get_cluster_assignment(density_clean, projection, file_name, interactive, dir_name):
    rho, delta, cluster, cluster_center_index, distance_mtx_condensed, distance_cutoff = DB_cluster(density_clean, interactive=interactive)
    halo = calculate_halo(cluster, distance_mtx_condensed, distance_cutoff, rho)
    draw_clustered_decision_graph(rho, delta, cluster_center_index, file_name, dir_name)
    return assign_projection_cluster(projection, cluster, density_clean)