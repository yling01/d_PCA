#!/usr/bin/env python
# coding: utf-8

import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.lib.distances import calc_dihedrals

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import colorsys
from scipy.spatial import distance

import matplotlib.font_manager as ftman
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar as cb
import optparse    
import os
import sys
import matplotlib
import time

from MakeFigure.py import *

matplotlib.use('TkAgg', warn=False)

'''
Arguments: 
    
    u: (mda object) protein trajectory

Returns: 
    
    dihedral_angle: (np array (n observations, m features)) dihedral angles

Does:
    
    Calculates all of the phi psi angles in the protein
'''
def calculate_phi_psi(u):
    num_res = len(u.residues)
    num_frame = len(u.trajectory)
    num_dihedral = num_res * 2
    ags_phi = [res.phi_selection() for res in u.residues[1:]]
    ags_psi = [res.psi_selection() for res in u.residues[0:-1]]
    R_phi = Dihedral(ags_phi).run()
    R_psi = Dihedral(ags_psi).run()
    last_psi_group = u.select_atoms("resid " + str(num_res) +\
                                    " and name N", "resid " +\
                                    str(num_res) +\
                                    " and name CA", "resid " +\
                                    str(num_res) +\
                                    " and name C", "resid 1 and name N")

    first_phi_group = u.select_atoms("resid " +\
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
    return dihedral_angle[1:].T

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
Arguments: 

    s_pop: (np array (n dimension)) population histogram of analysis target
    
    ref_pop: (np array (n dimension)) reference population histogram
    
Returns:

    NIP: (float) NIP score between the analysis target and reference
    
Does:

    Performs NIP calculation
'''
def calc_NIP_helper(s_pop, ref_pop):
    numerator = 0.0
    s_denom = 0.0
    ref_denom = 0.0
    s_pop = s_pop.flatten()
    ref_pop = ref_pop.flatten()
    for i in range(len(s_pop)):
        numerator += (s_pop[i] * ref_pop[i])
        s_denom += (s_pop[i]) ** 2
        ref_denom += (ref_pop[i]) ** 2
    
    NIP = (2 * numerator) / (s_denom + ref_denom)

    return NIP

'''
Arguments:
    
    s1_h: (np array (n dimension)) population histogram of analysis target 1
    
    s2_h: (np array (n dimension)) population histogram of analysis target 2
    
Returns:

    NIP1: NIP score of analysis target 1
    
    NIP2: NIP score of analysis target 2
    
Does:

    Create a reference histogram (the average of s1 and s2) and calculates 
    the two NIP scores.
'''
def calc_NIP(s1_h, s2_h):
    s1_pop = s1_h[0]
    s2_pop = s2_h[0]
    
    ref_pop = (s1_pop + s2_pop) / 2
    return calc_NIP_helper(s1_pop, ref_pop), calc_NIP_helper(s2_pop, ref_pop)

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

#Arguments: index: integer, the current index
#           index_array: python array, the actual index stored in an array
#           binSize: the size of the bin
#Returns: Nothing, the index_array is appended

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


#Arguments: i, j, n: int, the ith, jth index and the number of data points
#Returns: the index of the distance between X[i], X[j] in condensed distance mtx 
def square_to_condensed(i, j, n):
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return int(n*j - j*(j+1)/2 + i - 1 - j)

#Arguments: distance_mtx: np array (n, 1), the condensed distance matrix
#           density: np array (n, 1), the density of all observations
#           distance_cutoff: float
#Returns: rho: np array, the density array (n, 1)
#         rho_order: np array, the array that sorts rho in descending order (n, 1)
#         nearest_neightbor: np array, the array that has the nearest neighbor of heigher density
#         delta: np array, the array that has the distance to the nearest neighbor 
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

#Arguments: event
#Returns: Nothing
def onpick3(event):
    ind = event.ind
    current_color = col.get_facecolors()[ind]
    if current_color[0][0]: #selecting a new point
        col._facecolors[ind,:] = (0, 1, 0, 1) #plots the point green
    else: #deselect a old point
        col._facecolors[ind,:] = (1, 0, 0, 1) #plots the point green

    fig.canvas.draw()
    

#Arguments: density_clean: np array (n, d+1)
#           distance_cutoff_percent: float, default at 0.02
#           delta_cutoff: float, default at 0.5
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

def draw_clustered_decision_graph(rho, delta, cluster_center_index, file_name):
    global dir_name
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


#Arguments: density_cube: np array, returned by combine_density_coor
#           cluster_assignment: np array, returned by DB_cluster
#Returns: population: np array
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
    

#Arguments: projection: np array, returned by dimension_reduction
#           cluster: np array, returned by DB_cluster
#           density_cube: np array, returned by combine_density_coor
#Returns: projection_cluster_assignment: np array, the assignment of cluster of each projection
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

def calcDensity2D (Xs, Ys, Ws=None):
    assert len(Xs) == len(Ys)
    Bins = np.linspace(start=-180, stop=180, num=101)
    density2D, xedges, yedges = np.histogram2d(Xs, Ys, bins=Bins, weights=Ws, density=True)
    xmidps = 0.5 * (xedges[:-1] + xedges[1:])
    ymidps = 0.5 * (yedges[:-1] + yedges[1:])
    return xmidps, ymidps, density2D

def get_cluster_assignment(density_clean, projection, file_name, interactive=False):
    rho, delta, cluster, cluster_center_index, distance_mtx_condensed, distance_cutoff = DB_cluster(density_clean, interactive=interactive)
    halo = calculate_halo(cluster, distance_mtx_condensed, distance_cutoff, rho)
    draw_clustered_decision_graph(rho, delta, cluster_center_index, file_name)
    return assign_projection_cluster(projection, cluster, density_clean)

def ThreeToOne(three_letter_code):
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    return d[three_letter_code]

parser = optparse.OptionParser()
parser.add_option('--traj1', dest = 'traj1',
    default = 'NO_INPUT',   
    help = 'System 1 trajectory in PDB format')

parser.add_option('--traj2', dest = 'traj2',
    default = 'NO_INPUT',
    help = 'System 2 trajectory in PDB format')

parser.add_option('--components', dest = 'n',
    default = '3',
    help = 'integer if greater than 1, indicating the number of components\
            float if less than 1, indicating the percent variance explained')

parser.add_option('--timer', dest = 'time_procedure',
    default = 0,
    help = 'True to time to analysis procedure')

parser.add_option('--interactive', dest = 'interactive',
    default = 0,
    help = 'True to select cluster centers')

parser.add_option('--debug', dest = 'debug', 
    default = 0,
    help = 'True to print out useful information')


(options,args) = parser.parse_args()

traj1 = options.traj1
traj2 = options.traj2
n = float(options.n)
time_procedure = bool(options.time_procedure)
interactive = bool(options.interactive)
debug = bool(options.debug)

if n >= 1:
    n = int(n)

if 'NO_INPUT' in [traj1, traj2]:
    sys.exit("Input trajectory not specified!\n")

print("Reading trajectories...")
u1 = mda.Universe(traj1)
u2 = mda.Universe(traj2)

res_name = list(u1.residues.resnames)

global dir_name 
dir_name = ''
for res in res_name:
    dir_name += ThreeToOne(res)

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

u1_num_frame = len(u1.trajectory)
u2_num_frame = len(u2.trajectory)

assert u1_num_frame == u2_num_frame

if time_procedure:
    start = time.perf_counter()


print("Getting dihedrals...")
dihedral_angle1 = calculate_phi_psi(u1)
dihedral_angle2 = calculate_phi_psi(u2)
trig_trr1 = convert_dihedral_to_trig(dihedral_angle1)
trig_trr2 = convert_dihedral_to_trig(dihedral_angle2)
trig_trr_ttl = np.vstack((trig_trr1, trig_trr2))
u1_num_frame = len(u1.trajectory)
u2_num_frame = len(u2.trajectory)


print("Writing projection...")
projection, pca = dimension_reduction(trig_trr_ttl, n)
projection1 = projection[:u1_num_frame]
projection2 = projection[u2_num_frame:]
h1, h2 = create_histogram(projection1, projection2)
s1_density, s1_density_clean = combine_density_coor(h1, 0.1000)
s2_density, s2_density_clean = combine_density_coor(h2, 0.1000)

print("Performing cluster analysis...")
projection_cluster_assignment1 = get_cluster_assignment(s1_density_clean, projection1, "s1_decision_graph.png", interactive)
projection_cluster_assignment2 = get_cluster_assignment(s2_density_clean, projection2, "s2_decision_graph.png", interactive)

clusters1, outliers1 = get_top_clusters(5, dihedral_angle1, projection_cluster_assignment1)
clusters2, outliers2 = get_top_clusters(5, dihedral_angle2, projection_cluster_assignment2)

projection1_clean = projection1[np.argwhere(projection_cluster_assignment1 != 0).flatten()]
projection2_clean = projection2[np.argwhere(projection_cluster_assignment2 != 0).flatten()]

print("Calculating NIP scores...")
h1_clean, h2_clean = create_histogram(projection1_clean, projection2_clean)
NIP_clean = calc_NIP(h1_clean, h2_clean)
NIP_ttl = calc_NIP(h1, h2)

print("Plotting...")
MakeFigure(clusters1, res_name, u1_num_frame, NIP_ttl[0], NIP_clean[0], "s1_rama.png")
MakeFigure(clusters2, res_name, u2_num_frame, NIP_ttl[1], NIP_clean[1], "s2_rama.png")
plt.close('all')

if time_procedure:
    end = time.perf_counter()
    print("=" * 80)
    print(f"This analysis procedure finishes in {end - start:0.4f} seconds")

if debug:
    covar = pca.get_covariance() 
    np.savetxt("covar.dat", covar, fmt="%10.6f")
    np.savetxt("dihedral.trr", trig_trr_ttl, fmt="%10.6f")
    np.savetxt("projection.txt", projection, fmt="%10.6f")
    np.savetxt("s1_projection.txt", projection1, fmt="%10.6f")
    np.savetxt("s2_projection.txt", projection2, fmt="%10.6f")
    np.savetxt("s1_density.txt", s1_density, fmt="%10.5f")
    np.savetxt("s2_density.txt", s2_density, fmt="%10.5f")
    np.savetxt("s1_density_kept.txt", s1_density_clean, fmt="%10.5f")
    np.savetxt("s2_density_kept.txt", s2_density_clean, fmt="%10.5f")
    np.savetxt("s1_phi_psi.xvg", dihedral_angle1, fmt="%10.5f")
    np.savetxt("s2_phi_psi.xvg", dihedral_angle2, fmt="%10.5f")

    s1_distance_mtx = calculate_distance_matrix(s1_density_clean)
    s2_distance_mtx = calculate_distance_matrix(s2_density_clean)
    np.savetxt("s1_distance.dmtx", s1_distance_mtx, fmt="%5d%5d%10.5f%10.5f%10.5f")
    np.savetxt("s2_distance.dmtx", s2_distance_mtx, fmt="%10.5f")

    np.savetxt("cluster.txt", np.hstack((np.arange(1, len(projection1) + 1).reshape((-1,1)), projection1, projection_cluster_assignment.reshape((-1,1)))), fmt="%5d%10.5f%10.5f%10.5f%5d")





