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
from CalcNIP.py import *
from Cluster.py import *
from MakeDensityMtx.py import *
from MakeProjection.py import *
from Miscellaneous.py import * 

matplotlib.use('TkAgg', warn=False)



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





