# Dihedral Pricipal Component Analysis
Created by Tim Ling @ YSL lab

## Dependencies

### Python
* numpy
* matplotlib
* scipy

### MDAnalysis
* MDAnalysis

### scikit-learn
* sklearn

## Reference

	Rodriguez, A., and A. Laio. “Clustering by Fast Search and Find of Density Peaks.” Science, vol. 344, no. 6191, 2014,
		pp.1492–1496., doi:10.1126/science.1242072.

## Protocol Description
	
	This program implements the density peak clustering algorithm.

## Quickstart
	
	To run the program properly, ensure that there are xtc files and a topology file in 
	the structure directories.
	To run the program, simply do (XTC1_dir and XTC2_dir can be provided upon request):

				python dPCA.py --traj1 XTC1_dir/ --traj2 XTC2_dir/

	The program can also take the following flags:
		
		--components: (integer) if greater than 1, it indicates the number of components
					  otherwise it indicates the percent variance explained.

		--timer: (bool) True to time the analysis process.

		--interactive: (bool) True to select the cluster centers interactively.

		--debug: (bool) True to output useful information.

## Known Issues:
	
* The time required to perform this analysis protocol increases significantly with more components.
* The program takes a lot of memory (Solution might be to use sparse matrix from numpy).
* When interactive mode is activated, the point with the highest rho has to be selected as the cluster center.

## Acknowledgments
The implementation is based the previous work from (not ranked):
* Dr. Diana Slough
* Dr. He (Agnes) Huang
* Dr. Jiayuan Miao
* Dr. Hongtao Yu
* Jovan Damjanovic




