'''
Tim Ling

Last update: 2020.05.18
'''

import numpy as np
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