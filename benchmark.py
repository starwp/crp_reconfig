#!/usr/bin/python
'''
This software is the product of research performed at the University of Texas at Austin

Author: Wei-Ju Chen, albertwj@cs.utexas.edu
'''

import sys
import copy
import operator
from random import random
from random import randint
from tempfile import TemporaryFile
import numpy as np
import time
import smtplib
from random import shuffle
from itertools import combinations, permutations, combinations_with_replacement
from reconfig import *
import math
from header import _SUCCESS, _CRPFAIL, _DPRFAIL_2, _DPRFAIL
import collections
import matplotlib.pyplot as plt


def print_all(gph):
    gph.print_digraph()
    for ph in gph.physicals:
        for rp in ph.rps:
            print(ph.pid, ph.slice_size, rp.idn, rp.befPeriod, rp.aftPeriod, rp.offset)

def version(A):
    if A:
        print('111')

def test_uniform():
    sch = Schedule()
    count = 0
    prefail = 0
    fail1 = 0
    fail2 = 0
    bef = 9
    aft = 9
    num_of_partitions = 20
    max_budget = 20
    res = [0]*(max_budget+1)
    max_period = 10
    
    for i in range(1000):
        #sample_uniform(number of partitions, \
                # list of periods, befu, aftu, max reconfiguration regularity):
        # befu, aftu: 0 for 0 < total utilization <= 0.1
        # befu, aftu: 9 for 0.9 < total utilization <= 1.0
        # max rr: randint(1, max)
        #sample_non_uniform(self, psize, max_degree_out, size_list, num_crps, \
        #        edge_p, period_list, befu, aftu, max_rc):

        count += 1
        test = sch.sample_uniform(num_of_partitions, [2**x for x in range(1,max_period+1)], bef, aft, 5)
        if i % 10 == 0:
            print('count',i)
        assert(test == _SUCCESS)
        t = np.random.choice(range(0, 2**(max_period)))
        for budget in range(max_budget+1):
            physicals_copy = copy.deepcopy(sch.gph.physicals)
            for ph in physicals_copy:
                # setting the regularity%
                for y, rp in enumerate(ph.rps):
                    if y < num_of_partitions*20//100:
                        rp.rc = randint(2, 5)
                    else:
                        rp.rc = 1
    
                
                
                #Alg1(partitions, time of the reconfig, max length of reconfig, 
                #      , reconfiguration blocking time, environment: UNIFORM)
                [test, schedule1, offsets, size] = Alg1(ph.rps, t, budget, 0, UNIFORM)
                if test != _SUCCESS:
                    fail1 += 1
                    break 
            
            if test == _SUCCESS:
                for j in range(budget, max_budget+1):
                    res[j] += 1
                break
        
    
    print(res)
    #print("Uniform Fail:", float(fail1)/(count), "Utilization:", bef, "->", aft)


#KERNEL = [UNIFORM, OFFONE, LC, NAIVE]
KERNEL = [NAIVE]
def test_non_uniform():
    sch = Schedule()
    count = 1
    max_budget = 1
    fail_crp = [0,0,0,0]
    fail_dpr1 = [0,0,0,0]
    fail_dpr2 = [0,0,0,0]
    
    #irregular_precentage = [0, 20, 40, 60, 80, 100]
    irregular_precentage = [60]
    
    
    fail = [[[0] * (max_budget+1) for i in range(len(irregular_precentage))] for _\
        in range(len(fail_crp))]
    
    
    for i in range(count):
        print(i)
        while sch.sample_non_uniform(5, 3, [2, 4, 8, 16, 32], 5, 1, [2, 4, 8, 16, 32], 8, 8, 5) != _SUCCESS:
            continue
        

        t = np.random.choice(range(0, 64))
        
        
        for irg, irregular_precent in enumerate(irregular_precentage):
            tmp = collections.defaultdict(lambda: True)
            for budget in range(1, max_budget + 1):
                for environment in KERNEL:
                    physicals_copy = copy.deepcopy(sch.gph.physicals)
                    for ph in physicals_copy:
                        
                        for i, rp in enumerate(ph.rps):
                            if i < int(irregular_precent / 100 * len(ph.rps)):
                                rp.rc = randint(2, 5)
                            else:
                                rp.rc = 1
        
                        [test, schedule, offsets, size] = Alg1(ph.rps, t, budget, 0, environment)
                        if test == _CRPFAIL or test == _DPRFAIL or test == _DPRFAIL_2:
                            if budget == 0: 
                                tmp[environment,budget] = False
                            else:
                                tmp[environment,budget] = tmp[environment,budget-1] or False 
                            if not tmp[environment, budget]:
                                fail[environment][irg][budget] += 1
                            break
#                        elif test == _DPRFAIL:
#                            fail_dpr1[environment] += 1
#                            break
#                        elif test == _DPRFAIL_2:
#                            fail_dpr2[environment] += 1
#                            break
                        else:#succes

                            assert validate_rc_regular_non_uniform(ph.rps, t, schedule, budget)
                            # TODO: validate CRP regular
                            #哪一些offset, task 回来
      
        

            
#            for i in KERNEL: 
#                fail[i] = fail_crp[i] + fail_dpr1[i] + fail_dpr2[i] 
#   
                           
    print("CRP-Mul Fail Rate:", 'iregular',  "Best:", fail[UNIFORM],\
          ", NAIVE:", fail[NAIVE])
                

def plot_schedulability(arr1=None, arr2=None, arr3=None, arr4=None):
    
#    arr1 = [float(i) for i in arr1]
#    arr2 = [float(i) for i in arr2]
#    arr3 = [float(i) for i in arr3]
#    arr4 = [float(i) for i in arr4]
#    
#    arr1 = np.asarray(arr1).reshape(-1,)
#    arr2 = np.asarray(arr2).reshape(-1,)
#    arr3 = np.asarray(arr3).reshape(-1,)
#    arr4 = np.asarray(arr4).reshape(-1,)
#  
#    arr1 = list(arr1)
    xx = [0,2,4,8,12,16,20]
#    
    params = {'legend.fontsize': 15,
              'legend.handlelength': 2,
          'figure.figsize': (12, 8),
         'axes.labelsize': 15,
         'axes.titlesize': 15,
         'xtick.labelsize':15,
         'ytick.labelsize':15,
         'font.family':'Times New Roman'}
    plt.rcParams.update(params)
    lw = 2
    res = [395, 411, 422, 430, 432, 436, 437, 442, \
           446, 448, 448, 449, 449, 449, 452, 454, 455, 456, 456, 457, 457]
    res = [416, 423, 433, 441, 445, 447, 449, 454, \
           458, 459, 461, 462, 463, 466, 466, 467, 467, 468, 468, 469, 470]
    res_short = [res[i] for i in xx]
    
    arr1 = list(map(lambda x:x/10, res_short))
    
##########################
    
    fig = plt.figure(figsize=(5, 4))
    plt.plot(xx, arr1,'ok-',markersize=8, label='DPR',markerfacecolor='none',lw=1)        
#    plt.plot(xx, arr2,'sr-', label='CRS',lw=3)
#    plt.plot(xx, arr3,'ob-', label='CR-EDF',lw=2)  
#    plt.plot(xx, arr4,'xg-', label='TS-EDF',lw=2)
    
#    plt.xlim([5, 25])
#    plt.ylim([0.0, 15])
    plt.yticks(np.arange(20, 70, step=10))
    plt.xticks(np.arange(0, 21, step=2))
    plt.xlabel('Budget', fontsize = 17)
        
    plt.ylabel('Schedulability (%)', fontsize = 17)
    plt.legend(loc="upper left",labelspacing = 0,fontsize=15,handletextpad=0.2,frameon=False)
    plt.show()
    fig.savefig('plot_rrp+schedulability.png',dpi=300, bbox_inches = "tight")


#test()
#test_uniform()
#test_non_uniform()
plot_schedulability()