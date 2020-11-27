# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:28:23 2019

This software is the product of a joint research performed at the University of
Texas at Austin and University of Connecticut

Author: Wei-Ju Chen, albertwj@cs.utexas.edu; Peng Wu

"""
import sys
import copy
import operator
#from gurobipy import *
from random import random
from random import randint
import json
#import xlwt
from tempfile import TemporaryFile
import numpy as np
import time
import smtplib
from random import shuffle
from itertools import combinations, permutations, combinations_with_replacement
from libCRP import *
import math

class Schedule():
    def __init__(self):
        M = []
        self.period_combos = None
        # psize: num of physical resource
        # size_list: physical resource size
        # num_crps: num of applicaton
        # edge_p: probablity of selecting this edge
        # max_rc: regurlarity
    def sample_non_uniform(self, psize, max_degree_out, size_list, num_crps, \
            edge_p, period_list, befu, aftu, max_rc):
        self.gph = Graph()
        self.gph.generate_digraph(psize, max_degree_out)
        self.gph.sample_crps(num_crps, edge_p)
        self.gph.configure_physicals(size_list)
        return self.gph.configure_rps(period_list, befu, aftu, max_rc)
    
    def sample_uniform(self, num_ps, period_list, befu, aftu, max_rc):
        # single physical resource
        self.gph = Graph()
        self.gph.physicals.append(PhysicalResource(0))
        for i in range(1, num_ps + 1):
            self.gph.crps.append(Crp(i))
            rp = Partition(i, 0, 1, 1, 0, 0)
            self.gph.physicals[0].rps.append(rp)
            self.gph.crps[i-1].rps.append(rp)

        self.gph.configure_physicals([1])
        if self.period_combos is None:
            self.period_combos = self.period_combo(period_list, num_ps)
        return self.gph.configure_rps(period_list, befu, aftu, max_rc, self.period_combos)

        # create a list of period lists where
    # period_list[0] is the list for 0 ~ 0.1
    # period_list[1] is the list for 0.1 ~ 0.2
    # period_list[9] is the list for 0.9 ~ 1.0
    # len(period_list[i]) = targetNumber 
    def period_combo(self, list_all_period, target_len):
        assert(list_all_period and len(list_all_period) >= 4)
        assert(target_len >= 4)
        lst = list(combinations_with_replacement(list_all_period, target_len))
        cleaned = [tuple(n for n in sublist if n != 0) for sublist in lst]
        period_set = [set([]) for i in range(0, 10)]
    
        for i, e in enumerate(cleaned):
            u = 0
            assert(len(e) == target_len)
            for j in e:
                u += 1.0/j
            for k in range(0, 10):
                if u > k * 0.1 and u <= 0.1 + k * 0.1:
                    period_set[k].add(e)
                    break
    
        period_list = []
        for i, l in enumerate(period_set):
            if not l and i >= 1:
                assert(l) # shall not be empty
            period_list.append(list(l))
    
        shuffle_list = []
        for l in period_list:
            line = []
            for item in l:
                tmp = list(item)
                shuffle(tmp)
                line.append(tmp)
            shuffle_list.append(line)
    
        return shuffle_list
    
    def Init_Combo(self, L0, L1, rc):
        M = []
        sumu0 = 0
        sumu1 = 0
        for i, l in enumerate(L0):
            sumu0 += 1.0/l

            #def __init__(self, idn, offset, befPeriod, aftPeriod, rc, rcflag):
            M.append(Partition(i + 1, 0, l, l, rc, True))

        pmax = 0
        for p in M:
            pmax = max(pmax, p.befPeriod)

        m = [0] * int(pmax)
        queue = sorted(copy.deepcopy(M), key = lambda Paritition: (Paritition.befPeriod))
        
        u_temp = 0
        while len(queue) != 0:
            p = queue.pop(0)
            #print('pop: %d deadline is %d' % (p.idn, p.e))
            u_temp += 1/p.befPeriod
#            print(u_temp)
            p.offset = find_node(p.befPeriod, p.befPeriod, m, pmax, p.idn)
#            print(p.offset)
            M[p.idn-1].offset = p.offset

        for i, l in enumerate(L1):
            sumu1 += 1/l
            if i + 1 > len(M) :
                M.append(Partition(i + 1, 0, 128, l, rc, False)) #randint(1, rc)
            else:
                M[i].aftPeriod = l
                M[i].aftfactor = float(1.0/l)
                M[i].rcflag = True

        while(1):
            if len(L1) < len (L0) and  len(L1) < len(M):
                M.pop()
            else:
                break

        info = [[], [] ,[], [], [], []]
        for i, p in enumerate(M):
            info[0].append(int(p.idn))
            info[1].append(int(p.offset))
            info[2].append(int(p.befPeriod))
            info[3].append(int(p.aftPeriod))
            info[4].append(int(p.rc))
            info[5].append(p.rcflag)

        return M, sumu0, sumu1, info
    
    
    def Period_Combo_DP(self, targetNumber, taskNumber, utilization, maxPeriod, Dataset):
        
        
        if taskNumber > targetNumber:
            return Dataset
        
        if utilization > 1:
            return Dataset[:-1]
            
        
        periodList  = [2, 4, 8, 16, 32, 64, 128]
        
        remaining_utilization = utilization
        
        availblePeriods = [i for i in periodList if 1/i <= remaining_utilization]
        
        
        for i in availblePeriods:
           
           print(Dataset+[i])
           print(remaining_utilization - 1/i)

           self.Init_Combo(targetNumber, taskNumber + 1, remaining_utilization - 1/i, maxPeriod, Dataset+[i])
           
        
    
    def Init0(self, taskNumber, utilization, maxPeriod):

        remainingUti = utilization
#
#        if remainingUti < 0.3:
#            periodList  = [4, 8, 16, 32]
#        else:
#            periodList  = [2, 4, 8, 16, 32]
            
        if remainingUti < 0.3:
            periodList  = [4, 8, 16, 32, 64, 128]
        else:
            periodList  = [2, 4, 8, 16, 32, 64, 128]    

#        if remainingUti < 0.3:
#            periodList  = [4, 8, 16, 32, 64, 128, 256]
#        else:
#            periodList  = [2, 4, 8, 16, 32, 64, 128, 256]


#        if remainingUti < 0.3:
#            periodList  = [8, 16,32, 64, 128, 256]
#        else:
#            periodList  = [4, 8, 16,32, 64, 128, 256]

        sumUti = 0
        count = 0
        L = []
        while(1):

            p = np.random.choice(periodList, 1, replace=True)[0]
            count += 1

            if len(L) < taskNumber/2:
                if sumUti + 1/p > 1:
                    break
                if 1/p <= remainingUti/2.0  :

                    L.append(p)
                    remainingUti = remainingUti - 1/p
                    sumUti += 1/p
                    count = 0
                else:
                    break
            else:
                if sumUti + 1/p > 1:
                    if count > 10:
                        break
                    else:
                        continue
                if 1/p <= remainingUti:
                    L.append(p)
                    remainingUti = remainingUti - 1/p
                    sumUti += 1/p
                    count = 0
                else:
                    break

            if len(L) >= taskNumber:
                break

        if remainingUti ==0:
            return L, sumUti
        possiblePeriod = 1/(remainingUti)

        last = 0
        for x in periodList:
           if x >= possiblePeriod:
               last = x
               L.append(last)
               sumUti += 1/last
               break


        if utilization-sumUti > 0.10:
            possiblePeriod = 1/(utilization-sumUti)
            for x in periodList:
                if x >= possiblePeriod:
                   last = x
                   L.append(last)
                   sumUti += 1/last
                   break

        countN = 0
        lastList  = [32, 64, 128]  
        
        while(len(L) < taskNumber):
           p = np.random.choice(lastList, 1, replace=True)[0]
           
           
           if sumUti + 1/p > utilization + 0.05:
               print(sumUti)
               break
           sumUti += 1/p
           L.append(p)
           
           countN += 1
#           if countN > 2:
#               break
           
        
        return L, sumUti


    def Init1(self, taskNumber1, taskNumber2, beforeUti, afterUti, maxPeriod, rc):


        M = []

        while(1):
            [L0, u0] = self.Init0(taskNumber1, beforeUti, maxPeriod)
            [L1, u1] = self.Init0(taskNumber2, afterUti, maxPeriod)
            if u0 < 1 and u1 < 1:
                break


        sumu0 = 0
        sumu1 = 0
        for i, l in enumerate(L0):
            sumu0 += 1/l
            if i == 0 or i == 1:
                M.append(Partition(i + 1, 0, l, l, rc, True))
            else:
                M.append(Partition(i + 1, 0, l, l, rc, True))
#            print('i', i+1, 'l', l)
#            np.random.choice([1], 1, replace=False)[0]

#        print(len(L0), len(L1)) 
        for i, l in enumerate(L1):
            sumu1 += 1/l
            if i + 1 > len(M) :
                M.append(Partition(i + 1, 0, maxPeriod, l, rc, False)) #randint(1, rc)
            else:
                M[i].aftfactor = float(1.0/l)
                M[i].aftPeriod = l
#                if M[i].aftPeriod != M[i].befPeriod:
                M[i].rcflag = True



        while(1):
            if len(L1) < len (L0) and  len(L1) < len(M):
                M.pop()
            else:
                break

#        print(len(L0), len(L1))
#        print(sumu0, sumu1)
        pmax = 0
        for p in M:
            pmax = max (pmax, p.befPeriod)


        deleteList = []
        m = [0] * int(pmax)
        queue = sorted(copy.deepcopy(M), key = lambda Paritition: (Paritition.befPeriod))
        
        u_temp = 0
        while len(queue) != 0:
            p = queue.pop(0)
            #print('pop: %d deadline is %d' % (p.idn, p.e))
            u_temp += 1/p.befPeriod
            print(u_temp)
            
            p.offset = find_node(p.befPeriod, p.befPeriod, m, pmax, p.idn)
#            print(p.offset)

            M[p.idn-1].offset = p.offset




        info = [[], [] ,[], [], [], []]
        M2 = []
        for i, p in enumerate(M):
#            info[0].append(p.idn)
#            info[1].append(p.offset)
#            info[2].append(p.befPeriod)
#            info[3].append(p.aftPeriod)
#            info[4].append(p.rc)
#            info[5].append(p.rcflag)
            if p.offset != None:
                M2.append(Partition(p.idn, p.offset, p.befPeriod, p.aftPeriod, p.rc, p.rcflag))
#            M2.append(p)
            info[0].append(int(p.idn))
            info[1].append(int(p.offset))
            info[2].append(int(p.befPeriod))
            info[3].append(int(p.aftPeriod))
            info[4].append(int(p.rc))
            info[5].append(p.rcflag)

#        for p in M2:
#            print('idnzzzzz', p.idn, 'ffset', p.offset, 'p.befPeriod',p.befPeriod)

        return M2, sumu0, sumu1, info





    def Init2(self, taskNumber, M, afterUti, maxPeriod, offsetList, rc):
        while(1):
#            [L0, u0] = self.Init0(taskNumber, beforeUti, maxPeriod)
            [L1, u1] = self.Init0(taskNumber, afterUti, maxPeriod)
            if  u1 < 1:
                break

        Length = len(M)
        sumu0 = 0
        sumu1 = 0
        for p in M:
            p.befPeriod = p.aftPeriod
            p.beffactor = p.aftfactor
            sumu0 += 1/p.aftPeriod
            p.rcflag = False
#            print('i', i+1, 'l', l)
#            np.random.choice([1], 1, replace=False)[0]

        # False : New
        # oringal : True
        
        for i, l in enumerate(L1):
            sumu1 += 1/l
            if i + 1 > len(M) :
                M.append(Partition(i + 1, 0, maxPeriod, l, rc, True))#np.random.choice([2], 1, replace=False)[0]
            else:
                M[i].aftPeriod = l
                if M[i].aftPeriod != M[i].befPeriod:
                    M[i].rcflag = True



        while(1):
            if len(L1) < Length and  len(L1) < len(M):
                M.pop()
            else:
                break





        info = [[], [] ,[], [], [], []]
        M2 = []
        for i, p in enumerate(M):
#            info[0].append(p.idn)
#            info[1].append(p.offset)
#            info[2].append(p.befPeriod)
#            info[3].append(p.aftPeriod)
#            info[4].append(p.rc)
#            info[5].append(p.rcflag)

            if p.idn < len(offsetList):
                M2.append(Partition(p.idn, offsetList[p.idn-1], p.befPeriod, p.aftPeriod, p.rc, p.rcflag))
            else:
                M2.append(Partition(p.idn, randint(1, p.befPeriod), p.befPeriod, p.aftPeriod, p.rc, p.rcflag))


#        for p in M2:
#            print('id',p.idn, 'offset', p.offset, 'period', p.befPeriod)
#        print('fss')
#            M2.append(p)
#            info[0].append(int(p.idn))
#            info[1].append(int(p.offset))
#            info[2].append(int(p.befPeriod))
#            info[3].append(int(p.aftPeriod))
#            info[4].append(int(p.rc))
#            info[5].append(p.rcflag)
        return M2, sumu0, sumu1, info


    def Compare(self):
        '''Mode 1: Comparison Uitilization Change
           Mode 2: Comparison between Heu and ILP
        '''    
        Mode = 1
        
        if Mode == 1:            
            uBeforeSet = [i for i in range(1, 10)] 
            uAfterSet = [i for i in range(1, 10)]
            ratioMax = 1
            budget_Set = [0]
            reconf_max = 1
        else:
            uBeforeSet = [9] 
            uAfterSet = [9]
            ratioMax = 10
            budget_Set = [0, 1, 2, 3, 4, 6, 8, 12, 20]
            reconf_max = 1
        
        maxPeriod = 128
        countHeuSet = []
        countILPSet = []
        countOverTimeSet = []
        taskNumber = 15
        minimumT = 1
        Uset0 = []
        Uset1 = []
        TaskNumSet = []
        countNewSet = []
        countDecreaseSet = []
        countIncreaseSet = []
        budgetUseSet = []
        data_period = Period_Combo(taskNumber, minimumT,  maxPeriod)
        count_all = 0  
        book = xlwt.Workbook()
        sheet1 = book.add_sheet('sheet3')
        
        for u_before in uBeforeSet:
            for u_after in uAfterSet:
                countFeasible = 0
                countFeasibleILP = 0
                countOverTime = 0
                
                U0 = []
                U1 = []
                TaskNum = []
                countNew = []
                countDecrease = []
                countIncrease = []
                budgetUse = []
                count = 0

                offsetList = []

                countRatioFeasible = []
                countILP = []
                countLast = []
                
                HeuTimeSet = [] 
                ILPTimeSet = []
                HeuTimeSet_averge = []
                ILPTimeSet_averge = []

                for i in range(0, 10):
                    countILP.append([])
                    countRatioFeasible.append([])
                    countLast.append([])
                    
                    HeuTimeSet.append([])
                    ILPTimeSet.append([])
                    
                    for j in range(0, 10):
                        countILP[i].append([])
                        countRatioFeasible[i].append([])
                        countLast[i].append([])
                        for k in range(0, 10):
                            countILP[i][j].append([])
                            countRatioFeasible[i][j].append([])
                            countLast[i][j].append([])
                        
                            countRatioFeasible[i][j][k] = 0
                            countILP[i][j][k] = 0
                            countLast[i][j][k] = False
                
                max_count = min(len(data_period[u_before]), \
                        len(data_period[u_after])) 
                
                while  count < 1000:
                    currentTime = np.random.choice(range(0, 128)) # current time random
                    if len(data_period[u_before]) - 1 == 0:
                        sub_count0 = 0
                    else:
                        sub_count0 = np.random.choice(range(0, len(data_period[u_before]) - 1))

                    if len(data_period[u_after]) - 1 == 0:
                        sub_count1 = 0
                    else:
                        sub_count1 = np.random.choice(range(0, len(data_period[u_after]) - 1))  
                    
#                    sub_count0 = count%(len(data_period[u_before]) - 1)
#                    sub_count1 = count%(len(data_period[u_after]) - 1) 
#                    
                    
                    L0 = list(data_period[u_before][sub_count0]) 
                    L1 = list(data_period[u_after][sub_count1])
                    
                    shuffle(L0)
                    shuffle(L1)


                    [M, sumU0, sumU1, info] = self.Init_Combo(L0, L1, 1)
                        
#                    print('s') 
                    
#                    [M, sumU0, sumU1, info] = self.Init1(20, 20, u_before, u_after, maxPeriod, 1)
#                    [M, sumU0, sumU1, info] = self.Init1(10, u_before, u_after, maxPeriod, 1)
#                    print('sxcv')
#                    M2 = copy.deepcopy(M)

#                    for p in M:
#                        p.PrintPartition()


                    if  validate_regular(M, True) == False:
                        print('fail validation')
                        continue
#                        modeCountSet.append(modeCount)
#                    else:
#
#                        [M, sumU0, sumU1, info] = self.Init2(taskNumber, copy.deepcopy(M), j,  maxPeriod, offsetList, 2)
#                        modeCount += 1

                    for ratio in range(0, ratioMax):
                        
#                        if ratio != 2:
#                            continue
                        
                        if ratioMax != 1:
                            targetRc_range = int(len(M)*ratio/(ratioMax - 1))
                        else:
                            targetRc_range = 0
#                        print('rg', targetRc_range)
                        for p in M:
                            if p.idn - 1 < targetRc_range or (ratio == ratioMax - 1 and ratio != 0):
                                p.rc  = randint(2, 5)


                        
                        U0.append(sumU0)
                        U1.append(sumU1)
                        TaskNum.append(len(M))
                        
                        
                        countFalse = 0
                        count_cg0 = 0
                        count_cg1 = 0
                        increase_value = []
                        decrease_value = []
                        increase_True = []
                        for p in M:
                            if p.rcflag == False:
                                
                                countFalse += 1
                                
                            if p.befPeriod > p.aftPeriod and p.rcflag == True:
                                decrease_value.append(p.aftPeriod - p.befPeriod)
                                count_cg0 += 1 
                            if p.befPeriod < p.aftPeriod and p.rcflag == True:
                                Increase.append(p.befPeriod - p.aftPeriod)
                                count_cg1 += 1
                             
                        countNew.append(countFalse)
                        countIncrease.append(count_cg0)
                        countDecrease.append(count_cg1)
                        TaskNum.append(len(M))
                        
                        for reconf in range(0, reconf_max):
                            reconfig_time = (reconf_max - 1 - reconf) * 1 
                            success = True
                            for bd in range(0, len(budget_Set)):
                                budget = budget_Set[bd]
                                Heustart = time.time()
                                M_copy = copy.deepcopy(M)
                                [m12, offsetList, Giveout_bd] = self.Alg1(M, currentTime, budget, reconfig_time)
                                HeuTime = time.time() - Heustart 
                                HeuTimeSet[bd].append(HeuTime)
                                
                                if m12 != None: #heuristic works
                                    countRatioFeasible[ratio][reconf][bd] += 1
                                    countFeasible = countRatioFeasible[ratio][reconf][bd]
                                    budgetUse.append(Giveout_bd)

    #                                print('used budget:', Giveout_bd)
    
                                force_flag = 1 
                                if force_flag == 0: 
#                                if (countLast[ratio - 1][reconf][bd] == True and ratio >= 1) or (countLast[ratio][reconf][bd - 1] == True and bd >= 1) or (countLast[ratio][reconf - 1][bd] == True and reconf >= 1):
                                    
                                    countLast[ratio][reconf][bd] = True
                                    countILP[ratio][reconf][bd] += 1

                                
                                else:
    
                                    ILPstart = time.time()
                                    if u_after != 99:
                                        Result = False
                                    else:
                                        [Result, Ilp_budget] = self.ILP0(M_copy, currentTime, budget, maxPeriod, reconfig_time)
#                                    Result = True

                                    ILPTime = time.time() - ILPstart 
                                    ILPTimeSet[bd].append(ILPTime)
                                    
                                    
                                    
                                    if Result == True:#ILP works
                                        
                                        countILP[ratio][reconf][bd] += 1
                                        countLast[ratio][reconf][bd] = True
                                        countFeasibleILP = countILP[ratio][reconf][bd]
#                                        if m12 == None:
#                                            print('Heuristic is wrong')#heuristic wrong
#                                            print('ratio', ratio)
#                                            print('current_time', currentTime, 'budget', budget)
#                                            with open("Partition"+'.json','w') as outfile:
#                                                json.dump(info,outfile,ensure_ascii=False)
#                                                outfile.write('\n')
#    #                                        return
#                                            
#                                        else:
#                                            aaa =  1
    #                                        if Ilp_budget != Giveout_bd:
    #                                            print('ratio', ratio)
    #                                            print('budged inconsist')#ILP wrong
    #                                            print('current_time', currentTime, 'budget', budget)
    #                                            with open("Partition"+'.json','w') as outfile:
    #                                                json.dump(info,outfile,ensure_ascii=False)
    #                                                outfile.write('\n')
    #                                            return
                                    else:
                                        countLast[ratio][reconf][bd] = False
                                        
                                        if m12 != None and ILPTime > 60 and u_after == 9:
                                            countOverTime += 1
                                        
                                        if m12 != None and ILPTime < 59 and u_after == 9:
                                            print('ratio', ratio, 'ILP time',  ILPTime)
                                            print('ILP is wrong')#ILP wrong
                                            print('current_time', currentTime, 'budget', budget, 'reconf', reconfig_time)
                                            
                                            with open("Partition1"+'.json','w') as outfile:
                                                json.dump(info,outfile,ensure_ascii=False)
                                                outfile.write('\n')                                           
                                            
                                            return
                                         





                    count += 1
                    count_all += 1
                    print('count:', count_all)

                print(count)
                
                budgetUseSet.append(float('%0.2f'%np.mean(budgetUse)))
                TaskNumSet.append(float('%0.2f'%np.mean(TaskNum)))
                countNewSet.append(float('%0.2f'%np.mean(countNew)))          
                countDecreaseSet.append(float('%0.2f'%np.mean(countDecrease)))
                countIncreaseSet.append(float('%0.2f'%np.mean(countIncrease)))
                Uset0.append(float('%0.3f'%np.mean(U0)))
                Uset1.append(float('%0.3f'%np.mean(U1)))
#                countHeuSet.append(float('%0.3f'%(countFeasible/max_count)))
                countHeuSet.append(countFeasible)
                countILPSet.append(countFeasibleILP)
                countOverTimeSet.append(countOverTime)
#                countBFSet.append(countBF)
#                countTotal.append(countInfeasible )
            
                
              

            


        for i in range(0, 10):
            HeuTimeSet_averge.append(np.mean(HeuTimeSet[i]))
            ILPTimeSet_averge.append(np.mean(ILPTimeSet[i]))
        
        
        
        

#        for i in range(0, 2):
    
        if Mode == 2:
            for j in range(0, 10):
                if j == 0:    
                    sheet1.write(2, 1 + j, 'budget')
                else:
                    sheet1.write(2, 1 + j, budget_Set[j-1])
        
        
        
        for i, row1 in enumerate(countRatioFeasible):
            for j, row2 in enumerate(row1):
                for k, e in enumerate(row2):
                    if k < 9:
                        if j < 1:
                            sheet1.write(i + 3, j * 10 + 2 + k, e)
                        elif j < 2:
                            sheet1.write(i + 3, j * 10 + 3 + k, e)
                        elif j < 3:
                            sheet1.write(i + 17, (j - 2) * 10 + 2 + k , e)
                        elif j < 4:
                            sheet1.write(i + 17, (j - 2) * 10 + 3 + k , e)
                        elif j < 5:
                            sheet1.write(i + 31, (j - 4)  * 10 + 2 + k, e)
                        else:
                            teamwork = 1



        name = "Reconf_time_result01.xls"
        #name = "re.xls"
        book.save(name)
        book.save(TemporaryFile())

#        print('modeCountSet',modeCount)
        print('Heu uti countHeuSet', countHeuSet)
        print('ILP uti countILPSet', countILPSet)
        print('ILP over time Set', countOverTimeSet)
        print('Heu time', HeuTimeSet_averge)
        print('ILP time', ILPTimeSet_averge)
#
        print('Heu', countRatioFeasible)
        print('ILP', countILP)
#        print('All', count)
        print('U before :', Uset0)
        print('U after :', Uset1)
        print('taskNumber :', TaskNumSet)
        print('countNew :', countNewSet)
        print('countIncreaseSet :', countIncreaseSet)
        print('countDecreaseSet :', countDecreaseSet)
        #print('budgetUse :', budgetUseSet)
        with open("test_1.txt", "a+") as f:
            f.write('##########------test1------##########---\r\n')
            f.write('Heu : %s \r\n' %countHeuSet)
#            f.write('ILP : %s \r\n' % countBFSet)
            f.write('U before : %s \r\n' % Uset0)
            f.write('U before : %s \r\n' % Uset1)
            f.write('U after : %s \r\n' % Uset1)
            f.write('taskNumber %s \r\n:'% TaskNumSet)
            f.write('budgetUse %s \r\n:'% budgetUseSet)


    def ILP(self, M,  currentTime, budget, pmax):
        InstanceIndex = [0]*len(M)
        InstanceBudget = [0]*len(M)

       
        for T in range(0, budget + 1):
            
            L = perm_all(T, len(M))
        
#            print(L)
            for ls in L:
                
                
                InstanceBudget = ls
#                print(InstanceBudget,'budgt', T)
                for i, p in enumerate(M):
                    InstanceIndex[i] =  int(InstanceBudget[i] + pmax/p.aftPeriod)
    
    

                # Create optimization model
                m = Model('flowshop')
                m.setParam('OutputFlag', 0)
                m.setParam("LogToConsole", 0)
                # Create variables vtype=GRB.INTEGER
    
                Instances = [(i, p.idn) for p in M for i in range(1, InstanceIndex[p.idn-1] + 1)]
                y=[(i, m.idn, j, h.idn) for m in M for i in range(1, InstanceIndex[m.idn-1] + 1)  for h in M for j in range(1, InstanceIndex[h.idn-1]+1)  if i!=j or m!=h]
                X = m.addVars(Instances, vtype=GRB.INTEGER, name = "X")
                eq = m.addVars(y, vtype=GRB.BINARY, name="eq")
    
                # Constraints not equal
                m.addConstrs(
                        (X[i, m.idn] - X[j, k.idn] - eq[i,m.idn,j,k.idn]*1000 <=-1  for m in M for i in range(1, InstanceIndex[m.idn-1]+1) for k in M  for j in range(1, InstanceIndex[k.idn-1]+1) if i!=j or  m!=k) ,"Resource1")
                # Constraints not equal
                m.addConstrs(
                        (X[i, m.idn] - X[j, k.idn] - eq[i,m.idn,j,k.idn]*1000 >= 1 - 1000  for m in M for i in range(1, InstanceIndex[m.idn-1]+1) for k in M  for j in range(1, InstanceIndex[k.idn-1]+1) if i!=j or  m!=k) ,"Resource2")
    
    

    
                for p in M:
                    for i in range(1, InstanceIndex[p.idn-1]):
                        m.addConstr(
                                X[i, p.idn] - 1 <= X[i + 1, p.idn], "vals")
    
    
                Mcopy = copy.deepcopy(M)
                for i, p in enumerate(Mcopy):
                    p.r = 0#restore r
                    if p.rcflag == True:#reconfigured or unchanged partition
                        if currentTime <= p.offset:#corner case when t <= offset
                            p.d = -1.0 * currentTime * p.beffactor
                        else:
                            t_t = currentTime % p.befPeriod
                            if t_t <= p.offset:#corner case where t_t <= p.offset
                                t_t += p.befPeriod
                            p.d = p.beffactor * ((p.offset + 1) - t_t)
    
                        assert p.d <= 0
                    #print('divation is %f : %f'%(p.d, p.beffactor))
                    else:#new partition
                        p.d = 0
    
                    deadline = int((p.rc + p.d)/p.aftfactor)
    
#                    print('ID',p.idn, 'deadline', deadline)
                    m.addConstr(
                         X[1, p.idn] <= deadline - 1, "vals1")
                    m.addConstr(
                         X[1, p.idn] <= T + p.aftPeriod - 1, "vals1")
                    m.addConstr(
                         X[1, p.idn] >= 0, "vals")
    
    
    
    
                for p in M:
                    for i in range(InstanceBudget[p.idn-1] + 1, InstanceIndex[p.idn-1]):
                        m.addConstr(
                                X[i + 1, p.idn] - X[i, p.idn] == p.aftPeriod, "vals")
    
    
    
                for p in M:
                    for i in range(1, InstanceBudget[p.idn-1] + 1):
                        
                        for num0 in range(1, i + 1):
                            for num1 in range(1, i + 2):
                                if num0 <= i and num1 > num0:
                                    assert num1 <= InstanceBudget[p.idn-1] + 1
                                    m.addConstr(
                                            #p.aftfactor * (X[num1, p.idn] - 1 -  X[num0, p.idn]) - (num1 - num0) >= - p.rc, "vals")
    					(num1 - num0) - p.aftfactor * (X[num1, p.idn] - 1 -  X[num0, p.idn]) >= - p.rc, "vals")
    
    
    
    #
    #            m.setParam('PoolSolutions', 200000)
    #            m.setParam('PoolSearchMode', 2)
    #            m.params.solutionLimit = 1
    
                m.setParam('TimeLimit', 60)
                m.optimize()
                # Print solution
                if m.status == GRB.Status.OPTIMAL:
    
                    solution_x = m.getAttr('x', X)
#                    print('xxxxxxx', solution_x)
    #                solution_pool_size = m.getAttr('SolCount')
    #                print ("Solution pool contains {0} solutions".format(solution_pool_size))
    
                    return True, sum(ls)
                    
                    #solution_x = m.getAttr('x', X)
                    #print(sorted(solution, key=solution.get))
                    #print(solution_x.values())

        return False, None


    def ILP0(self, M,  currentTime, budget, pmax, reconf_time):
#                print(InstanceBudget,'budgt', T)
        amount_Instance = [0]*len(M)
        for i, p in enumerate(M):
            if p.rcflag == True:    
                amount_Instance[i] =  int(budget + pmax/p.aftPeriod)
            else:
                amount_Instance[i] =  int(pmax/p.aftPeriod)



        # Create optimization model
        m = Model('flowshop')
#        m.setParam('OutputFlag', 0)
#        m.setParam("LogToConsole", 0)
        # Create variables vtype=GRB.INTEGER

        Instances = [(p.idn, i) for p in M for i in range(0, amount_Instance[p.idn-1])]
        y=[(m.idn, i , h.idn, j) for m in M for i in range(0, amount_Instance[m.idn-1])  for h in M for j in range(0, amount_Instance[h.idn-1])  if i!=j or m!=h]
        f = m.addVars(Instances, vtype = GRB.INTEGER, name = "f") #finish time
        eq = m.addVars(y, vtype = GRB.BINARY, name ="eq")
        u = m.addVars(Instances, vtype = GRB.BINARY, name = "u")

        hugeNumber = 5000000
        
        # Constraints not equal
        m.addConstrs(
                (f[m.idn, i] - f[k.idn, j] - eq[m.idn, i, k.idn, j]*hugeNumber <=-1  for m in M for i in range(0, amount_Instance[m.idn-1]) for k in M  for j in range(0, amount_Instance[k.idn-1]) if i!=j or  m!=k) ,"Resource1")
        # Constraints not equal
        m.addConstrs(
                (f[m.idn, i] - f[k.idn, j] - eq[m.idn, i, k.idn, j]*hugeNumber >= 1 - hugeNumber  for m in M for i in range(0, amount_Instance[m.idn-1]) for k in M  for j in range(0, amount_Instance[k.idn-1]) if i!=j or  m!=k) ,"Resource2")



        # constraints for reconfig_time
        for p in M:
            for i in range(0, amount_Instance[p.idn-1]):
                m.addConstr(
                         f[p.idn, i] >= reconf_time + 1, "vals")
        
        

        #transistion schedule : first job
        Mcopy = copy.deepcopy(M)
        for i, p in enumerate(Mcopy):
            p.r = 0#restore r
            if p.rcflag == True:#reconfigured or unchanged partition
                if currentTime <= p.offset:#corner case when t <= offset
                    p.d = -1.0 * currentTime * p.beffactor
                else:
                    t_t = currentTime % p.befPeriod
                    if t_t <= p.offset:#corner case where t_t <= p.offset
                        t_t += p.befPeriod
                    p.d = p.beffactor * ((p.offset + 1) - t_t)

                assert p.d <= 0
            #print('divation is %f : %f'%(p.d, p.beffactor))
            else:#new partition
                p.d = 0

            deadline = int((p.rc + p.d)/p.aftfactor)

            print('ID',p.idn, 'deadline', deadline)
            
            if p.rcflag == True:
                m.addConstr(
                     f[p.idn, 0] <= deadline, "vals1")
                m.addConstr(
                     f[p.idn, 0] <= budget + p.aftPeriod, "vals1")
                m.addConstr(
                     f[p.idn, 0] >= 1, "vals")
            else:
                m.addConstr(
                     f[p.idn, 0] <= budget + p.aftPeriod, "vals1")

        #rc-flga : false
        for p in M:
            if p.rcflag == False:
                for i in range(0, amount_Instance[p.idn-1] - 1):
                    m.addConstr(
                            u[p.idn, i] == 1, "vals")

        #transistion schedule: the remaining jobs
        
        
        for p in M:
            for i in range(0, amount_Instance[p.idn-1] - 1):
                m.addConstr(
                        f[p.idn, i + 1] - 1  >= f[p.idn, i] - hugeNumber * u[p.idn, i] , "vals")
                
                m.addConstr(
                        f[p.idn, i] <= budget + hugeNumber * u[p.idn, i] , "vals")
                
        
#        for p in M:
#            for i in range(0, amount_Instance[p.idn-1]):
#                
#                for num0 in range(0, i):
#                    for num1 in range(0, i + 1):
#                        if num0 <= i and num1 > num0:
#                            assert num1 <= amount_Instance[p.idn-1]
#                            m.addConstr(
                                    #p.aftfactor * (X[num1, p.idn] - 1 -  X[num0, p.idn]) - (num1 - num0) >= - p.rc, "vals")
				#	(num1 - num0) - p.aftfactor * (f[p.idn, num1] - 1 -  f[p.idn, num0]) >= - p.rc, "vals")
#                                   p.aftfactor * (f[p.idn, num1] - 1) <= (num1 - num0) + p.rc + p.aftfactor * f[p.idn, num0] + hugeNumber * u[p.idn, num0], "vals")
                    
        for p in M:  
                for num0 in range(0, amount_Instance[p.idn-1]):
                    for num1 in range(num0, amount_Instance[p.idn-1]):
                        if num1 > num0:
                            assert num1 <= amount_Instance[p.idn-1]
                            m.addConstr(
                                    p.aftfactor * (f[p.idn, num1] - 1) <= (num1 - num0) + p.rc + p.aftfactor * f[p.idn, num0], "vals")
        
        
        
        



        #cyclic schedule: 
        for p in M:
            for i in range(0, amount_Instance[p.idn-1] - 1):
                m.addConstr(
                        f[p.idn, i + 1] >= f[p.idn, i] + p.aftPeriod - hugeNumber * (1 - u[p.idn, i]), "vals")
                m.addConstr(
                        f[p.idn, i + 1] <= f[p.idn, i] + p.aftPeriod + hugeNumber * (1 - u[p.idn, i]), "vals")
                m.addConstr(
                        f[p.idn, i] >= budget + 1 - hugeNumber * (1 - u[p.idn, i]), "vals")
            if amount_Instance[p.idn-1] - 1 == 0:
                m.addConstr(
                        f[p.idn, 0] >= budget + 1 - hugeNumber * (1 - u[p.idn, 0]), "vals")




#
#            m.setParam('PoolSolutions', 200000)
#            m.setParam('PoolSearchMode', 2)
#            m.params.solutionLimit = 1

        m.setParam('TimeLimit', 60)
        m.optimize()
        # Print solution
        if m.status == GRB.Status.OPTIMAL:

            solution_x = m.getAttr('x', f)
#            print('xxxxxxx', solution_x)
#                solution_pool_size = m.getAttr('SolCount')
#                print ("Solution pool contains {0} solutions".format(solution_pool_size))
            return True, 3
            
            
            #solution_x = m.getAttr('x', X)
            #print(sorted(solution, key=solution.get))
            #print(solution_x.values())

        return False, None



def partitions(n):
	# base case of recursion: zero is the sum of the empty list
	if n == 0:
		yield []
		return
		
	# modify partitions of n-1 to form partitions of n
	for p in partitions(n-1):
		yield [1] + p
		if p and (len(p) < 2 or p[1] > p[0]):
			yield [p[0] + 1] + p[1:]
            

class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield list(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1


def perm_all(budget, tasknum):
    
    ls = []
    for i in partitions(budget):
        diff = tasknum - len(i)
        while(diff > 0):
            i.append(0)
            diff -= 1
        ls.append(i)    
        
    new_ls = []
    for i in ls:
        new_ls += list(perm_unique(i))
    
    return new_ls
    

def main():
    sch = Schedule()
    test = 1
    start = time.time()
    Dataset = []
#    print(sch.Init_Combo(4, 0, 1, 64, Dataset))
#    your_list = Period_Combo(20, 128)
#    print(your_list)
#    your_list.append(your_list[1])
    
#    print(len(your_list) != len(set(your_list)))
#    print(len(your_list))
    
    if test == 0:
#        print(perm_all(1, 5))
        M = []
        f3=open("Partition1.json","r")
        json1_str = f3.read()
        info = json.loads(json1_str)
        for i in range(0, len(info[0])):
            M.append(Partition(info[0][i], info[1][i], info[2][i], info[3][i], info[4][i], info[5][i]))
#        
#        current_time = 40

#        for p in M:
#            p.PrintPartition()

        ratioMax = 10
        
#        M[0].rc = 5
#        M[1].rc = 5
#        M[2].rc = 4
#        M[3].rc = 2
         
        ratio = 2 
        targetRc_range = int(len(M)*ratio/(ratioMax - 1))
        for p in M:
            if p.idn - 1 < targetRc_range or ratio == ratioMax - 1:
                p.rc  = 1    
 
        for p in M:
            p.PrintPartition()
            
        current_time = 38
        budget = 0 
        reconf_time = 0
        pmax = 128
        [m12, offsetList, Giveout_bd] = Alg1(M, current_time, budget, reconf_time)
        if m12 == None:
            print('heu fails')
        else:
            print('heu works')
            [flag, bd] = sch.ILP0(M, current_time, budget, pmax, reconf_time)
            if flag == True:
                print('ILP works')
            else:
                print('ILP wrong')
    else:
        sch.Compare()
        
    done = time.time()
    elapsed = done - start
    print('elapsed:', elapsed)

if __name__ == "__main__":
    main()

