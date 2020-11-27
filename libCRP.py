#!/usr/bin/python
'''
This software is the product of research performed at the University of Texas at Austin

Author: Wei-Ju Chen, albertwj@cs.utexas.edu
'''

from header import _SUCCESS, _CRPFAIL , _DPRFAIL , _DPRFAIL_2 , _GEN_FAIL, UNIFORM, LC, OFFONE, NAIVE, AFT, PRE, TRAN
from random import randint, shuffle, random
from itertools import combinations, permutations, combinations_with_replacement
import math

#Test whether M is a valid set of regular resource partition
#M is the set of resource partition
#is_bef indicate the setting is before RPRR
def validate_regular(M, phase):
    max_p = 0
    mark_idn = []
    mark_offset = []

    #find max_p
    for rp in M:
        max_p = max(max_p, rp.period[phase])

    #check invalid idn
    for rp in M:
        if rp.idn in mark_idn: #duplicate idn
            return False
        else:
            mark_idn.append(rp.idn)

    #check invalid offset
    for rp in M:
        num_period = int(max_p/rp.period[phase])
        test_period = rp.period[phase]

        if rp.offset[phase] >= test_period:
            return False
        for i in range(0, num_period):
            if phase == 0 and rp.rcflag == 0: # new partition
                continue
            elif rp.offset[phase] + i * test_period in mark_offset: # duplicate
                return False
            else:
                mark_offset.append(rp.offset[phase] + i * test_period)

    return True

# Test the RC regularity in nonuniform-Environment
# M is the resource partition set
# t is the time of R3P
# m1 is the transition schedule
# m2 is the cyclic schedule
def validate_rc_regular(M, t, m12):
    #construct the before schedule

    m0 = [0]* t
    for p in M:
        for i in range(0, int(math.ceil((float(t)/p.period[0])))):
            if p.period[0] * i + p.offset[0] < t:
                m0[p.period[0] * i + p.offset[0]] = p.idn

    if len(m12) == 0:
        return m0, False

    #for all (a,b], a after execution, b before execution
    #d = # of slices - \alpha * (b - a) #       3 schedule offset, a after execution 最近的request.
    
    

    
    
    #check d > -Rc
    total = m0 + m12
    for p in M:
        start = []
        end = []
        for i, task in enumerate(total):
            if task == p.idn:
                start.append(i + 1)
                end.append(i)

        for j, s in enumerate(start):
            for k, e in enumerate(end):
                if e <= s:
                    continue

                if e <= t:
                    pre_int = e - s
                    aft_int = 0
                elif s >= t:
                    pre_int = 0
                    aft_int = e - s
                else:
                    pre_int = t - s
                    aft_int = e - t

                if (k - j - 1) - p.beffactor * pre_int - p.aftfactor * aft_int  <= -p.rc:
                    return False

    return True



# Test the RC regularity in nonuniform-Environment
# M is the resource partition set
# t is the time of R3P
# m1 is the transition schedule
# m2 is the cyclic schedule
def validate_rc_regular_non_uniform(M, t, m12, budget):
    #construct the before schedule
    print('schedule:-----')
    print(m12[:budget], m12[budget:])
    print('request')
    for p in M:
        print(p.idn, 'id --------')
        print(p.requests)
        
    return True
    
    
    m0 = [0]* t
    for p in M:
        for i in range(0, int(math.ceil((float(t)/p.period[0])))):
            if p.period[0] * i + p.offset[0] < t:
                m0[p.period[0] * i + p.offset[0]] = p.idn

    if len(m12) == 0:
        return m0, False

    #for all (a,b], a after execution, b before execution
    #d = # of slices - \alpha * (b - a) #       3 schedule offset, a after execution 最近的request.
    
    
    #a: resource 之后， 最近的一个request点，c
    #b: resource 之前
    
    # 1
    # 1 2
    
    #对于 id 1 parition: b = 1， c = 1
    
    # before 1
    # [1, 2] 当作 
    # 如果resource offset之前没有resource offset的话，当作 -1-0有 resource offset。
    
    
    
    
    
    
    #check d > -Rc
    total = m0 + m12
    for p in M:
        start = []
        end = []
        for i, task in enumerate(total):
            if task == p.idn:
                start.append(i + 1)
                end.append(i)

        for j, s in enumerate(start):
            for k, e in enumerate(end):
                if e <= s:
                    continue

                if e <= t:
                    pre_int = e - s
                    aft_int = 0
                elif s >= t:
                    pre_int = 0
                    aft_int = e - s
                else:
                    pre_int = t - s
                    aft_int = e - t

                if (k - j - 1) - p.beffactor * pre_int - p.aftfactor * aft_int  <= -p.rc:
                    return False

    return True


# init p.allowed[phase] based on the request offsets
# precondition: all _from partition has allowed[phase] set
# side effect: set p.allowed[phase]
def init_allowed(p, phase, environment):
    p.allowed[phase] = [] # period 4, 0,1,2,3 # allow #0,1,2,#,4 #每个resource parition可以被schedule
    if phase == TRAN:
        period_index = AFT
    else:
        period_index = phase
    for i in range(0, p.period[period_index]):
        p.allowed[phase].append(i)
        #print(p.period[period_index])
    if environment == UNIFORM:
        return _SUCCESS
    elif environment == OFFONE and phase == TRAN:
        return _SUCCESS
    #else LC, NAIVE, OFFONE and AFT
    
#    print(p.idn, p._from)
    for _from in p._from:
        if _from.allowed[phase] == None:
            continue
        ratio = float(_from.slice_size) / p.slice_size
        chunk_from = _from.period[period_index] * _from.slice_size
        chunk_p = p.period[period_index] * p.slice_size
        # check structure
        if _from.slice_size * _from.period[period_index] < p.slice_size:
            return _CRPFAIL
    
        if chunk_p > chunk_from:
            repeat = chunk_p // chunk_from # change '/' to '//'
        else:
            repeat = 1
    
        # test is the request offset, need to use floor
        # in trasition pahse, for the allow time slots, we need to shrink them 
        
        for i in range(0, repeat):
            for a in _from.allowed[phase]:
                test = (a + 1 + _from.period[period_index] * i) * ratio
                if test == int(test): # continue also need to be counted in request_set
                    p.requests[phase].append(int(test)% p.period[period_index])
                    continue
                test = int(test) % p.period[period_index]
                if test in p.allowed[phase]:
                    p.allowed[phase].remove(test)
                    p.requests[phase].append(test)
                    
                    
    
        if len(p.allowed[phase]) == 0:
            return _GEN_FAIL
    
    return _SUCCESS

# pre-set an allowed[TRAN] set
def allowed_alg(alg, p, select):
    assert(p.allowed[TRAN] and select % p.period[AFT] in p.allowed[TRAN])
    if alg == UNIFORM:
        pass
    elif alg == NAIVE:
        pass
    elif alg == OFFONE:
        pass
    elif alg == LC:#Largest Contiguous
        size_smallest = 99999
        for rp in p._to:
            if rp.slice_size > p.slice_size:
                size_smallest = min(size_smallest, rp.slice_size)

        if size_smallest != 99999:
            ratio = size_smallest // p.slice_size
            target = select % p.period[AFT]
            bottom = (target - (target & (ratio - 1))) 
            tmp = []
            for i in range(bottom, bottom + ratio):
                if i in p.allowed[TRAN]:
                    tmp.append(i)
            p.allowed[TRAN] = tmp
    else:
        assert(False)

    assert(p.allowed[TRAN])

# find a slot for the transition
def dsedf(p, m, tb, environment, phase):
    assert p.e > p.r
    # do not modify deadline in-place
    tmp_e = p.e
    if tmp_e > tb:
        tmp_e = tb

    for t_t in range(tmp_e - 1, p.r - 1, -1):
        if m[t_t] == 0 and (t_t % p.period[AFT]) in p.allowed[phase]:
            return t_t

    return None

# fill the cyclic schedule based on the chosen offset
def mark_schedule(select, p, m, pmax):
    assert(len(m) >= pmax)
    tmp = select % p.period[AFT]
    for tk in range(0, int(pmax/p.period[AFT])):
        if m[tmp + tk * p.period[AFT]] != 0:
            return None
        m[tmp + tk * p.period[AFT]] = p.idn
        #p.aft_offsets.append(tmp + tk * p.period[AFT])
    return select

            
# find the right most allowed node in the tree
# precondition: rp.allowed is inited
# parameters: e deadline, p period, m schedule, idn partition id,
#               rp partition 
# return: node location in m
def find_feasible_node(e, p, m, pmax, idn, rp, phase):
    assert(len(m) >= pmax)
    for t_t in range(min(e, p) - 1, -1, -1):
        if m[t_t] == 0 and t_t in rp.allowed[phase]:
            for tk in range(0, int(pmax/p)):
                if m[t_t+tk*p] != 0:
                    return None
                m[t_t+tk*p] = idn
            return t_t

    return None

# convert partitions to tasks
# M: partitions/tasks, t: time of R3P
def Alg2(M, t, environment):
    for p in M:
        p.r = 0 # restore r
        if p.rcflag:#reconfigured or unchanged partition
            if t <= p.offsets[0]:#corner case when t <= offset
                p.d = -1.0 * t / p.period[0]
            else:
                t_t = t % p.period[0]
                if t_t <= p.offsets[0]:#corner case where t_t <= p.offset
                    t_t += p.period[0]

                p.d = float((p.offsets[0] + 1) - t_t) / p.period[0]

            assert p.offsets[0] < p.period[0]
            assert p.d <= 0
        else:#new partition
            p.d = 0

        p.e = int((p.rc + p.d) * p.period[1])
        assert p.e >= p.r
        if p.e == 0:
            return False

def rde_update(p, l, environment):
    if environment == UNIFORM or environment == LC or environment == NAIVE:
        
        #l + 1 - p.r means the distance between two release
        p.d = min(0, p.d + 1 - p.aftfactor * (l + 1 - p.r))
        p.e = int((p.rc + p.d)/p.aftfactor + l + 1)
        p.r = l + 1
        assert p.d <= 0
        assert p.d > -1 * p.rc
        assert p.e >= 0 and p.e > p.r
    elif environment == OFFONE:
        
        p.d = min(-p.aftfactor + 0.00001, p.d + 1 - p.aftfactor * (l + 1 - p.r))
        p.e = int((p.rc + p.d)/p.aftfactor + l + 1)
        p.r = l + 1
        assert p.d <= 0
        assert p.d > -1*p.rc
        assert p.e >= 0 and p.e > p.r
    else:
        assert False

BLOCK = -99999
# compute the transition schedule
# M: tasks, tb: budget of reconfiguration, reconf_time: blocking time
def Alg3(M, tb, reconf_time, environment):
    m = [0] * tb
    E = []
    for i in range(0, tb):
        if i < reconf_time:
            m[i] = BLOCK

    queue = sorted(M, key = lambda Paritition: (Paritition.e, Paritition.period[1]))

    while len(queue) != 0:
        p = queue.pop(0)
        l = dsedf(p, m, tb, environment, TRAN)
        if l != None:
            p.tran_offsets.append(l)
            m[l] = p.idn
            if p.configured == False:
                allowed_alg(environment, p, l)
                p.configured = True

        if l == None:
            if p.e <= tb:
                #//deadline will miss
                return _DPRFAIL, None, None
            #need to update the deadline to counting from tb
            p.r = 0
            p.e = p.e - tb
            E.append(p)
            p.allowed[TRAN] = p.tran_offsets
        else:
            rde_update(p, l, environment)

            #assume to requeue the task with the order
            for i, px in enumerate(queue):
                if p.e == px.e:
                    if p.period[1] <= px.period[1]:
                       queue.insert(i, p)
                       break
                elif p.e < px.e:
                    queue.insert(i, p)
                    break
            else:
                queue.append(p)

    return  _SUCCESS, E, m

# compute the cyclic schedule
# E: tasks, reconf_time: blocking time
def Alg4(E, reconf_time, environment):
    pmax = 0
    m = []
    offsetList = {}

    for p in E:
        pmax = max(pmax, p.period[1])
    m = [0] * pmax
    if reconf_time >= pmax:
        return _DPRFAIL, None, None
    for i in range(0, reconf_time):
        m[i] = -99999

    queue = sorted(E, key = lambda Paritition: (Paritition.period[1], \
                Paritition.e))

    while len(queue) != 0:
        p = queue.pop(0)
        init_allowed(p, AFT, environment)
        # Optimal: select the right most feasible node
        p.e = min(p.e, p.period[AFT])
        t = dsedf(p, m, pmax, environment, AFT)
        p.allowed[AFT] = [t]

        if t ==  None:
            return _DPRFAIL, None, None
        t2 = mark_schedule(t, p, m, pmax)
        if t2 ==  None:
            return _DPRFAIL, None, None

        #legacy code
        offsetList[p.idn] = t

    return _SUCCESS, m, offsetList

def reset_task(M, environment):
    # reset instance
    for rp in M:
        rp.r = 0
        rp.e = 0
        rp.d = 0
        rp.beffactor = float(1.0/rp.period[0]) #(power of 1/2)
        rp.aftfactor = float(1.0/rp.period[1]) #(power of 1/2)
        rp.allowed= [[],[],[]]
        init_allowed(rp, TRAN, environment)
        rp.configured = False
        rp.tran_offsets = []

#Entry point for the reconfiguration
#M: task sets, t:the time of the reconfiguration,
#   budget: reconfiguration budget, reconfig_time: reconfiguration
#   operation blocking time, environment: UNIFORM or others
def Alg1(M, t, budget, reconfig_time, environment):
    for t_t in range(budget, budget + 1):
        schedule = []

        reset_task(M, environment)

        if Alg2(M, t, environment) == False:
            return _DPRFAIL, None, None, None

        [test, E, m1] = Alg3(M, t_t, reconfig_time, environment)
        if E == _CRPFAIL:
            return _CRPFAIL, None, None, None
        if E == None:
            continue

        if reconfig_time > t_t:
            [test, m2, offsetList] = Alg4(E, reconfig_time - t_t, environment)
        else:
            [test, m2, offsetList] = Alg4(E, 0, environment)

        if test == _CRPFAIL:
            return _CRPFAIL, None, None, None
        if m2 == None:
            continue

        schedule = m1 + m2
        #update request offsets
        return _SUCCESS, schedule, offsetList, len(m1)
    return _DPRFAIL, None, None, None

# TODO: 1 phase need to use request offsets
def alg_nu(sample, t, budget, reconfig_time):
    for ph in sample.gph.physicals:
        u = 0.0
        ua = 0.0
        [schedule, offsets, size] = Alg1(ph.rps, t, budget, reconfig_time, True)
        if schedule == None:
            return None

# create a list of period lists where
# period_list[0] is the list for 0 ~ 0.1
# period_list[1] is the list for 0.1 ~ 0.2
# period_list[9] is the list for 0.9 ~ 1.0
# len(period_list[i]) = targetNumber 
def period_combo(list_all_period, target_len):
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

def Period_Combo(targetNumber, minimum, maxPeriod):
    assert(maxPeriod >= 2 and maxPeriod <= 128)
    lstA = list(combinations_with_replacement([2, 4, 8, 16, 32, 64, 128] \
        , targetNumber))
    # TODO: period distribution need to be refined
    period_set = [set([]) for i in range(0, 10)]
    cleaned = [tuple(n for n in sublist if n != 0) for sublist in lstA]
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    const = 20
    del_ls = []

    for i, e in enumerate(cleaned):
        u = 0
        assert(len(e) == targetNumber)
        for j in e:
            u += 1.0/j
        for k in range(0, 10):
            if u > k * 0.1 and u <= 0.1 + k * 0.1:
                period_set[k].add(e)
                break

    period_list = []
    for i, l in enumerate(period_set):
        #assert(l) # shall not be empty
        period_list.append(list(l))

    return period_list

# Digraph
#   physical resources as nodes, edges
class Graph:
    def __init__(self):
        self.physicals = []
        self.edges = []
        self.crps = []
        self.period_list = None # 10/312020 peng for cache 

    def configure_physicals(self, size_list):
        for p in self.physicals:
            p.slice_size = size_list[randint(0, len(size_list) - 1)]
        return

        # period * size is at least larger than this size
        for p in self.physicals:
            small = 1000
            if p.edges == None:
                p.slice_size = size_list(len(size_list) - 1)
                continue
            for edge in p.edges:
                if edge._to == p.pid:
                    small = min(small, self.physicals[edge._from].slice_size)
            
            if small == 2:
                p.slice_size = 2
                continue

            for i in range(len(size_list) - 1, -1, -1):
                if size_list[i] < small:
                    p.slice_size = size_list[i]
                    break

            #print small, p.slice_size
            assert(p.slice_size)

        return

        for p in self.physicals:
            p.slice_size = size_list[randint(0, len(size_list) - 1)]
            continue
            # find all _from
            smallest_ps = 1024 * 1024
            for edge in p.edges:
                if edge._to == p.pid:
                    for p2 in self.physicals[edge._from].rps:
                        smallest_ps = min(smallest_ps, \
                                p2.period[1] * p2.slice_size)

            if smallest_ps == 1024 * 1024:
                p.slice_size = size_list[len(size_list) -1]
                for rp in p.rps:
                    rp.slice_size = p.slice_size
                continue

            target = -1
            for i in range(len(size_list) - 1, -1, -1):
                if size_list[i] < smallest_ps:
                    target = i
                    break
            if target == -1 or target == 0:
                target = 0
                target_b = 0
            else:
                target_b = target - 1

            #print p.pid, target, target_b, smallest_ps
            p.slice_size = size_list[randint(target_b, target)]
            assert(p.slice_size > 0)
            for rp in p.rps:
                rp.slice_size = p.slice_size

    def generate_digraph(self, psize, max_degree_out):
        for i in range(0, psize):
            self.physicals.append(PhysicalResource(i))

        for i in range(0, psize - 1):
            tmp_index = i
            for j in range(0, randint(0, max_degree_out)):
                if tmp_index == psize - 1:
                    break

                if tmp_index + 1 == psize - 1:
                    out = psize - 1
                    tmp_index = out
                else:
                    out = randint(tmp_index + 1, psize - 1)
                    tmp_index = out

                edge = Edge(i, out)
                self.edges.append(edge)
                self.physicals[i].edges.append(edge)
                self.physicals[out].edges.append(edge)

    def print_digraph(self):
        for i in self.physicals:
            print(i)
        for i in self.edges:
            print(i)
    
    #edge_p probability to select each edge
    def get_sub_graph(self,  edge_p):
        sub_phys = set([])
        sub_edges = []

        for e in self.edges:
            if random() < edge_p:
                sub_edges.append(e)
                sub_phys.add(self.physicals[e._from])
                sub_phys.add(self.physicals[e._to])

        return sub_phys, sub_edges

    # number 0 means NONE
    def sample_crps(self, num_crps, edge_p):
        for i in range(1, num_crps + 1):
            self.crps.append(Crp(i))

        for c in self.crps:
            c.sub_phys, c.sub_edges = self.get_sub_graph(edge_p)
            record = {}
            for e in c.sub_edges:
                if e._from not in record:
                    rp = Partition(c.cid, 0, 1, 1, 0, 0)
                    rp.loc = e._from
                    self.physicals[e._from].rps.append(rp)
                    c.rps.append(rp)
                    record[e._from] = 1
                if e._to not in record:
                    rp = Partition(c.cid, 0, 1, 1, 0, 0)
                    rp.loc = e._to
                    self.physicals[e._to].rps.append(rp)
                    c.rps.append(rp)
                    record[e._to] = 1
                
                rp_from = None
                for rp in self.physicals[e._from].rps:
                    if rp.idn == c.cid:
                        rp_from = rp

                rp_to = None
                for rp in self.physicals[e._to].rps:
                    if rp.idn == c.cid:
                        rp_to = rp

                assert(rp_from and rp_to)
                rp_from._to.append(rp_to)
                rp_to._from.append(rp_from)
                
    
    # modify offset and allowed in place of rp
    def test_and_set_cyclic_schedule(self, physical, phase):
        pmax = 0
        for rp in physical.rps:
            pmax = max(pmax, rp.period[phase])

        m = [0] * int(pmax)
        # inplace sorting
        queue = sorted(physical.rps,key = lambda Paritition: (Paritition.period[phase]))
        
        u_temp = 0
        while len(queue) != 0:
            rp = queue.pop(0)
            # set allowed
            test = init_allowed(rp, phase, LC)
            if test != _SUCCESS:
                return test

            rp.offsets[0] = find_feasible_node(rp.period[phase],\
                    rp.period[phase], m, pmax, rp.idn, rp, phase)

            if rp.offsets[0] == None:
                return _CRPFAIL
            # finish allowed
            rp.allowed[phase] = [rp.offsets[0]]
        return _SUCCESS

    def configure_rps(self, all_period, befRatio, aftRatio, max_rc, period_list):
        max_number_rp = 4 #minimum # of rp on each physical resource
        
        
        for i in self.physicals:
            max_number_rp = max(max_number_rp, len(i.rps))
        
        #---- deleted by peng 10/31/2020 for cache -----
        #period_list = period_combo(all_period, max_number_rp) 
  
        
        bef_size = len(period_list[befRatio])
        aft_size = len(period_list[aftRatio])
        
        for p in self.physicals:
            # fill physical resource with resource partitions to max
            pre_size = len(p.rps)
            if pre_size < max_number_rp:
                for i in range(0, max_number_rp - pre_size):
                    p.rps.append(Partition(-1 - i, 0, 1, 1, 0, 0))

            assert(len(p.rps) == max_number_rp)
            bef_list = period_list[befRatio][randint(0, bef_size - 1)]
            aft_list = period_list[aftRatio][randint(0, aft_size - 1)]
           
            for i, rp in enumerate(p.rps):
                rp.period[0] = bef_list[i]
                rp.period[1] = aft_list[i]
                rp.beffactor = float(1.0/rp.period[0]) #(power of 1/2)
                rp.aftfactor = float(1.0/rp.period[1]) #(power of 1/2)
                rp.d  = 0 #deivation
                rp.e  = 0 #deadline
                rp.r = 0  #release time
                rp.rc = randint(1, max_rc)
                rp.rcflag = 1
                rp.slice_size = p.slice_size
                rp.pid = p.pid

            _tmp = self.test_and_set_cyclic_schedule(p, 1)
            if _tmp != _SUCCESS:
                return _tmp
            _tmp = self.test_and_set_cyclic_schedule(p, 0)
            if _tmp != _SUCCESS:
                return _tmp
                
        return _SUCCESS

# Composite resource partition
class Crp:
    def __init__(self, cid):
        self.cid = cid
        self.rps = []
        self.sub_phys = []
        self.sub_edges = []

# Physical resource (hardware)
class PhysicalResource:
    def __init__(self, pid):
        self.pid = pid
        self.slice_size = 0
        self.edges = []
        self.rps = []
    
    def __str__(self):
        return str(self.pid)

# Digraph edge
#    a pair of nodes: _from -> _to (physical resource)
class Edge:
    def __init__(self, _from, _to):
        self._from = _from
        self._to = _to

    def __str__(self):
        return str(self._from) + ":" + str(self._to)

# Resource partition
class Partition:
    def __init__(self, idn, offset, bef_period, aft_period, rc, rcflag):
        self.idn = idn
        self.offsets = [offset, offset]
        self.period = [0,0, 0]
        self.period[0] = bef_period # harmonic
        self.period[1] = aft_period # harmonic
        self.beffactor = float(1.0/bef_period) #(power of 1/2)
        self.aftfactor = float(1.0/aft_period) #(power of 1/2)
        self.rc = rc
        self.r = 0  #release time
        self.d  = 0 #shortfall
        self.e  = 0 #deadline
        self.rcflag = rcflag # 0 = new partition, 1 = reconfigured or old
        self.allowed = [[],[],[]]
        self.slice_size = 1
        self._from = []
        self._to = []
        self.loc = None
        self.bef_offsets = []
        self.aft_offsets = []
        self.tran_offsets = []
        self.pid = 0
        self.requests = [[],[],[]]
        
        

    def PrintPartition(self):
        print('ID:',self.idn, 'offset', self.offset, 'bef', self.period[0], 'aft', self.period[1], 'rc', self.rc, 'flag', self.rcflag, 'bef', self.beffactor, 'aft', self.aftfactor)
