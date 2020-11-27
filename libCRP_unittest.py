#!/usr/bin/python
import unittest
from libCRP import *
from header import _SUCCESS, _CRPFAIL , _DPRFAIL , _DPRFAIL_2 , _GEN_FAIL 

class TestSampling(unittest.TestCase):
    def test_Period_Combo(self):
        for test in range(4, 10):
            period_list = Period_Combo(test, 2, 128)
            for i, pl in enumerate(period_list):
                for j in pl:
                    self.assertTrue(len(j) == test)
                    u = 0
                    for item in j:
                        u+= 1.0/item
                    self.assertTrue(0.1 * i < u and u <= 0.1 + 0.1 * i)

    def test_period_combo(self):
        for test in range(4, 10):
            period_list = period_combo([2, 4, 8, 16, 32, 64, 128], test)
            for i, pl in enumerate(period_list):
                for j in pl:
                    self.assertTrue(len(j) == test)
                    u = 0
                    for item in j:
                        u+= 1.0/item
                    self.assertTrue(0.1 * i < u and u <= 0.1 + 0.1 * i)

class TestPhysicalResource(unittest.TestCase):
    def test_physical_resource(self):
        p = PhysicalResource(3)
        self.assertEqual(p.pid,3)

class TestGraph(unittest.TestCase):
    def test_generate_digraph(self):
        g = Graph()
        g.generate_digraph(5, 3)
        self.assertEqual(len(g.physicals), 5)
        for p in g.physicals:
            self.assertTrue(len(p.edges) <= 2 * 3)
            for e in p.edges:
                self.assertTrue(g.physicals[e._to] and g.physicals[e._from])
                self.assertTrue(e._to > e._from)

    def test_sub_graph(self):
        g = Graph()
        g.generate_digraph(10, 4)
        #g.print_digraph()
        sub_phys, sub_edges = g.get_sub_graph(0.5)
        for e in sub_edges:
            self.assertTrue(g.physicals[e._from] in sub_phys \
                    and g.physicals[e._to] in sub_phys)

    def test_sample_crps(self):
        g = Graph()
        g.generate_digraph(10, 4)
        g.sample_crps(2, 0.5)
        for p in g.physicals:
            for rp in p.rps:
                if rp.idn > 0:
                    self.assertTrue(rp in g.crps[rp.idn - 1].rps)

        for crp in g.crps:
            for e in crp.sub_edges:
                rp_from = None
                rp_to = None
                for rp in g.physicals[e._from].rps:
                    if rp.idn == crp.cid:
                        rp_from = rp
                for rp in g.physicals[e._to].rps:
                    if rp.idn == crp.cid:
                        rp_to = rp
                self.assertTrue(rp_from and rp_to)
                self.assertTrue(rp_to in rp_from._to and rp_from in rp_to._from)

    def test_configure_p(self):
        # testing the rule
        # 1. p1 * s1 > s2
        # 2.  first physical with largest size
        g = Graph()
        g.generate_digraph(5, 4)
        g.sample_crps(3, 0.5)
        g.configure_physicals([1, 2, 4, 8, 16, 32])
        g.configure_rps([2, 4, 8, 16, 32, 64, 128], 1, 3, 5)

    def test_configure_rps(self):
        g = Graph()
        g.generate_digraph(10, 4)
        g.sample_crps(3, 0.5)
        g.configure_physicals([2, 4, 8, 16, 32])
        while g.configure_rps([2, 4, 8, 16, 32, 64, 128], 1, 3, 5) != _SUCCESS:
            continue
        
        for p in g.physicals:
            ubef = 0
            uaft = 0
            for rp in p.rps:
                ubef += 1.0/rp.period[0]
                uaft += 1.0/rp.period[1]
            self.assertTrue(ubef <= 0.2)
            self.assertTrue(0.3 < uaft and uaft <= 0.4)
            self.assertTrue(validate_regular(p.rps, 0))

        g = Graph()
        g.generate_digraph(10, 4)
        g.sample_crps(3, 0.5)
        g.configure_physicals([2])
        self.assertTrue(g.configure_rps([2, 4, 8, 16, 32, 64, 128], 1, 3, 5) == 0)


class TestAlgorithm(unittest.TestCase):
    def test_init_allowed(self):
        #p1->p3 (small to large)
        #p2->p3 (large to small)
        #p3->p4 (small to large)
        #p3->p5 (large to small)
        p1 = Partition(1, 0, 4, 4, 1, 1)
        p2 = Partition(2, 3, 4, 8, 1, 1)
        p3 = Partition(3, 2, 4, 4, 1, 1)
        p4 = Partition(4, 4, 4, 4, 1, 1)
        p5 = Partition(5, 1, 4, 2, 1, 1)
        p1._to.append(p3)
        p2._to.append(p3)
        p3._to.append(p4)
        p3._to.append(p5)
        p3._from.append(p1)
        p3._from.append(p2)
        p4._from.append(p3)
        p5._from.append(p3)
        p1.slice_size = 2
        p2.slice_size = 8
        p3.slice_size = 4
        p4.slice_size = 16
        p5.slice_size = 2
        p1.allowed[TRAN].append(2)
        p2.allowed[TRAN].append(1)
        p2.allowed[TRAN].append(2)
        init_allowed(p3, TRAN, LC)
        self.assertTrue(p3.allowed[TRAN] == [0, 2])
        init_allowed(p4, TRAN, LC)
        self.assertTrue(p4.allowed[TRAN] == [])
        init_allowed(p5, TRAN, LC)
        self.assertTrue(p5.allowed[TRAN] == [0, 1])

    def test_dsedf(self):
        #p1->p3 (small to large)
        #p2->p3 (large to small)
        #p3->p4 (small to large)
        #p3->p5 (large to small)
        p1 = Partition(1, 0, 4, 4, 1, 1)
        p2 = Partition(2, 3, 4, 8, 1, 1)
        p3 = Partition(3, 2, 4, 4, 1, 1)
        p4 = Partition(4, 4, 4, 4, 1, 1)
        p5 = Partition(5, 1, 4, 2, 1, 1)
        p1._to.append(p3)
        p2._to.append(p3)
        p3._to.append(p4)
        p3._to.append(p5)
        p3._from.append(p1)
        p3._from.append(p2)
        p4._from.append(p3)
        p5._from.append(p3)
        p1.slice_size = 2
        p2.slice_size = 8
        p3.slice_size = 4
        p4.slice_size = 16
        p5.slice_size = 2
        p1.allowed[TRAN].append(2)
        p2.allowed[TRAN].append(1)
        p2.allowed[TRAN].append(2)
        m = [0, 0, 4, 4]
        p3.r = 0
        p3.e = 4
        init_allowed(p3, TRAN, LC)
        l = dsedf(p3, m, p3.period[1], LC, TRAN)
        self.assertTrue(p3.allowed[TRAN] == [0, 2] and l == 0)
        m = [0, 0, 4, 4, 0, 4, 5 ,6, 7]
        p4.r = 0
        p4.e = 4
        init_allowed(p4, TRAN, LC)
        l = dsedf(p4, m, p4.period[1], LC, TRAN)
        self.assertTrue(p4.allowed[TRAN] == [] and not l)
        m = [0, 0, 0, 4, 5, 0]
        p5.r = 0
        p5.e = 2
        init_allowed(p5, TRAN, LC)
        l = dsedf(p5, m, p5.period[1], LC, TRAN)
        self.assertTrue(p5.allowed[TRAN] == [0, 1] and l == 1)

    def test_allowed_alg(self):
        p = Partition(5, 1, 4, 4, 1, 1)
        p.allowed[TRAN] = [3, 4]
        allowed_alg(UNIFORM, p, 7)
        self.assertTrue(p.allowed[TRAN] == [3, 4])

        p1 = Partition(1, 1, 4, 8, 1, 1)
        p2 = Partition(2, 1, 4, 8, 1, 1)
        p3 = Partition(3, 1, 4, 8, 1, 1)
        p3._to.append(p1)
        p3._to.append(p2)
        p1.slice_size = 2
        p2.slice_size = 4
        p3.slice_size = 1
        p3.allowed[TRAN] = [1, 2, 5, 6, 7]
        allowed_alg(LC, p3, 14)
        self.assertTrue(p3.allowed[TRAN] == [6, 7])

    def test_validate_regular(self):
        # self, idn, offset, period[0], period[1], rc, rcflag
        M = []
        M.append(Partition(1, 0, 2, 2, 1, 1))
        M.append(Partition(1, 1, 2, 2, 1, 1))
        self.assertFalse(validate_regular(M, 0))
        M = []
        M.append(Partition(1, 0, 2, 2, 1, 1))
        M.append(Partition(2, 2, 4, 2, 1, 1))
        self.assertFalse(validate_regular(M, 0))
        M = []
        M.append(Partition(1, 0, 4, 4, 1, 1))
        M.append(Partition(2, 2, 4, 2, 1, 1))
        M.append(Partition(3, 2, 4, 2, 1, 0))
        M.append(Partition(4, 2, 4, 2, 1, 0))
        self.assertTrue(validate_regular(M, 0))
        M = []
        M.append(Partition(1, 0, 4, 4, 1, 1))
        M.append(Partition(2, 2, 4, 2, 1, 1))
        self.assertFalse(validate_regular(M, 1))
        M = []
        M.append(Partition(1, 0, 4, 2, 1, 1))
        M.append(Partition(2, 2, 4, 4, 1, 1))
        self.assertFalse(validate_regular(M, 1))
        M = []
        M.append(Partition(1, 0, 4, 2, 1, 1))
        M.append(Partition(2, 2, 4, 4, 1, 0))
        self.assertFalse(validate_regular(M, 1))
        M = []
        M.append(Partition(1, 0, 4, 4, 1, 1))
        M.append(Partition(2, 2, 4, 4, 1, 1))
        self.assertTrue(validate_regular(M, 1))

    def test_validate_regular(self):
        M = []
        M.append(Partition(1, 0, 4, 2, 2, 1))
        M.append(Partition(2, 1, 4, 4, 1, 1))
        self.assertTrue(validate_rc_regular(M, 6, [0, 2, 1, 0, 1, 2, 1]))
        self.assertFalse(validate_rc_regular(M, 6, [2, 1, 0, 0, 1, 2, 1]))
        M[0].rc = 1
        self.assertFalse(validate_rc_regular(M, 6, [2, 1, 0, 0, 1, 2, 1]))
        self.assertTrue(validate_rc_regular(M, 6, [1, 2, 1, 0, 1, 2, 1]))
        self.assertFalse(validate_rc_regular(M, 6, [0, 1, 2, 0, 1, 2, 1]))


    def test_Alg2(self):
        M = []
        # new partition
        M.append(Partition(1, 1, 4, 4, 2, 0))
        # rc partition
        M.append(Partition(2, 1, 4, 8, 2, 1))
        # unchanged partition
        M.append(Partition(3, 2, 4, 4, 1, 1))
        # corner case partition
        M.append(Partition(4, 3, 4, 8, 2, 1))
        # corner case partition
        M.append(Partition(5, 7, 8, 8, 2, 1))

        Alg2(M,6, UNIFORM)
        self.assertTrue(M[0].d == 0 and M[0].e == 8 and M[0].r == 0)
        self.assertTrue(M[1].d == 0 and M[1].e == 16 and M[1].r == 0)
        self.assertTrue(M[2].d == -0.75 and M[2].e == 1 and M[2].r == 0)
        self.assertTrue(M[3].d == -0.5 and M[3].e == 12 and M[3].r == 0)
        self.assertTrue(M[4].d == -0.75 and M[4].e == 10 and M[4].r == 0)

    def test_Alg3(self):
        M = []
        M.append(Partition(1, 1, 4, 4, 2, 0))
        M.append(Partition(2, 1, 4, 8, 2, 1))
        M.append(Partition(3, 2, 4, 4, 1, 1))
        M.append(Partition(4, 3, 8, 4, 2, 1))
        reset_task(M, UNIFORM)
        Alg2(M, 6, UNIFORM)
        test, E, m = Alg3(M, 2, 0, UNIFORM)
        self.assertTrue(m[0]== 3 and m[1] == 3)
        M.append(Partition(5, 3, 4, 4, 1, 0))
        # test recompute
        reset_task(M, UNIFORM)
        Alg2(M, 6, UNIFORM)
        for p in E:
            if p.idn == 1:
                self.assertTrue(p.e == 8)
            elif p.idn == 2:
                self.assertTrue(p.e == 16)
            elif p.idn == 3:
                self.assertTrue(p.e == 1)
            elif p.idn == 4:
                self.assertTrue(p.e == 7)
            elif p.idn == 5:
                self.assertTrue(p.e == 4)
        test, E, m = Alg3(M, 2, 0, UNIFORM)
        self.assertTrue(m[0]== 3 and m[1] == 5)
        for p in E:
            if p.idn == 1:
                self.assertTrue(p.e == 6)
            if p.idn == 2:
                self.assertTrue(p.e == 14)
            if p.idn == 3:
                self.assertTrue(p.e == 3)
            if p.idn == 4:
                self.assertTrue(p.e == 5)
            if p.idn == 5:
                self.assertTrue(p.e == 4)
        reset_task(M, UNIFORM)
        Alg2(M,6, UNIFORM)
        M[4].e = 1
        test, E, m = Alg3(M,2,0, UNIFORM)
        self.assertFalse(m)

    def test_Alg4(self):
        M = []
        M.append(Partition(1, 1, 4, 4, 2, 0))
        M.append(Partition(2, 1, 4, 8, 2, 1))
        M.append(Partition(3, 2, 4, 4, 1, 1))
        M.append(Partition(4, 3, 8, 4, 2, 1))
        reset_task(M, UNIFORM)
        Alg2(M, 6, UNIFORM)
        test, E, m = Alg3(M, 2, 0, UNIFORM)
        test, m2, offsetList = Alg4(E,0, UNIFORM)
        self.assertTrue(m2 == [0, 1, 4, 3, 2, 1, 4, 3])
        Alg2(M, 6, UNIFORM)
        test, E, m = Alg3(M,2,0, UNIFORM)
        E[0].e = 3
        E[1].e = 3
        E[2].e = 3
        E[3].e = 3
        test, m2, offsetList = Alg4(E,0, UNIFORM)
        self.assertFalse(m2)

        #LC algorithm
        M = []
        M.append(Partition(1, 1, 4, 4, 4, 0))
        M.append(Partition(2, 1, 4, 8, 8, 1))
        M.append(Partition(3, 1, 4, 16, 16, 1))
        M[0].allowed = [0, 1]
        M[0].r = 0
        M[0].e = 1
        M[1].allowed = [0, 1, 2]
        M[1].r = 0
        M[1].e = 3
        M[2].allowed = [0, 1, 2]
        M[2].r = 0
        M[2].e = 2
        [test, m, offsetList] = Alg4(M, 0, LC)
        # TODO fix this
        # self.assertTrue(m == [1, 3, 2, 0, 1, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0])

    def print_rp(self, M):
        for rp in M:
            print rp.idn, rp.r, rp.e

    def test_reconfig(self):
        M = []
        M.append(Partition(1, 1, 4, 4, 2, 0))
        M.append(Partition(2, 1, 4, 8, 2, 1))
        M.append(Partition(3, 3, 4, 4, 1, 1))
        M.append(Partition(4, 3, 8, 4, 2, 1))
        reset_task(M, UNIFORM)
        Alg2(M, 6, UNIFORM)
        test, E, m = Alg3(M, 2, 0, UNIFORM)
        self.assertTrue(m == [4, 3])
        test, m2, offsetList = Alg4(E, 0, UNIFORM)
        self.assertTrue(m2 == [0, 4, 1, 3, 2, 4, 1, 3])

        reset_task(M, UNIFORM)
        Alg2(M, 6, UNIFORM)
        test, E, m = Alg3(M, 2, 2, UNIFORM)
        self.assertTrue(not m)
        reset_task(M, UNIFORM)

        Alg2(M, 6, UNIFORM)
        test, E, m = Alg3(M, 2, 1, UNIFORM)
        self.assertTrue(m == [BLOCK, 3])
        test, m2, offsetList = Alg4(E, 0, UNIFORM)
        self.assertTrue(m2 == [0, 1, 4, 3, 2, 1, 4, 3])

        reset_task(M, UNIFORM)
        Alg2(M, 6, UNIFORM)
        test, E, m = Alg3(M, 1, 2, UNIFORM)
        self.assertTrue(m == [BLOCK])
        test, m2, offsetList = Alg4(E, 1, UNIFORM)
        self.assertTrue(not m2)

        M[2].offset[0] = 0
        reset_task(M, UNIFORM)
        Alg2(M, 6, UNIFORM)
        test, E, m = Alg3(M, 1, 2, UNIFORM)
        self.assertTrue(m == [BLOCK])
        test, m2, offsetList = Alg4(E, 1, UNIFORM)
        self.assertTrue(m2 == [BLOCK, 3, 1, 4, 2, 3, 1, 4])

        reset_task(M, UNIFORM)
        Alg2(M, 6, UNIFORM)
        test, m3, ol, lenl = Alg1(M, 6, 2, 2, UNIFORM)
        self.assertTrue(m3 == [BLOCK, BLOCK, 3, 0, 1, 4, 3, 2, 1, 4])

    def test_uniform(self):
        M = []
        M.append(Partition(1, 0, 8, 8, 1, 1))
        M.append(Partition(-2, 1, 8, 8, 1, 1))
        M.append(Partition(3, 2, 8, 8, 1, 1))
        M.append(Partition(-4, 3, 8, 8, 1, 1))
        M.append(Partition(-5, 0, 2, 2, 3, 0))
        test, m, ol, lenl = Alg1(M, 8, 2, 0, UNIFORM)
        self.assertFalse(m)
        test, m, ol, lenl = Alg1(M, 8, 3, 0, UNIFORM)
        self.assertTrue(m == [1, -2, 3, -4, -5, 3, -5, 1, -5, -2, -5])

        M = []
        M.append(Partition(1, 6, 8, 128, 1, 1))
        M.append(Partition(2, 3, 4, 64, 1, 1))
        M.append(Partition(3, 5, 8, 2, 1, 1))
        M.append(Partition(4, 4, 8, 64, 1, 1))
        M.append(Partition(5, 2, 8, 4, 1, 1))
        test, m, ol, lenl = Alg1(M, 64, 0, 0, UNIFORM)
        self.assertFalse(m)

if __name__ == '__main__':
    unittest.main()
