#!/usr/bin/python
import unittest
from reconfig import *

class TestRun(unittest.TestCase):
    def test_period_combo(self):
        # if unique
        test_list = Period_Combo(10,1,64)
        for i, l in enumerate(test_list):
            self.assertTrue(len(set(l)) == len(l))

    def test_uniform_configure(self):
        sch = Schedule()
        sch.sample_uniform(5, [2, 4, 8, 16, 32, 64, 128], 7, 7, 5)
        for p in sch.gph.physicals:
            for rp in p.rps:
                if rp.idn > 0:
                    self.assertTrue(rp in sch.gph.crps[rp.idn - 1].rps)
# da faq?
#         for i in range(0, 100):
#             sch.sample_uniform(10, [2, 4, 8, 16, 32, 64, 128], 8, 8, 1)
#             [test, schedule_u, offsets1, size1] = Alg1(sch.gph.physicals[0].rps,\
#                     64, 0, 0, UNIFORM)
#             if not schedule_u:
#                 print "no"

    def test_nonuniform_configure(self):
        sch = Schedule()
        sch.sample_non_uniform(6, 3, [2, 4, 8, 16, 32], 3, 0.4, [2, 4, 8, 16, 32, 64], 7, 7, 5)
        for p in sch.gph.physicals:
            for rp in p.rps:
                if rp.idn > 0:
                    self.assertTrue(rp in sch.gph.crps[rp.idn - 1].rps)


if __name__ == '__main__':
    unittest.main()
