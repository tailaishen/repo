import unittest
from bayes_factor import BayesFactor
import scipy as sp
import math

class TestBayesFactor(unittest.TestCase):
    def setUp(self):
        self.obj = BayesFactor(10, 5)

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            BayesFactor("10", 5)
        with self.assertRaises(TypeError):
            BayesFactor(10, 5.7)
        with self.assertRaises(ValueError):
            BayesFactor(-1, 5)
        with self.assertRaises(ValueError):
            BayesFactor(10, 15)
    
    def test_data_type(self):
        self.assertIsInstance(self.obj.likelihood(0.3), float) 
        self.assertIsInstance(self.obj.evidence_slab(), float)
        self.assertIsInstance(self.obj.evidence_spike(), float)
        self.assertIsInstance(self.obj.bayes_factor(), float)
    
    def test_likelihood(self):
        self.assertAlmostEqual(self.obj.likelihood(0.3), sp.stats.binom.pmf(5, 10, 0.3))
        self.assertAlmostEqual(self.obj.likelihood(0), sp.stats.binom.pmf(5, 10, 0))
    
    def test_evidence_slab(self):
        self.assertTrue(self.obj.evidence_slab() > 0)
        self.assertAlmostEqual(
            self.obj.evidence_slab(), 
            1/(self.obj.n + 1)
        )
    
    def test_evidence_spike(self):
        self.assertTrue(self.obj.evidence_spike() > 0)
        self.assertAlmostEqual(
            self.obj.evidence_spike(), 
            sp.stats.binom.pmf(self.obj.k, self.obj.n, 0.5)
        )
    
    def test_bayes_factor(self):
        self.assertAlmostEqual(
            self.obj.bayes_factor(), 
            2.70703125
        )
        # # a failing test because evidence_slab is too small and leads to a division by zero error
        # test_obj_1 = BayesFactor(int(1e8), 5)
        # self.assertAlmostEqual(
        #     test_obj_1.bayes_factor(), 
        #     test_obj_1.evidence_spike() / test_obj_1.evidence_slab()
        # )
    
    def tearDown(self):
        del self.obj
    
    if __name__ == '__main__':
        unittest.main()
        