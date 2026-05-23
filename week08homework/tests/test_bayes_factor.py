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
        self.assertIsInstance(self.obj.evidence_spike(0.4, 0.5), float)
        self.assertIsInstance(self.obj.bayes_factor(0, 1), float)
    
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
        self.assertTrue(self.obj.evidence_spike(0.4, 0.5) > 0)
        self.assertAlmostEqual(
            self.obj.evidence_spike(0, 1), 
            1/(self.obj.n + 1)
        )
    
    def test_bayes_factor(self):
        with self.assertRaises(ValueError):
            self.obj.bayes_factor(0.48, 0.48)
        self.assertAlmostEqual(
            self.obj.bayes_factor(0, 1), 
            1.0
        )
        test_obj_1 = BayesFactor(0, 0)
        self.assertAlmostEqual(
            test_obj_1.bayes_factor(0.48, 0.52), 
            1.0
        )
        test_obj_2 = BayesFactor(5, 0)
        self.assertAlmostEqual(
            test_obj_2.bayes_factor(0.48, 0.52), 
            0.18850048000000047
        )
        # a failing test because evidence_slab is too small and leads to a division by zero error
        test_obj_3 = BayesFactor(int(1e8), 5)
        self.assertAlmostEqual(
            test_obj_3.bayes_factor(0.48, 0.52), 
            test_obj_3.evidence_spike(0.48, 0.52) / test_obj_3.evidence_slab()
        )
        # intentionally failing test
        # self.assertAlmostEqual(self.obj.bayes_factor(0, 1), 5.0)
    
    def tearDown(self):
        del self.obj
    
    if __name__ == '__main__':
        unittest.main()
        