import unittest
from signal_detection import SignalDetection
from scipy.stats import norm
import matplotlib.pyplot as plt

class TestSignalDetection(unittest.TestCase):
    def setUp(self):
        self.obj = SignalDetection(100, 40, 30, 120)
    
    def test_hit_rate(self):
        self.assertEqual(self.obj.hit_rate(), 100 / 140)
    
    def test_false_alarm_rate(self):
        self.assertEqual(self.obj.false_alarm_rate(), 20 / 150)
    
    def test_d_prime(self):
        self.assertAlmostEqual(self.obj.d_prime(), 1.4075700555057773)
    
    def test_criterion(self): 
        self.assertAlmostEqual(self.obj.criterion(), 0.13783620582002554)
    
    def test_invalid_constructor_values(self):
        with self.assertRaises(ValueError):
            SignalDetection(-1, 10, 5, 5)
        with self.assertRaises(TypeError):
            SignalDetection(10, "6", 3, 12)
        with self.assertRaises(ValueError):
            SignalDetection(10, 6, 3, 12)
    
    def test_invalid_operator_argu(self):
        for bad in [10, None, "abc", [1, 2, 3]]:
            with self.assertRaises(TypeError):
                self.obj + bad
            with self.assertRaises(TypeError):
                self.obj - bad
        for bad in [-2, [10, 5, 4, 14]]:
            with self.assertRaises(ValueError):
                self.obj * bad
    
    def test_add_operation(self):
        result = self.obj + self.obj
        self.assertEqual(result.hits, 200)
        self.assertEqual(result.misses, 80)
        self.assertEqual(result.false_alarms, 60)
        self.assertEqual(result.correct_rejections, 240)
    
    def test_sub_operation(self):
        result = self.obj - SignalDetection(10, 5, 5, 15)
        self.assertEqual(result.hits, 90)
        self.assertEqual(result.misses, 35)
        self.assertEqual(result.false_alarms, 25)
        self.assertEqual(result.correct_rejections, 105)
    
    def test_mul_operation(self):
        result = self.obj * 2
        self.assertEqual(result.hits, 200)
        self.assertEqual(result.misses, 80)
        self.assertEqual(result.false_alarms, 60)
        self.assertEqual(result.correct_rejections, 240)
    
    def test_plot_sdt(self):
        fig, ax = self.obj.plot_sdt()
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_xlabel(), "Response")
        texts = [t.get_text() for t in ax.texts]
        self.assertIn("d'", texts)
    

      
if __name__ == '__main__':
    unittest.main()