import numpy as np
from unittest import TestCase
import math

def cross_entropy_loss(p, q):
    return -np.sum(p * np.log(q))

p = [0, 0, 0, 1]
q = [0.45, 0.2, 0.02, 0.33]
cross_entropy_loss(p, q)  # 1.1086626245216111
type(cross_entropy_loss(p, q)) # <class 'numpy.float64'>

def cross_entropy_loss_v2(p, q):
    return -sum([p[i] * np.log(q[i]) for i in range(len(p))])

p = [0, 0, 0, 1]
q = [0.45, 0.2, 0.02, 0.33]
cross_entropy_loss_v2(p, q)  # 1.1086626245216111
type(cross_entropy_loss_v2(p, q))  # <class 'numpy.float64'>

def cross_entropy_loss_v3(p, q):
    return -sum([p[i] * math.log(q[i]) for i in range(len(p))])

p = [0, 0, 0, 1]
q = [0.45, 0.2, 0.02, 0.33]
cross_entropy_loss_v3(p, q) # 1.1086626245216111
type(cross_entropy_loss_v3(p, q)) # <class 'float'>

class LossFunctionTest(TestCase):
    def test_cross_entropy_loss(self):
        p = [0, 0, 0, 1]
        q = [0.45, 0.2, 0.02, 0.33]
        # Test values
        self.assertEqual(cross_entropy_loss(p, q), 1.1086626245216111)
        self.assertEqual(cross_entropy_loss_v2(p, q), 1.1086626245216111)
        self.assertEqual(cross_entropy_loss_v3(p, q), 1.1086626245216111)

        self.assertIsInstance(cross_entropy_loss(p, q), np.float64)
        self.assertIsInstance(cross_entropy_loss_v2(p, q), np.float64)
        self.assertIsInstance(cross_entropy_loss_v3(p, q), float)

LossFunctionTest().test_cross_entropy_loss()