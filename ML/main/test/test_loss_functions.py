import numpy as np
from unittest import TestCase
from main.src.loss_functions import cross_entropy_loss, cross_entropy_loss_v2, cross_entropy_loss_v3

class LossFunctionTest(TestCase):
    def test_cross_entropy_loss(self):
        p = [0, 0, 0, 1]
        q = [0.45, 0.2, 0.02, 0.33]
        # Test values
        self.assertEqual(cross_entropy_loss(p, q), 1.1086626245216111)
        self.assertEqual(cross_entropy_loss_v2(p, q), 1.1086626245216111)
        self.assertEqual(cross_entropy_loss_v3(p, q), 1.1086626245216111)

        # Test types
        self.assertIsInstance(cross_entropy_loss(p, q), np.float64)
        self.assertIsInstance(cross_entropy_loss_v2(p, q), np.float64)
        self.assertIsInstance(cross_entropy_loss_v3(p, q), float)

LossFunctionTest().test_cross_entropy_loss()