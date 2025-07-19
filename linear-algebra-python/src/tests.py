"""
Created on Mon Feb 26 15:40:07 2018

@author: Christian Bender
@license: MIT-license

This file contains the test-suite for the linear algebra library.
"""

import math
import unittest

from lib import *


class Test(unittest.TestCase):
    def test_component(self):
        """
        test for method component
        """
        x = Vector([1, 2, 3])
        self.assertEqual(x.component(0), 1)
        self.assertEqual(x.component(2), 3)
        try:
            y = Vector()
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_str(self):
        """
        test for toString() method
        """
        x = Vector([0, 0, 0, 0, 0, 1])
        self.assertEqual(x.__str__(), "(0,0,0,0,0,1)")

    def test_size(self):
        """
        test for size()-method
        """
        x = Vector([1, 2, 3, 4])
        self.assertEqual(x.size(), 4)

    def test_euclidLength(self):
        """
        test for the eulidean length
        """
        x = Vector([1, 2])
        self.assertAlmostEqual(x.eulidLength(), 2.236, 3)

    def test_add(self):
        """
        test for + operator
        """
        x = Vector([1, 2, 3])
        y = Vector([1, 1, 1])
        self.assertEqual((x + y).component(0), 2)
        self.assertEqual((x + y).component(1), 3)
        self.assertEqual((x + y).component(2), 4)

    def test_sub(self):
        """
        test for - operator
        """
        x = Vector([1, 2, 3])
        y = Vector([1, 1, 1])
        self.assertEqual((x - y).component(0), 0)
        self.assertEqual((x - y).component(1), 1)
        self.assertEqual((x - y).component(2), 2)

    def test_mul(self):
        """
        test for * operator
        """
        x = Vector([1, 2, 3])
        a = Vector([2, -1, 4])  # for test of dot-product
        b = Vector([1, -2, -1])
        self.assertEqual((x * 3.0).__str__(), "(3.0,6.0,9.0)")
        self.assertEqual((a * b), 0)

    def test_zeroVector(self):
        """
        test for the global function zeroVector(...)
        """
        self.assertTrue(zeroVector(10).__str__().count("0") == 10)

    def test_unitBasisVector(self):
        """
        test for the global function unitBasisVector(...)
        """
        self.assertEqual(unitBasisVector(3, 1).__str__(), "(0,1,0)")

    def test_axpy(self):
        """
        test for the global function axpy(...) (operation)
        """
        x = Vector([1, 2, 3])
        y = Vector([1, 0, 1])
        self.assertEqual(axpy(2, x, y).__str__(), "(3,4,7)")

    def test_copy(self):
        """
        test for the copy()-method
        """
        x = Vector([1, 0, 0, 0, 0, 0])
        y = x.copy()
        self.assertEqual(x.__str__(), y.__str__())

    def test_changeComponent(self):
        """
        test for the changeComponent(...)-method
        """
        x = Vector([1, 0, 0])
        x.changeComponent(0, 0)
        x.changeComponent(1, 1)
        self.assertEqual(x.__str__(), "(0,1,0)")

    def test_str_matrix(self):
        A = Matrix([[1, 2, 3], [2, 4, 5], [6, 7, 8]], 3, 3)
        self.assertEqual("|1,2,3|\n|2,4,5|\n|6,7,8|\n", A.__str__())

    def test__mul__matrix(self):
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3, 3)
        x = Vector([1, 2, 3])
        self.assertEqual("(14,32,50)", (A * x).__str__())
        self.assertEqual("|2,4,6|\n|8,10,12|\n|14,16,18|\n", (A * 2).__str__())

    def test_changeComponent_matrix(self):
        A = Matrix([[1, 2, 3], [2, 4, 5], [6, 7, 8]], 3, 3)
        A.changeComponent(0, 2, 5)
        self.assertEqual("|1,2,5|\n|2,4,5|\n|6,7,8|\n", A.__str__())

    def test_component_matrix(self):
        A = Matrix([[1, 2, 3], [2, 4, 5], [6, 7, 8]], 3, 3)
        self.assertEqual(7, A.component(2, 1), 0.01)

    def test__add__matrix(self):
        A = Matrix([[1, 2, 3], [2, 4, 5], [6, 7, 8]], 3, 3)
        B = Matrix([[1, 2, 7], [2, 4, 5], [6, 7, 10]], 3, 3)
        self.assertEqual("|2,4,10|\n|4,8,10|\n|12,14,18|\n", (A + B).__str__())

    def test__sub__matrix(self):
        A = Matrix([[1, 2, 3], [2, 4, 5], [6, 7, 8]], 3, 3)
        B = Matrix([[1, 2, 7], [2, 4, 5], [6, 7, 10]], 3, 3)
        self.assertEqual("|0,0,-4|\n|0,0,0|\n|0,0,-2|\n", (A - B).__str__())

    def test_squareZeroMatrix(self):
        self.assertEqual(
            "|0,0,0,0,0|\n|0,0,0,0,0|\n|0,0,0,0,0|\n|0,0,0,0,0|" + "\n|0,0,0,0,0|\n",
            squareZeroMatrix(5).__str__(),
        )

    def test_norm_vector(self):
        x = Vector([1, 2, 3])
        self.assertAlmostEqual(x.norm().component(0), (1 / math.sqrt(14)), 0.001)
        self.assertAlmostEqual(x.norm().component(1), math.sqrt(2.0 / 7), 0.001)

    def test__eq__vector(self):
        x = Vector([1, 2, 3])
        y = Vector([1, 0, 1])
        self.assertTrue(x == x)
        self.assertFalse(x == y)

    def test__eq__matrix(self):
        A = Matrix([[1, 2, 3], [2, 4, 5], [6, 7, 8]], 3, 3)
        B = Matrix([[1, 2, 7], [2, 4, 5], [6, 7, 10]], 3, 3)
        self.assertTrue(A == A)
        self.assertFalse(A == B)


if __name__ == "__main__":
    unittest.main()
