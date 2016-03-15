"""
Inferring 3D Shape from 2D Images

Custom unit test case class for Infer3DShape.

Created on Mar 13, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""
import unittest
import numpy as np


class I3DTestCase(unittest.TestCase):
    def assertAlmostIn(self, val, arr):
        found = False
        for comp in arr:
            try:
                self.assertAlmostEqual(val, comp)
            except AssertionError:
                pass
            else:
                found = True
                break
        if not found:
            raise AssertionError("Value cannot be found in array.")

    def assertNumpyArrayEqual(self, arr1, arr2):
        if not np.allclose(arr1, arr2):
            raise AssertionError("Numpy arrays are not equal: {0:s} - {1:s}".format(arr1, arr2))

    def assertNumpyArrayNotEqual(self, arr1, arr2):
        if np.allclose(arr1, arr2):
            raise AssertionError("Numpy arrays are equal: {0:s} - {1:s}".format(arr1, arr2))

    def assertNumpyArrayListEqual(self, l1, l2):
        if len(l1) != len(l2):
            raise AssertionError("Lists {0:s} and {1:s} does not have the same number of elements.".format(l1, l2))
        for i1 in l1:
            found = False
            for i2 in l2:
                if np.allclose(i1, i2):
                    found = True
                    break
            if not found:
                raise AssertionError("Item {0:s} cannot be found in list {1:s}".format(i1, l2))

