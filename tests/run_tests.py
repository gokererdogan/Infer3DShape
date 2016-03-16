"""
Inferring 3D Shape from 2D Images

Script for running tests. Enables the user to skip sampling tests and run only the selected test modules.
    Run using
        python run_tests module1 module2 ... --nosampling

Created on Mar 16, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""

import unittest
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Infer3DShape unittests.")
    parser.add_argument('modules', type=str, nargs='+', help='Test module names to run. If discover, '
                                                             'uses unittest.discover to find tests in '
                                                             'the current folder.')
    # note that --nosampling parameter seems to have no effect here but it is checked in TestCase classes using
    # unittest skipIf decorator.
    parser.add_argument('--nosampling', action='store_true', help='Do not run sampling tests.')

    args = parser.parse_args()

    loader = unittest.TestLoader()
    if 'discover' in args.modules:
        tests = loader.discover('./')
    else:
        tests = loader.loadTestsFromNames(args.modules)

    unittest.TextTestRunner(verbosity=2).run(tests)
