"""
Inferring 3D Shape from 2D Images

Unit tests for run_experiment method.

Created on Dec 9, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import os
import time
import unittest

from Infer3DShape.run_experiment import *

class RunExperimentTest(unittest.TestCase):
    def test_run_experiment(self):
        # test:
        #   raises Exception with missing or wrong parameter values
        #   runs the right hypothesis class
        #   runs with correct number of viewpoints
        #   correct ll_variance, and other parameters
        #   saves correct render files
        #   saves the correct run results (pickle)
        self.assertRaises(ValueError, run_chain)
        params = {'input_file': 'test1', 'results_folder': '.', 'data_folder': '../data', 'hypothesis_class': 'XXX',
                  'single_view': True, 'render_size': (200, 200), 'max_part_count': 10, 'max_depth': 10,
                  'add_part_prob': 0.6, 'll_variance': 1.0,
                  'max_pixel_value': 175.0, 'change_size_variance': 0.1, 'change_viewpoint_variance': 0.1,
                  'move_part_variance': 0.1, 'move_object_variance': 0.1, 'burn_in': 0, 'sample_count': 1,
                  'best_sample_count': 1, 'thinning_period': 10, 'report_period': 10}

        # wrong hypothesis class
        self.assertRaises(ValueError, run_chain, kwargs=params)

        params['hypothesis_class'] = 'Shape'
        results = run_chain(**params)
        self.assertIn('run_id', results.keys())
        self.assertIn('run_file', results.keys())
        self.assertIn('mean_acceptance_rate', results.keys())
        self.assertIn('start_time', results.keys())
        self.assertIn('end_time', results.keys())
        self.assertIn('duration', results.keys())
        self.assertIn('best_ll', results.keys())
        self.assertIn('best_posterior', results.keys())
        self.assertIn('mse', results.keys())
        self.assertIn('mean_best_ll', results.keys())
        self.assertIn('mean_best_posterior', results.keys())
        self.assertIn('mse_mean', results.keys())
        self.assertIn('mean_sample_posterior', results.keys())
        self.assertIn('mean_sample_ll', results.keys())
        self.assertIn('mse_sample', results.keys())

        # saved the right files
        start = results['start_time']
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(start))
        fname = "{0:s}/{1:s}/{2:s}_{3:s}_{4:06d}.pkl".format(params['results_folder'], params['hypothesis_class'],
                                                             params['input_file'], time_str, results['run_id'])
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

        fname = "{0:s}/{1:s}/{2:s}/s{3:d}_0.png".format(params['results_folder'], params['hypothesis_class'],
                                                        params['input_file'], 0)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

        fname = "{0:s}/{1:s}/{2:s}/b{3:d}_0.png".format(params['results_folder'], params['hypothesis_class'],
                                                        params['input_file'], 0)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

        folder = "{0:s}/{1:s}/{2:s}".format(params['results_folder'], params['hypothesis_class'], params['input_file'])
        os.rmdir(folder)
        folder = "{0:s}/{1:s}".format(params['results_folder'], params['hypothesis_class'])
        os.rmdir(folder)

