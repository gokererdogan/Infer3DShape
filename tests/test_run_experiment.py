"""
Inferring 3D Shape from 2D Images

Unit tests for run_experiment method.

Created on Dec 9, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import os
import time

import mcmclib.proposal as proposal
import Infer3DShape.shape as shape
import Infer3DShape.vision_forward_model as vfm

from i3d_test_case import *
from Infer3DShape.run_chain import *


class RunExperimentTest(I3DTestCase):
    def test_run_chain(self):
        fwm = vfm.VisionForwardModel(render_size=(50, 50))
        s = shape.Shape(forward_model=fwm, viewpoint=[(3.0, 45.0, 45.0)],
                        params={'LL_VARIANCE': 1.0, 'MAX_PIXEL_VALUE': 175.0})

        data = np.zeros((1, 50, 50))
        kernel = proposal.RandomMixtureProposal(moves={'aaa': shape.shape_change_part_size_local},
                                                params={'CHANGE_SIZE_VARIANCE': 1.0})

        params = {'name': 'unittest', 'results_folder': '.', 'sampler': 'xxx', 'burn_in': 0, 'sample_count': 1,
                  'best_sample_count': 1, 'thinning_period': 10, 'data': data, 'kernel': kernel, 'initial_h': s,
                  'report_period': 10}

        # wrong sampler
        self.assertRaises(ValueError, run_chain, **params)

        # need to supply temperatures if sampler is pt
        params['sampler'] = 'pt'
        self.assertRaises(ValueError, run_chain, **params)

        params['temperatures'] = [2.0, 1.0]
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
        fname = "{0:s}/{1:s}_{2:s}_{3:06d}.pkl".format(params['results_folder'], params['name'], time_str,
                                                       results['run_id'])
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

        fname = "{0:s}/{1:s}/s{2:d}_0.png".format(params['results_folder'], params['name'], 0)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

        fname = "{0:s}/{1:s}/b{2:d}_0.png".format(params['results_folder'], params['name'], 0)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

        folder = "{0:s}/{1:s}".format(params['results_folder'], params['name'])
        os.rmdir(folder)

    def test_run_chain_mh(self):
        fwm = vfm.VisionForwardModel(render_size=(50, 50))
        s = shape.Shape(forward_model=fwm, viewpoint=[(3.0, 45.0, 45.0)],
                        params={'LL_VARIANCE': 1.0, 'MAX_PIXEL_VALUE': 175.0})

        data = np.zeros((1, 50, 50))
        kernel = proposal.RandomMixtureProposal(moves={'aaa': shape.shape_change_part_size_local},
                                                params={'CHANGE_SIZE_VARIANCE': 1.0})

        params = {'name': 'unittest', 'results_folder': '.', 'sampler': 'mh', 'burn_in': 0, 'sample_count': 1,
                  'best_sample_count': 1, 'thinning_period': 10, 'data': data, 'kernel': kernel, 'initial_h': s,
                  'report_period': 10}

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
        fname = "{0:s}/{1:s}_{2:s}_{3:06d}.pkl".format(params['results_folder'], params['name'], time_str,
                                                       results['run_id'])
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

        fname = "{0:s}/{1:s}/s{2:d}_0.png".format(params['results_folder'], params['name'], 0)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

        fname = "{0:s}/{1:s}/b{2:d}_0.png".format(params['results_folder'], params['name'], 0)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

        folder = "{0:s}/{1:s}".format(params['results_folder'], params['name'])
        os.rmdir(folder)

