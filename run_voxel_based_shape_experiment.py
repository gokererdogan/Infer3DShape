"""
Inferring 3D Shape from 2D Images

This file contains the experiment script for running chains with VoxelBasedShape hypothesis on different inputs
and with different parameters.

Created on Oct 12, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import gmllib.experiment as exp

from run_chain import *


def run_voxel_based_shape_experiment(**kwargs):
    """This method runs the chain with a VoxelBasedShape hypothesis and given parameters.

    This method is intended to be used in an Experiment instance. This method prepares the necessary data and
    calls `run_chain`.

    Parameters:
        kwargs (dict): Keyword arguments are as follows
            input_file (str): mame of the data file containing the observed image
            data_folder (str): folder containing the data files
            results_folder (str):
            sampler (str): see `run_chain` function
            max_depth (int): maximum depth of hypothesis
            ll_variance (float): variance of the Gaussian likelihood
            max_pixel_value (float): maximum pixel intensity value
            change_viewpoint_variance (float): variance for the change viewpoint move
            scale_space_variance (float): variance for the scale space move
            burn_in (int): see `run_chain` function
            sample_count (int): see `run_chain` function
            best_sample_count (int): see `run_chain` function
            thinning_period (int): see `run_chain` function
            report_period (int): see `run_chain` function
            temperatures (list): see `run_chain` function

    Returns:
        dict: run results
    """
    try:
        input_file = kwargs['input_file']
        results_folder = kwargs['results_folder']
        data_folder = kwargs['data_folder']
        sampler = kwargs['sampler']
        max_depth = None
        if 'max_depth' in kwargs:
            max_depth = kwargs['max_depth']
        ll_variance = kwargs['ll_variance']
        max_pixel_value = kwargs['max_pixel_value']
        change_viewpoint_variance = kwargs['change_viewpoint_variance']
        scale_space_variance = kwargs['scale_space_variance']
        burn_in = kwargs['burn_in']
        sample_count = kwargs['sample_count']
        best_sample_count = kwargs['best_sample_count']
        thinning_period = kwargs['thinning_period']
        report_period = kwargs['report_period']
        temperatures = None
        if 'temperatures' in kwargs:
            temperatures = kwargs['temperatures']
    except KeyError as e:
        raise ValueError("All experiment parameters should be provided. Missing parameter {0:s}".format(e.message))

    import numpy as np

    import mcmclib.proposal as proposal
    import i3d_proposal
    import vision_forward_model as vfm

    # read the data file
    data = np.load("{0:s}/{1:s}_single_view.npy".format(data_folder, input_file))
    render_size = data.shape[1:]

    fwm = vfm.VisionForwardModel(render_size=render_size)

    shape_params = {'LL_VARIANCE': ll_variance, 'MAX_PIXEL_VALUE': max_pixel_value}

    kernel_params = {'CHANGE_VIEWPOINT_VARIANCE': change_viewpoint_variance,
                     'SCALE_SPACE_VARIANCE': scale_space_variance}

    import voxel_based_shape as vox

    moves = {'change_viewpoint': i3d_proposal.change_viewpoint_z,
             'voxel_flip_full_vs_empty': vox.voxel_based_shape_flip_full_vs_empty,
             'voxel_flip_partial_vs_full': vox.voxel_based_shape_flip_full_vs_partial,
             'voxel_flip_partial_vs_empty': vox.voxel_based_shape_flip_empty_vs_partial,
             'voxel_scale_space': vox.voxel_scale_space}

    viewpoint = [(np.sqrt(8.0), -45.0, 45.0)]
    hypothesis_class = 'VoxelBasedShape'
    if max_depth is None:
        h = vox.VoxelBasedShape(forward_model=fwm, viewpoint=viewpoint, params=shape_params)
    else:
        hypothesis_class = 'VoxelBasedShapeMaxD'
        h = vox.VoxelBasedShapeMaxD(forward_model=fwm, viewpoint=viewpoint, params=shape_params, max_depth=max_depth)
        kernel_params['MAX_DEPTH'] = max_depth

    # form the proposal
    kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)

    results = run_chain(name=input_file, sampler=sampler, initial_h=h, data=data, kernel=kernel, burn_in=burn_in,
                        thinning_period=thinning_period, sample_count=sample_count, best_sample_count=best_sample_count,
                        report_period=report_period,
                        results_folder="{0:s}/{1:s}".format(results_folder, hypothesis_class),
                        temperatures=temperatures)

    return results

if __name__ == "__main__":
    MAX_PIXEL_VALUE = 177.0  # this is usually 256.0 but in our case because of the lighting in our renders, it is lower

    experiment = exp.Experiment(name="TestVoxelBasedShape", experiment_method=run_voxel_based_shape_experiment,
                                sampler=['pt'], input_file=['test6'],
                                results_folder='./results',
                                data_folder='./data/',
                                max_depth=5,
                                max_pixel_value=MAX_PIXEL_VALUE,
                                ll_variance=[0.0001],
                                scale_space_variance=[0.0025],
                                change_viewpoint_variance=[10.0],
                                burn_in=5000, sample_count=10, best_sample_count=10, thinning_period=10000,
                                report_period=10000, temperatures=[[3.0, 1.5, 1.0]])

    experiment.run(parallel=True, num_processes=3)

    print(experiment.results)
    experiment.save('./results')
    experiment.append_csv('./results/TestVoxelBasedShape.csv')
