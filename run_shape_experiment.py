"""
Inferring 3D Shape from 2D Images

This file contains the experiment script for running the chains with Shape hypothesis on different inputs and
with different parameters.

Created on Oct 12, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import gmllib.experiment as exp

from run_chain import *


def run_shape_experiment(**kwargs):
    """This method runs the chain with a Shape hypothesis and given parameters.

    This method is intended to be used in an Experiment instance. This method prepares the necessary data and
    calls `run_chain`.

    Parameters:
        kwargs (dict): Keyword arguments are as follows
            input_file (str): mame of the data file containing the observed image
            data_folder (str): folder containing the data files
            results_folder (str):
            sampler (str): see `run_chain` function
            max_part_count (int): maximum number of parts in the shape hypothesis
            ll_variance (float): variance of the Gaussian likelihood
            max_pixel_value (float): maximum pixel intensity value
            change_size_variance (float): variance for the change part size move
            move_part_variance (float): variance for the move part move
            change_viewpoint_variance (float): variance for the change viewpoint move
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
        max_part_count = None
        if 'max_part_count':
            max_part_count = kwargs['max_part_count']
        ll_variance = kwargs['ll_variance']
        max_pixel_value = kwargs['max_pixel_value']
        change_size_variance = kwargs['change_size_variance']
        change_viewpoint_variance = kwargs['change_viewpoint_variance']
        move_part_variance = kwargs['move_part_variance']
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

    kernel_params = {'CHANGE_SIZE_VARIANCE': change_size_variance, 'MOVE_PART_VARIANCE': move_part_variance,
                     'CHANGE_VIEWPOINT_VARIANCE': change_viewpoint_variance}

    import shape

    moves = {'change_viewpoint': i3d_proposal.change_viewpoint_z,
             'shape_add_remove_part': shape.shape_add_remove_part,
             'shape_move_part_local': shape.shape_move_part_local,
             'shape_change_part_size_local': shape.shape_change_part_size_local}

    viewpoint = [(np.sqrt(8.0), -45.0, 45.0)]
    hypothesis_class = 'Shape'
    if max_part_count is None:
        h = shape.Shape(forward_model=fwm, viewpoint=viewpoint, params=shape_params)
    else:
        import shape_maxn
        hypothesis_class = 'ShapeMaxD'
        h = shape_maxn.ShapeMaxN(forward_model=fwm, maxn=max_part_count, viewpoint=viewpoint, params=shape_params)
        kernel_params['MAX_PART_COUNT'] = max_part_count

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

    experiment = exp.Experiment(name="TestShape", experiment_method=run_shape_experiment,
                                sampler=['mh'], input_file=['test2'],
                                results_folder='./results',
                                data_folder='./data/',
                                max_part_count=10,
                                max_pixel_value=MAX_PIXEL_VALUE,
                                ll_variance=[0.0001],
                                change_size_variance=[0.00005],
                                move_part_variance=[0.00005],
                                change_viewpoint_variance=[10.0],
                                burn_in=5000, sample_count=10, best_sample_count=10, thinning_period=10000,
                                report_period=10000)

    experiment.run(parallel=True, num_processes=3)

    print(experiment.results)
    experiment.save('./results')
    experiment.append_csv('./results/TestShape.csv')
