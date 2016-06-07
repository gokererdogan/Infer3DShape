"""
Inferring 3D Shape from 2D Images

This file contains the experiment script for running the chains with BDAoOSSShape hypothesis on different inputs
and with different parameters.

Created on Oct 12, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import gmllib.experiment as exp

from run_chain import *


def run_bdaooss_experiment(**kwargs):
    """This method runs the chain with a BDAoOSSShape hypothesis and given parameters.

    This method is intended to be used in an Experiment instance. This method prepares the necessary data and
    calls `run_chain`.

    Parameters:
        kwargs (dict): Keyword arguments are as follows
            input_file (str): mame of the data file containing the observed image
            data_folder (str): folder containing the data files
            results_folder (str):
            sampler (str): see `run_chain` function
            inverted (bool): whether the input image is inverted, i.e., upside down
            max_depth (int): maximum depth of the hypothesis trees
            ll_variance (float): variance of the Gaussian likelihood
            max_pixel_value (float): maximum pixel intensity value
            change_size_variance (float): variance for the change part size move
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
        inverted = kwargs['inverted']
        ll_variance = kwargs['ll_variance']
        max_pixel_value = kwargs['max_pixel_value']
        max_depth = None
        if 'max_depth' in kwargs:
            max_depth = kwargs['max_depth']
        change_size_variance = kwargs['change_size_variance']
        change_viewpoint_variance = kwargs['change_viewpoint_variance']
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
    if not inverted:
        viewpoint = [[np.sqrt(8.0), -45.0, 45.0]]
        data = np.load("{0:s}/{1:s}_single_view.npy".format(data_folder, input_file))
        custom_lighting = True
    else:
        data = np.load("{0:s}/{1:s}_single_view_inverted.npy".format(data_folder, input_file))
        viewpoint = [[np.sqrt(8.0), -45.0, 135.0]]
        # for inverted images we use VTK's default lighting because in the custom lighting, bottoms of objects are not
        # well illuminated
        custom_lighting = False

    render_size = data.shape[1:]
    fwm = vfm.VisionForwardModel(render_size=render_size, custom_lighting=custom_lighting, offscreen_rendering=True)

    shape_params = {'LL_VARIANCE': ll_variance, 'MAX_PIXEL_VALUE': max_pixel_value}

    kernel_params = {'CHANGE_SIZE_VARIANCE': change_size_variance,
                     'CHANGE_VIEWPOINT_VARIANCE': change_viewpoint_variance}

    import bdaooss_shape as bdaooss

    moves = {'change_viewpoint': i3d_proposal.change_viewpoint_z,
             'bdaooss_add_remove_part': bdaooss.bdaooss_add_remove_part,
             'bdaooss_change_part_size_local': bdaooss.bdaooss_change_part_size_local,
             'bdaooss_change_part_dock_face': bdaooss.bdaooss_change_part_dock_face}

    if max_depth is None:
        hypothesis_class = 'BDAoOSShape'
        h = bdaooss.BDAoOSSShape(forward_model=fwm, viewpoint=viewpoint, params=shape_params)

    else:
        import bdaooss_shape_maxd as bdaooss_maxd
        hypothesis_class = 'BDAoOSSShapeMaxD'
        h = bdaooss_maxd.BDAoOSSShapeMaxD(forward_model=fwm, max_depth=max_depth, viewpoint=viewpoint,
                                          params=shape_params)
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
    MAX_PIXEL_VALUE = 180.0  # this is slightly higher for inverted images because of lighting.

    experiment = exp.Experiment(name="BDAoOSShapeInverted", experiment_method=run_bdaooss_experiment,
                                sampler=['mh'],
                                input_file=['o8_t1_cs_d1', 'o8_t1_cs_d2', 'o8_t2_ap_d1', 'o8_t2_ap_d2', 
                                            'o8_t2_rp_d1', 'o8_t2_rp_d2', 'o8_t2_mf_d1', 'o8_t2_mf_d2'],
                                results_folder='./results',
                                data_folder='./data/stimuli20150624_144833',
                                inverted=True,  # whether the input image is inverted
                                max_depth=5,
                                max_pixel_value=MAX_PIXEL_VALUE,
                                ll_variance=[0.0001],
                                change_size_variance=[0.00005],
                                change_viewpoint_variance=[10.0],
                                burn_in=5000, sample_count=10, best_sample_count=10, thinning_period=10000,
                                report_period=10000)

    experiment.run(parallel=True, num_processes=-1)

    print(experiment.results)
    experiment.save('./results')
    experiment.append_csv('./results/BDAoOSShapeInverted.csv')

    """
    input_file=['o1_t1_cs_d2', 'o1_t2_mf_d2', 'o3_t2_mf_d1',
                'o3_t2_mf_d2', 'o10_t1_cs_d1', 'o10_t2_mf_d2'],
    """
