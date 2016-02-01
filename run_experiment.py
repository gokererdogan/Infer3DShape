"""
Inferring 3D Shape from 2D Images

This file contains the experiment script for running the
chains on different inputs and with different parameters.

Created on Oct 12, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import gmllib.experiment as exp
import os
import warnings


def run_chain(**kwargs):
    """This method runs the chain with the given parameters, saves the results and returns a summary of the results.

    This method is intended to be used in an Experiment instance.

    Args:
        input_file:
        results_folder:
        data_folder:
        hypothesis_class:
        sampler:
        single_view:
        render_size:
        max_part_count:
        max_depth:
        add_part_prob:
        ll_variance:
        max_pixel_value:
        change_size_variance:
        change_viewpoint_variance:
        move_part_variance:
        move_object_variance:
        scale_space_variance:
        burn_in:
        sample_count:
        best_sample_count:
        thinning_period:
        report_period:
        temperatures:

    Returns:
        dictionary of run results
    """
    try:
        input_file = kwargs['input_file']
        results_folder = kwargs['results_folder']
        data_folder = kwargs['data_folder']
        hypothesis_class = kwargs['hypothesis_class']
        sampler = kwargs['sampler']
        single_view = kwargs['single_view']
        render_size = kwargs['render_size']
        max_part_count = kwargs['max_part_count']
        max_depth = kwargs['max_depth']
        add_part_prob = kwargs['add_part_prob']
        ll_variance = kwargs['ll_variance']
        max_pixel_value = kwargs['max_pixel_value']
        change_size_variance = kwargs['change_size_variance']
        change_viewpoint_variance = kwargs['change_viewpoint_variance']
        move_part_variance = kwargs['move_part_variance']
        move_object_variance = kwargs['move_object_variance']
        scale_space_variance = kwargs['scale_space_variance']
        burn_in = kwargs['burn_in']
        sample_count = kwargs['sample_count']
        best_sample_count = kwargs['best_sample_count']
        thinning_period = kwargs['thinning_period']
        report_period = kwargs['report_period']
        if sampler == 'pt':
            temperatures = kwargs['temperatures']
            chain_count = len(temperatures)
    except KeyError as e:
        raise ValueError("All experiment parameters should be provided. Missing parameter {0:s}".format(e.message))

    import time
    import mcmclib.proposal as proposal
    import vision_forward_model as vfm
    import numpy as np
    fwm = vfm.VisionForwardModel(render_size=render_size)

    shape_params = {'ADD_PART_PROB': add_part_prob, 'LL_VARIANCE': ll_variance, 'MAX_PIXEL_VALUE': max_pixel_value}
    kernel_params = {'CHANGE_SIZE_VARIANCE': change_size_variance, 'MOVE_PART_VARIANCE': move_part_variance,
                     'MOVE_OBJECT_VARIANCE': move_object_variance,
                     'CHANGE_VIEWPOINT_VARIANCE': change_viewpoint_variance,
                     'SCALE_SPACE_VARIANCE': scale_space_variance}

    moves = {}
    if single_view:
        import i3d_proposal
        viewpoint = [(np.sqrt(2.0), -np.sqrt(2.0), 2.0)]
        moves['change_viewpoint'] = i3d_proposal.change_viewpoint
    else:
        viewpoint = None

    if hypothesis_class == 'Shape':
        import shape
        h = shape.Shape(forward_model=fwm, viewpoint=viewpoint, params=shape_params)
        moves['shape_add_remove_part'] = shape.shape_add_remove_part
        # moves['shape_move_part'] = shape.shape_move_part
        moves['shape_move_part_local'] = shape.shape_move_part_local
        # moves['shape_change_part_size'] = shape.shape_change_part_size
        moves['shape_change_part_size_local'] = shape.shape_change_part_size_local
        # moves['shape_move_object'] = shape.shape_move_object
        kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)
    elif hypothesis_class == 'ShapeMaxN':
        import shape
        import shape_maxn
        h = shape_maxn.ShapeMaxN(forward_model=fwm, maxn=max_part_count, viewpoint=viewpoint, params=shape_params)
        moves['shape_add_remove_part'] = shape.shape_add_remove_part
        # moves['shape_move_part'] = shape.shape_move_part
        moves['shape_move_part_local'] = shape.shape_move_part_local
        # moves['shape_change_part_size'] = shape.shape_change_part_size
        moves['shape_change_part_size_local'] = shape.shape_change_part_size_local
        # moves['shape_move_object'] = shape.shape_move_object
        kernel_params['MAX_PART_COUNT'] = max_part_count
        kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)
    elif hypothesis_class == 'BDAoOSSShape':
        import bdaooss_shape as bdaooss
        h = bdaooss.BDAoOSSShape(forward_model=fwm, viewpoint=viewpoint, params=shape_params)
        moves['bdaooss_add_remove_part'] = bdaooss.bdaooss_add_remove_part
        # moves['bdaooss_change_part_size'] = bdaooss.bdaooss_change_part_size
        moves['bdaooss_change_part_size_local'] = bdaooss.bdaooss_change_part_size_local
        moves['bdaooss_change_part_dock_face'] = bdaooss.bdaooss_change_part_dock_face
        # moves['bdaooss_move_object'] = bdaooss.bdaooss_move_object
        kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)
    elif hypothesis_class == 'BDAoOSSShapeMaxD':
        import bdaooss_shape as bdaooss
        import bdaooss_shape_maxd as bdaooss_maxd
        h = bdaooss_maxd.BDAoOSSShapeMaxD(forward_model=fwm, max_depth=max_depth, viewpoint=viewpoint,
                                          params=shape_params)
        moves['bdaooss_add_remove_part'] = bdaooss.bdaooss_add_remove_part
        # moves['bdaooss_change_part_size'] = bdaooss.bdaooss_change_part_size
        moves['bdaooss_change_part_size_local'] = bdaooss.bdaooss_change_part_size_local
        moves['bdaooss_change_part_dock_face'] = bdaooss.bdaooss_change_part_dock_face
        # moves['bdaooss_move_object'] = bdaooss.bdaooss_move_object
        kernel_params['MAX_DEPTH'] = max_depth
        kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)
    elif hypothesis_class == 'VoxelBasedShape':
        import voxel_based_shape as vox
        h = vox.VoxelBasedShape(forward_model=fwm, viewpoint=viewpoint, params=shape_params)
        moves['voxel_flip_full_vs_empty'] = vox.voxel_based_shape_flip_full_vs_empty
        moves['voxel_flip_partial_vs_full'] = vox.voxel_based_shape_flip_full_vs_partial
        moves['voxel_flip_partial_vs_empty'] = vox.voxel_based_shape_flip_empty_vs_partial
        moves['voxel_scale_space'] = vox.voxel_scale_space
        kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)
    elif hypothesis_class == 'VoxelBasedShapeMaxD':
        import voxel_based_shape as vox
        h = vox.VoxelBasedShapeMaxD(forward_model=fwm, viewpoint=viewpoint, params=shape_params, max_depth=max_depth)
        moves['voxel_flip_full_vs_empty'] = vox.voxel_based_shape_flip_full_vs_empty
        moves['voxel_flip_partial_vs_full'] = vox.voxel_based_shape_flip_full_vs_partial
        moves['voxel_flip_partial_vs_empty'] = vox.voxel_based_shape_flip_empty_vs_partial
        moves['voxel_scale_space'] = vox.voxel_scale_space
        kernel_params['MAX_DEPTH'] = max_depth
        kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)
    else:
        raise ValueError("Unknown hypothesis class {0:s}.".format(hypothesis_class))

    # read data (i.e., observed image) from disk
    s = ""
    if single_view:
        s = "_single_view"
    data = np.load("{0:s}/{1:s}{2:s}.npy".format(data_folder, input_file, s))

    if sampler == 'mh':
        from mcmclib.mh_sampler import MHSampler
        sampler = MHSampler(initial_h=h, data=data, proposal=kernel, burn_in=burn_in, sample_count=sample_count,
                            best_sample_count=best_sample_count, thinning_period=thinning_period,
                            report_period=report_period)
    elif sampler == 'pt':
        from mcmclib.parallel_tempering_sampler import ParallelTemperingSampler
        sampler = ParallelTemperingSampler(initial_hs=[h]*chain_count, data=data, proposals=[kernel]*chain_count,
                                           temperatures=temperatures, burn_in=burn_in, sample_count=sample_count,
                                           best_sample_count=best_sample_count,
                                           thinning_period=int(thinning_period / chain_count),
                                           report_period=int(report_period / chain_count))
    else:
        raise ValueError('Unknown sampler. Possible choices are mh and pt.')

    start = time.time()
    run = sampler.sample()
    end = time.time()

    try:
        os.mkdir("{0:s}/{1:s}".format(results_folder, hypothesis_class))
    except OSError as e:
        warnings.warn(e.message)

    # generate a random run id
    run_id = np.random.randint(1000000)
    run_file = "{0:s}/{1:s}/{2:s}_{3:s}_{4:06d}.pkl".format(results_folder, hypothesis_class, input_file,
                                                            time.strftime("%Y%m%d_%H%M%S", time.localtime(start)),
                                                            run_id)
    run.save(run_file)

    fwm2 = vfm.VisionForwardModel(render_size=(300, 300))

    try:
        os.mkdir("{0:s}/{1:s}/{2:s}".format(results_folder, hypothesis_class, input_file))
    except OSError as e:
        warnings.warn(e.message)
    for i, sample in enumerate(run.samples.samples):
        fwm2.save_render("{0:s}/{1:s}/{2:s}/s{3:d}.png".format(results_folder, hypothesis_class, input_file, i), sample)
    for i, sample in enumerate(run.best_samples.samples):
        fwm2.save_render("{0:s}/{1:s}/{2:s}/b{3:d}.png".format(results_folder, hypothesis_class, input_file, i), sample)

    sample_lls = [sample.log_likelihood(data) for sample in run.samples.samples]
    best_lls = [sample.log_likelihood(data) for sample in run.best_samples.samples]
    mse_best = -2 * shape_params['LL_VARIANCE'] * np.max(best_lls)
    mse_mean = -2 * shape_params['LL_VARIANCE'] * np.mean(best_lls)
    mse_sample = -2 * shape_params['LL_VARIANCE'] * np.mean(sample_lls)
    # form the results dictionary
    results = {'run_id': run_id, 'run_file': run_file, 'mean_acceptance_rate': run.run_log.IsAccepted.mean(),
               'start_time': start, 'end_time': end, 'duration': (end - start) / 60.0,
               'best_posterior': np.max(run.best_samples.log_probs), 'best_ll': np.max(best_lls), 'mse': mse_best,
               'mean_best_posterior': np.mean(run.best_samples.log_probs),
               'mean_best_ll': np.mean(best_lls), 'mse_mean': mse_mean,
               'mean_sample_posterior': np.mean(run.samples.log_probs),
               'mean_sample_ll': np.mean(sample_lls), 'mse_sample': mse_sample}

    # add acceptance rate by move to results
    acc_rate_by_move = run.acceptance_rate_by_move()
    acc_rates = dict(zip(acc_rate_by_move.MoveType, acc_rate_by_move.AcceptanceRate))
    results.update(acc_rates)

    return results

if __name__ == "__main__":
    ADD_PART_PROB = 0.6
    LL_VARIANCE = 0.0001  # in squared pixel distance
    MAX_PIXEL_VALUE = 175.0  # this is usually 256.0 but in our case because of the lighting in our renders, it is lower
    LL_FILTER_SIGMA = 2.0
    MOVE_PART_VARIANCE = 0.00005
    MOVE_OBJECT_VARIANCE = 0.00005
    CHANGE_SIZE_VARIANCE = 0.00005
    SCALE_SPACE_VARIANCE = 0.0025
    CHANGE_VIEWPOINT_VARIANCE = 30.0

    experiment = exp.Experiment(name="Stimuli20150624_144833", experiment_method=run_chain,
                                grouped_params=['ll_variance', 'change_size_variance', 'move_object_variance',
                                                'move_part_variance', 'change_viewpoint_variance',
                                                'scale_space_variance'],
                                single_view=True,
                                hypothesis_class=['BDAoOSSShapeMaxD'],
                                sampler='mh', input_file=['o1_t2_ap_d1', 'o9_t2_rp_d2', 'o6_t2_ap_d2', 'o1_t1_cs_d2',
                                                          'o5_t2_mf_d1','o7_t2_ap_d2','o4_t1_cs_d1','o5_t2_mf_d2',
                                                          'o8_t2_mf_d1','o7','o1_t2_rp_d2','o4_t2_mf_d2','o7_t2_ap_d1',
                                                          'o5_t2_rp_d2','o1','o2_t1_cs_d2','o5','o5_t2_ap_d2',
                                                          'o5_t1_cs_d2','o2_t1_cs_d1','o2','o2_t2_mf_d2','o6_t2_ap_d1',
                                                          'o5_t1_cs_d1','o9_t2_ap_d1','o6_t2_mf_d1','o2_t2_ap_d1',
                                                          'o1_t2_mf_d1','o1_t2_ap_d2','o10_t1_cs_d2','o5_t2_ap_d1',
                                                          'o4_t2_mf_d1','o6_t1_cs_d2','o10_t1_cs_d1','o6_t2_mf_d2',
                                                          'o10_t2_ap_d2','o2_t2_ap_d2','o10_t2_mf_d1','o10',
                                                          'o10_t2_ap_d1'],
                                results_folder='./results',
                                data_folder='./data/stimuli20150624_144833/',
                                render_size=(200, 200),
                                max_part_count=10, max_depth=5,
                                add_part_prob=ADD_PART_PROB,
                                max_pixel_value=MAX_PIXEL_VALUE,
                                ll_variance=[0.0001],
                                change_size_variance=[0.00005],
                                move_object_variance=[0.00005],
                                move_part_variance=[0.00005],
                                change_viewpoint_variance=[60.0],
                                scale_space_variance=[0.0025],
                                burn_in=0, sample_count=20, best_sample_count=20, thinning_period=10000,
                                report_period=10000, temperatures=[[3.0, 1.5, 1.0]])

    experiment.run(parallel=True, num_processes=9)

    print(experiment.results)
    experiment.save('./results')
    experiment.append_csv('./results/Run.csv')
    
    """
    input_file=['o1', 'o1_t1_cs_d1', 'o1_t1_cs_d2',
                                            'o1_t2_ap_d1', 'o1_t2_ap_d2',
                                            'o1_t2_rp_d1', 'o1_t2_rp_d2',
                                            'o1_t2_mf_d1', 'o1_t2_mf_d2'],

    'o2_t2_mf_d2', 'o5_t2_ap_d1', 'o2', 'o4', 'o1_t2_rp_d2', 'o4_t2_ap_d1', 'o1_t2_ap_d2', 'o2_t2_ap_d2',
    'o5_t2_ap_d2', 'o2_t2_ap_d1', o1_t2_mf_d1', 'o4_t2_mf_d1', 'o3_t2_rp_d2', 'o10_t2_ap_d2', 'o10_t2_ap_d1',
    'o10_t2_mf_d1'

    'o7', 'o7_t1_cs_d1', 'o7_t1_cs_d2',
                                                          'o7_t2_ap_d1', 'o7_t2_ap_d2',
                                                          'o7_t2_rp_d1', 'o7_t2_rp_d2',
                                                          'o7_t2_mf_d1', 'o7_t2_mf_d2',
                                                          'o8', 'o8_t1_cs_d1', 'o8_t1_cs_d2',
                                                          'o8_t2_ap_d1', 'o8_t2_ap_d2',
                                                          'o8_t2_rp_d1', 'o8_t2_rp_d2',
                                                          'o8_t2_mf_d1', 'o8_t2_mf_d2',

    Re-run with Shape
19   o10_t2_mf_d1    9.022714
11   o10_t1_cs_d2    9.352118
48    o1_t2_mf_d1   12.227496
8    o10_t1_cs_d1   12.447928
43    o1_t2_ap_d2   12.789126
16   o10_t2_ap_d2   15.901736
14   o10_t2_ap_d1   23.057532
78    o2_t2_ap_d1  127.857558

    Re-run with BDAoOSSShapeMaxD
'o1_t2_ap_d1',
'o9_t2_rp_d2',
'o6_t2_ap_d2',
'o1_t1_cs_d2',
'o5_t2_mf_d1',
'o7_t2_ap_d2',
'o4_t1_cs_d1',
'o5_t2_mf_d2',
'o8_t2_mf_d1',
'o7',
'o1_t2_rp_d2',
'o4_t2_mf_d2',
'o7_t2_ap_d1',
'o5_t2_rp_d2',
'o1',
'o2_t1_cs_d2',
'o5         ',
'o5_t2_ap_d2',
'o5_t1_cs_d2',
'o2_t1_cs_d1',
'o2',
'o2_t2_mf_d2',
'o6_t2_ap_d1',
'o5_t1_cs_d1',
'o9_t2_ap_d1',
'o6_t2_mf_d1',
'o2_t2_ap_d1',
'o1_t2_mf_d1',
'o1_t2_ap_d2',
'o10_t1_cs_d2',
'o5_t2_ap_d1',
'o4_t2_mf_d1',
'o6_t1_cs_d2',
'o10_t1_cs_d1',
'o6_t2_mf_d2',
'o10_t2_ap_d2',
'o2_t2_ap_d2',
'o10_t2_mf_d1',
'o10',
'o10_t2_ap_d1'
"""
