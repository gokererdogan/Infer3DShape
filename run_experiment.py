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

def run_chain(input_file, results_folder, data_folder, hypothesis_class, single_view, render_size, max_part_count,
              max_depth, add_part_prob, ll_variance, max_pixel_value, change_size_variance, change_viewpoint_variance,
              move_part_variance, move_object_variance, burn_in, sample_count, best_sample_count, thinning_period,
              report_period):
    """This method runs the chain with the given parameters, saves the results and returns a summary of the results.

    This method is intended to be used in an Experiment instance.

    Args:
        input_file:
        results_folder:
        max_depth:
        ll_variance:
        max_pixel_value:
        change_size_variance:
        burn_in:
        sample_count:
        best_sample_count:
        thinning_period:
        report_period:

    Returns:
        dictionary of run results
    """
    import time
    import mcmclib.mh_sampler as mcmc
    import mcmclib.proposal as proposal
    import vision_forward_model as vfm
    import numpy as np
    fwm = vfm.VisionForwardModel(render_size=render_size)

    shape_params = {'ADD_PART_PROB': add_part_prob, 'LL_VARIANCE': ll_variance, 'MAX_PIXEL_VALUE': max_pixel_value}
    kernel_params = {'CHANGE_SIZE_VARIANCE': change_size_variance, 'MOVE_PART_VARIANCE': move_part_variance,
                     'MOVE_OBJECT_VARIANCE': move_object_variance, 'CHANGE_VIEWPOINT_VARIANCE': change_viewpoint_variance}

    moves = {}
    if single_view:
        import i3d_proposal
        viewpoint = [(1.5, -1.5, 1.5)]
        moves['change_viewpoint'] = i3d_proposal.change_viewpoint
    else:
        viewpoint = None

    if hypothesis_class == 'shape':
        import shape
        h = shape.Shape(forward_model=fwm, viewpoint=viewpoint, params=shape_params)
        moves['shape_add_remove_part'] = shape.shape_add_remove_part
        moves['shape_move_part'] = shape.shape_move_part
        moves['shape_move_part_local'] = shape.shape_move_part_local
        moves['shape_change_part_size'] = shape.shape_change_part_size
        moves['shape_change_part_size_local'] = shape.shape_change_part_size_local
        moves['shape_move_object'] = shape.shape_move_object
        kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)
    elif hypothesis_class == 'shapeMaxN':
        import shape
        import shape_maxn
        h = shape_maxn.ShapeMaxN(forward_model=fwm, maxn=max_part_count, viewpoint=viewpoint, params=shape_params)
        moves['shape_add_remove_part'] = shape.shape_add_remove_part
        moves['shape_move_part'] = shape.shape_move_part
        moves['shape_move_part_local'] = shape.shape_move_part_local
        moves['shape_change_part_size'] = shape.shape_change_part_size
        moves['shape_change_part_size_local'] = shape.shape_change_part_size_local
        moves['shape_move_object'] = shape.shape_move_object,
        kernel_params['MAX_PART_COUNT'] = max_part_count
        kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)
    elif hypothesis_class == 'bdaoossShape':
        import bdaooss_shape as bdaooss
        h = bdaooss.BDAoOSSShape(forward_model=fwm, viewpoint=viewpoint, params=shape_params)
        moves['bdaooss_add_remove_part'] = bdaooss.bdaooss_add_remove_part
        moves['bdaooss_change_part_size'] = bdaooss.bdaooss_change_part_size
        moves['bdaooss_change_part_size_local'] = bdaooss.bdaooss_change_part_size_local
        moves['bdaooss_change_part_dock_face'] = bdaooss.bdaooss_change_part_dock_face
        moves['bdaooss_move_object'] = bdaooss.bdaooss_move_object
        kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)
    elif hypothesis_class == 'bdaoossShapeMaxD':
        import bdaooss_shape as bdaooss
        import bdaooss_shape_maxd as bdaooss_maxd
        h = bdaooss_maxd.BDAoOSSShapeMaxD(forward_model=fwm, max_depth=max_depth, viewpoint=viewpoint,
                                          params=shape_params)
        moves['bdaooss_add_remove_part'] = bdaooss.bdaooss_add_remove_part
        moves['bdaooss_change_part_size'] = bdaooss.bdaooss_change_part_size
        moves['bdaooss_change_part_size_local'] = bdaooss.bdaooss_change_part_size_local
        moves['bdaooss_change_part_dock_face'] = bdaooss.bdaooss_change_part_dock_face
        moves['bdaooss_move_object'] = bdaooss.bdaooss_move_object
        kernel_params['MAX_DEPTH'] = max_depth
        kernel = proposal.RandomMixtureProposal(moves=moves, params=kernel_params)
    else:
        raise Exception("Unknown hypothesis class.")

    # read data (i.e., observed image) from disk
    s = ""
    if single_view:
        s = "_single_view"
    data = np.load("{0:s}/{1:s}{2:s}.npy".format(data_folder, input_file, s))

    sampler = mcmc.MHSampler(initial_h=h, data=data, proposal=kernel, burn_in=burn_in, sample_count=sample_count,
                             best_sample_count=best_sample_count, thinning_period=thinning_period,
                             report_period=report_period)
    start = time.time()
    run = sampler.sample()
    end = time.time()

    run.save("{0:s}/{1:s}/{2:s}_{3:s}.pkl".format(results_folder, hypothesis_class, input_file,
                                                  time.strftime("%Y%m%d_%H%M%S", time.localtime(start))))

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
    mse_best = -2 * kernel_params['LL_VARIANCE'] * np.max(best_lls)
    mse_mean = -2 * kernel_params['LL_VARIANCE'] * np.mean(best_lls)
    mse_sample = -2 * kernel_params['LL_VARIANCE'] * np.mean(sample_lls)
    # form the results dictionary
    results = {'mean_acceptance_rate': run.iter_df.IsAccepted.mean(),
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
    LL_VARIANCE = 0.0005 # in squared pixel distance
    MAX_PIXEL_VALUE = 175.0 # this is usually 256.0 but in our case because of the lighting in our renders, it is lower
    LL_FILTER_SIGMA = 2.0
    MOVE_PART_VARIANCE = .005
    MOVE_OBJECT_VARIANCE = 0.01
    CHANGE_SIZE_VARIANCE = .040
    CHANGE_VIEWPOINT_VARIANCE = 60.0

    experiment = exp.Experiment(name="o1_single_view_variance", experiment_method=run_chain, single_view=True,
                                hypothesis_class=['bdaoossShape'],
                                input_file=['o1'],
                                results_folder='./results',
                                data_folder='./data/stimuli20150624_144833', render_size=(200, 200),
                                max_part_count=8, max_depth=10,
                                add_part_prob=ADD_PART_PROB, ll_variance=[0.01, 0.001, 0.0001],
                                max_pixel_value=MAX_PIXEL_VALUE,
                                change_size_variance=CHANGE_SIZE_VARIANCE,
                                change_viewpoint_variance=CHANGE_VIEWPOINT_VARIANCE,
                                move_part_variance=MOVE_PART_VARIANCE,
                                move_object_variance=MOVE_OBJECT_VARIANCE,
                                burn_in=0, sample_count=10, best_sample_count=20, thinning_period=20000,
                                report_period=10000)

    experiment.run(parallel=True, num_processes=3)

    print(experiment.results)
    experiment.save('./results')
    experiment.append_csv('./results/Stimuli20150624.csv')
    
    """
    input_file=['o1', 'o1_t1_cs_d1', 'o1_t1_cs_d2',
                                            'o1_t2_ap_d1', 'o1_t2_ap_d2',
                                            'o1_t2_rp_d1', 'o1_t2_rp_d2',
                                            'o1_t2_mf_d1', 'o1_t2_mf_d2'],
    """                         
