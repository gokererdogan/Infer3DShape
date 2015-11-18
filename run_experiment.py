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
    """
    This method runs the chain with the given parameters, saves the results and returns a summary of the results.
    This method is intended to be used in an Experiment instance.
    :param input_file:
    :param results_folder:
    :param max_depth:
    :param ll_variance:
    :param max_pixel_value:
    :param change_size_variance:
    :param burn_in:
    :param sample_count:
    :param best_sample_count:
    :param thinning_period:
    :param report_period:
    :return: dictionary of run results
    """
    import time
    import mcmc_sampler as mcmc
    import vision_forward_model as vfm
    import numpy as np
    fwm = vfm.VisionForwardModel(render_size=render_size)

    shape_params = {'ADD_PART_PROB': add_part_prob, 'LL_VARIANCE': ll_variance, 'MAX_PIXEL_VALUE': max_pixel_value}
    kernel_params = {'CHANGE_SIZE_VARIANCE': change_size_variance, 'MOVE_PART_VARIANCE': move_part_variance,
                     'MOVE_OBJECT_VARIANCE': move_object_variance, 'CHANGE_VIEWPOINT_VARIANCE': change_viewpoint_variance}

    if single_view:
        viewpoint = [(3.0, -3.0, 3.0)]
        allow_viewpoint_update = True
    else:
        viewpoint = None
        allow_viewpoint_update = False

    if hypothesis_class == 'shape':
        import hypothesis as hyp
        h = hyp.Shape(forward_model=fwm, viewpoint=viewpoint, params=shape_params)
        kernel = hyp.ShapeProposal(allow_viewpoint_update=allow_viewpoint_update, params=kernel_params)
    elif hypothesis_class == 'shapeMaxN':
        import shape_maxn
        h = shape_maxn.ShapeMaxN(forward_model=fwm, maxn=max_part_count, viewpoint=viewpoint, params=shape_params)
        kernel = shape_maxn.ShapeMaxNProposal(allow_viewpoint_update=allow_viewpoint_update, params=kernel_params)
    elif hypothesis_class == 'bdaoossShape':
        import bdaooss_shape as bdaooss
        h = bdaooss.BDAoOSSShape(forward_model=fwm, viewpoint=viewpoint, params=shape_params)
        kernel = bdaooss.BDAoOSSShapeProposal(allow_viewpoint_update=allow_viewpoint_update, params=kernel_params)
    elif hypothesis_class == 'bdaoossShapeMaxD':
        import bdaooss_shape as bdaooss
        h = bdaooss.BDAoOSSShapeMaxD(forward_model=fwm, max_depth=max_depth, viewpoint=viewpoint, params=shape_params)
        kernel = bdaooss.BDAoOSSShapeMaxDProposal(allow_viewpoint_update=allow_viewpoint_update, params=kernel_params)
    else:
        raise Exception("Unknown hypothesis class.")

    # read data (i.e., observed image) from disk
    s = ""
    if single_view:
        s = "_single_view"
    data = np.load("{0:s}/{1:s}{2:s}.npy".format(data_folder, input_file, s))

    sampler = mcmc.MHSampler(h, data, kernel, burn_in=burn_in, sample_count=sample_count,
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

    sample_lls = [sample.likelihood(data) for sample in run.samples.samples]
    best_lls = [sample.likelihood(data) for sample in run.best_samples.samples]
    # form the results dictionary
    results = {'mean_acceptance_rate': run.iter_df.IsAccepted.mean(), 'start_time': start, 'end_time': end,
               'best_posterior': np.max(run.best_samples.probs), 'best_ll': np.max(best_lls),
               'mean_best_posterior': np.mean(run.best_samples.probs), 'mean_best_ll': np.mean(best_lls),
               'mean_sample_posterior': np.mean(run.samples.probs), 'mean_sample_ll': np.mean(sample_lls)}

    # add acceptance rate by move to results
    acc_rate_by_move = run.acceptance_rate_by_move()
    acc_rates = dict(zip(acc_rate_by_move.MoveType, acc_rate_by_move.AcceptanceRate))
    results.update(acc_rates)

    return results

if __name__ == "__main__":
    ADD_PART_PROB = 0.6
    LL_VARIANCE = 0.001 # in squared pixel distance
    MAX_PIXEL_VALUE = 175.0 # this is usually 256.0 but in our case because of the lighting in our renders, it is lower
    LL_FILTER_SIGMA = 2.0
    MOVE_PART_VARIANCE = .005
    MOVE_OBJECT_VARIANCE = 0.05
    CHANGE_SIZE_VARIANCE = .040
    CHANGE_VIEWPOINT_VARIANCE = 60.0

    experiment = exp.Experiment(name="o2345_single_view", experiment_method=run_chain, single_view=True,
                                hypothesis_class=['bdaoossShapeMaxD'],
                                input_file=['o2', 'o2_t1_cs_d1', 'o2_t1_cs_d2',
                                            'o2_t2_ap_d1', 'o2_t2_ap_d2',
                                            'o2_t2_rp_d1', 'o2_t2_rp_d2',
                                            'o2_t2_mf_d1', 'o2_t2_mf_d2',
                                            'o3', 'o3_t1_cs_d1', 'o3_t1_cs_d2',
                                            'o3_t2_ap_d1', 'o3_t2_ap_d2',
                                            'o3_t2_rp_d1', 'o3_t2_rp_d2',
                                            'o3_t2_mf_d1', 'o3_t2_mf_d2',
                                            'o4', 'o4_t1_cs_d1', 'o4_t1_cs_d2',
                                            'o4_t2_ap_d1', 'o4_t2_ap_d2',
                                            'o4_t2_rp_d1', 'o4_t2_rp_d2',
                                            'o4_t2_mf_d1', 'o4_t2_mf_d2',
                                            'o5', 'o5_t1_cs_d1', 'o5_t1_cs_d2',
                                            'o5_t2_ap_d1', 'o5_t2_ap_d2',
                                            'o5_t2_rp_d1', 'o5_t2_rp_d2',
                                            'o5_t2_mf_d1', 'o5_t2_mf_d2'],
                                results_folder='./results',
                                data_folder='./data/stimuli20150624_144833', render_size=(200, 200),
                                max_part_count=8, max_depth=3,
                                add_part_prob=ADD_PART_PROB, ll_variance=LL_VARIANCE,
                                max_pixel_value=MAX_PIXEL_VALUE,
                                change_size_variance=CHANGE_SIZE_VARIANCE,
                                change_viewpoint_variance=CHANGE_VIEWPOINT_VARIANCE,
                                move_part_variance=MOVE_PART_VARIANCE,
                                move_object_variance=MOVE_OBJECT_VARIANCE,
                                burn_in=0, sample_count=10, best_sample_count=20, thinning_period=20000,
                                report_period=10000)

    experiment.run(parallel=True, num_processes=16)

    print(experiment.results)
    experiment.save('./results')
    experiment.append_csv('./results/Stimuli20150624.csv')

