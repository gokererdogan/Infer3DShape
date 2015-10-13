"""
Inferring 3D Shape from 2D Images

This file contains the experiment script for running the
chains on different inputs and with different parameters.

Created on Oct 12, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import gmllib.experiment as exp

def run_chain(input_file, results_folder, max_depth, ll_variance, max_pixel_value, change_size_variance,
              burn_in, sample_count, best_sample_count, thinning_period, report_period):
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
    import bdaooss_shape as bdaooss
    import numpy as np
    fwm = vfm.VisionForwardModel()

    kernel = bdaooss.BDAoOSSShapeMaxDProposal(params={'CHANGE_SIZE_VARIANCE': change_size_variance})

    # generate initial hypothesis shape randomly
    params = {'LL_VARIANCE': ll_variance, 'MAX_PIXEL_VALUE': max_pixel_value}
    h = bdaooss.BDAoOSSShapeMaxD(forward_model=fwm, max_depth=max_depth, params=params)

    # read data (i.e., observed image) from disk
    data = np.load("./data/{0:s}.npy".format(input_file))

    sampler = mcmc.MHSampler(h, data, kernel, burn_in=burn_in, sample_count=sample_count,
                             best_sample_count=best_sample_count, thinning_period=thinning_period,
                             report_period=report_period)
    start = time.time()
    run = sampler.sample()
    end = time.time()

    run.save("{0:s}/{1:s}.pkl".format(results_folder, input_file))

    fwm2 = vfm.VisionForwardModel(render_size=(300, 300))

    for i, sample in enumerate(run.samples.samples):
        fwm2.save_render("{0:s}/{1:s}/s{2:d}.png".format(results_folder, input_file, i), sample)
    for i, sample in enumerate(run.best_samples.samples):
        fwm2.save_render("{0:s}/{1:s}/b{2:d}.png".format(results_folder, input_file, i), sample)

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
    import hypothesis as hyp
    experiment = exp.Experiment(name="TestObjectsBDAoOSSShapeMaxD", experiment_method=run_chain,
                                input_file=['test1', 'test2', 'test3'], results_folder='./results/bdaoossShapeMaxD',
                                max_depth=3, ll_variance=hyp.LL_VARIANCE, max_pixel_value=hyp.MAX_PIXEL_VALUE,
                                change_size_variance=hyp.CHANGE_SIZE_VARIANCE,
                                burn_in=0, sample_count=10, best_sample_count=20, thinning_period=20000,
                                report_period=10000)

    experiment.run(parallel=True, num_processes=3)

    print(experiment.results)
    experiment.save()
    experiment.save_csv()

