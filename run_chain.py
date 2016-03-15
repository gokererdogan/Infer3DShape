"""
Inferring 3D Shape from 2D Images

This file contains the run_chain function used by run_experiment scripts.

Created on Oct 12, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import os
import warnings
import time

import numpy as np

import vision_forward_model as vfm


def run_chain(name, sampler, initial_h, data, kernel, burn_in, thinning_period, sample_count, best_sample_count,
              report_period, results_folder, temperatures=None):
    """Run an MCMC chain and save results.

    This function is used by run_experiment scripts to run chains and save results.

    Parameters:
        name (str): name of the chain. Used as the folder name to save sample images
        sampler (str): Sampler to use. 'mh' for Metropolis-Hastings, 'pt' for Parallel Tempering
        initial_h (I3DHypothesis): Initial hypothesis
        data (numpy.ndarray): Observed data
        kernel (mcmclib.Proposal): Transition kernel of the chain
        burn_in (int): Number of burn in iterations
        thinning_period (int): Keep every ith sample
        sample_count (int): Number of samples to take
        best_sample_count (int): Size of the best samples list
        report_period (int): Report the status of the chain every report_period iterations
        results_folder (str): Folder to save the results
        temperatures (list): Temperatures of each chain for Parallel Tempering sampler

    Returns:
        dict: results
    """
    if sampler == 'mh':
        from mcmclib.mh_sampler import MHSampler
        sampler = MHSampler(initial_h=initial_h, data=data, proposal=kernel, burn_in=burn_in, sample_count=sample_count,
                            best_sample_count=best_sample_count, thinning_period=thinning_period,
                            report_period=report_period)
    elif sampler == 'pt':
        if temperatures is None:
            raise ValueError('ParallelTempering sampler requires temperatures parameter.')

        chain_count = len(temperatures)
        from mcmclib.parallel_tempering_sampler import ParallelTemperingSampler
        sampler = ParallelTemperingSampler(initial_hs=[initial_h]*chain_count, data=data, proposals=[kernel]*chain_count,
                                           temperatures=temperatures, burn_in=burn_in, sample_count=sample_count,
                                           best_sample_count=best_sample_count,
                                           thinning_period=int(thinning_period / chain_count),
                                           report_period=int(report_period / chain_count))
    else:
        raise ValueError('Unknown sampler. Possible choices are mh and pt.')

    start = time.time()
    run = sampler.sample()
    end = time.time()

    # generate a random run id
    run_id = np.random.randint(1000000)
    run_file = "{0:s}/{1:s}_{2:s}_{3:06d}.pkl".format(results_folder, name,
                                                      time.strftime("%Y%m%d_%H%M%S", time.localtime(start)),
                                                      run_id)
    run.save(run_file)

    # save images of samples to disk
    fwm2 = vfm.VisionForwardModel(render_size=(300, 300))

    try:
        os.mkdir("{0:s}/{1:s}".format(results_folder, name))
    except OSError as e:
        warnings.warn(e.message)

    for i, sample in enumerate(run.samples.samples):
        fwm2.save_render("{0:s}/{1:s}/s{2:d}.png".format(results_folder, name, i), sample)
    for i, sample in enumerate(run.best_samples.samples):
        fwm2.save_render("{0:s}/{1:s}/b{2:d}.png".format(results_folder, name, i), sample)

    sample_lls = [sample.log_likelihood(data) for sample in run.samples.samples]
    best_lls = [sample.log_likelihood(data) for sample in run.best_samples.samples]
    mse_best = -2 * initial_h.params['LL_VARIANCE'] * np.max(best_lls)
    mse_mean = -2 * initial_h.params['LL_VARIANCE'] * np.mean(best_lls)
    mse_sample = -2 * initial_h.params['LL_VARIANCE'] * np.mean(sample_lls)

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
