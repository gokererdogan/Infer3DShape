"""
Inferring 3D Shape from 2D Images

This file contains the script for calculating predictions of our model.
WARNING: For pickle to run properly, i.e., import the necessary modules), run this script from the root Infer3DShape
folder.

Created on Feb 1, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""

import numpy as np
import scipy.misc as spmisc
import pandas as pd
import pandasql as psql

import Infer3DShape.vision_forward_model as vfm
from mcmclib.mcmc_run import MCMCRun


def calculate_probability_image_given_hypothesis(img, h, viewpoint_samples=180):
    """
    Calculates the log probability of image given hypothesis, p(I|H) = \int p(I,theta|H) dtheta, marginalizing out
    viewpoint theta. We assume p(theta) is uniform.

    Parameters:
        img (numpy.array): image
        h (I3DHypothesis): shape hypothesis
        viewpoint_samples (int): number of viewpoints to use to approximate the integral

    Returns:
        float: log probability of image given hypothesis, averaged over all views
        float: log probability of image given hypothesis for the best view
    """
    ll = np.zeros(viewpoint_samples)
    # rotate the viewpoint around z axis
    for i, theta in enumerate(np.arange(0, 360, 360.0 / viewpoint_samples)):
        # update all viewpoints
        for v in range(len(h.viewpoint)):
            r, _, phi = h.viewpoint[v]
            h.viewpoint[v] = (r, theta, phi)

        h._log_ll = None
        log_ll = h.log_likelihood(img)
        ll[i] = log_ll
    return spmisc.logsumexp(ll) - np.log(viewpoint_samples), np.max(ll)


def calculate_similarity_image_given_image(data1, samples, log_probs):
    """
    Calculate similarity between images data1 and data2 given samples from p(H, theta|data2).
    Similarity between data1 and data2 is defined to be p(data1|data2) calculated from
    p(data1|data2) = \iint p(data1|H, theta) p(H|data2) p(theta) dH dtheta.
    samples are a list of samples from p(H, theta|data2). We assume p(theta) is uniform.

    Parameters:
        data1 (np.array):
        samples (list of I3DHypothesis):
        log_probs: log posterior probabilities of samples in ``samples``

    Returns:
        float: log p(data1|data2) calculated based on samples
        float: log p(data1|data2) calculated by samples weighted by their posterior probabilities.
            Note this is not the correct approximation for p(data1|data2), but it sometimes gives
            good results.
        float: log p(data1|data2) calculated based on samples with p(data|H) calculated from only the best view.
        float: log p(data1|data2) calculated by samples weighted by their posterior probabilities and with p(data|H)
            calculated from only the best view.
    """
    # calculate p(H|data2) for each sample in samples
    sample_count = len(samples)
    logp_data1 = np.zeros(sample_count)  # log ll averaged over all views
    best_logp_data1 = np.zeros(sample_count)  # log ll from the best view
    for i, sample in enumerate(samples):
        print('.'),
        logp_data1[i], best_logp_data1[i] = calculate_probability_image_given_hypothesis(data1, sample)
    print

    p_avg = spmisc.logsumexp(logp_data1) - np.log(sample_count)
    p_wavg = spmisc.logsumexp(logp_data1 + log_probs - spmisc.logsumexp(log_probs))
    p_best = spmisc.logsumexp(best_logp_data1) - np.log(sample_count)
    p_wbest = spmisc.logsumexp(best_logp_data1 + log_probs - spmisc.logsumexp(log_probs))

    return p_avg, p_wavg, p_best, p_wbest


def read_samples(run_file, forward_model):
    """
    Load the samples from run file and restore their forward models.

    Parameters:
        run_file (string): filename of the pickled MCMCRun object
        forward_model (VisionForwardModel): forward model used for rendering

    Returns:
        (list of I3DHypothesis): Samples from the chain
        (list of float): log posterior probabilities for each sample
        (list of I3DHypothesis): Samples with the highest posterior from the chain
        (list of float): log posterior probabilities for high posterior samples
    """
    run = MCMCRun.load(run_file)

    samples = run.samples.samples[5:]
    log_probs = run.samples.log_probs[5:]
    for sample in samples:
        sample.forward_model = forward_model

    best_samples = run.best_samples.samples
    best_log_probs = run.best_samples.log_probs
    for sample in best_samples:
        sample.forward_model = forward_model

    return samples, log_probs, best_samples, best_log_probs

if __name__ == "__main__":
    fwm = vfm.VisionForwardModel()

    hypothesis = 'BDAoOSSShapeMaxD'
    run_date = '20160128'

    data_folder = "./data/stimuli20150624_144833"
    samples_folder = "./results/{0:s}/{1:s}".format(hypothesis, run_date)

    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    transformations = ['t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_mf_d1', 't2_mf_d2', 't2_rp_d1', 't2_rp_d2']
    variations = [o + '_' + t for t in transformations for o in objects]
    comparisons = {o: variations for o in objects}
    # comparisons = {o: [o + '_' + t for t in transformations] for o in objects}

    columns = ['Target', 'Comparison', 'p_comp_target', 'p_target_comp', 'p_avg', 'p_comp_target_w', 'p_target_comp_w',
               'p_avg_w', 'p_comp_target_MAP', 'p_target_comp_MAP', 'p_avg_MAP', 'p_comp_target_MAP_w',
               'p_target_comp_MAP_w', 'p_avg_MAP_w']
    df = pd.DataFrame(index=np.arange(0, len(objects) * len(comparisons['o1'])), columns=columns)

    i = 0
    for obj in objects:
        print(obj)
        obj_data = np.load('{0:s}/{1:s}_single_view.npy'.format(data_folder, obj))

        run_file = "{0:s}/{1:s}.pkl".format(samples_folder, obj)
        obj_samples, obj_log_probs, obj_best_samples, obj_best_log_probs = read_samples(run_file, fwm)

        for comparison in comparisons[obj]:
            print("\t{0:s}".format(comparison))
            comp_data = np.load('{0:s}/{1:s}_single_view.npy'.format(data_folder, comparison))

            run_file = "{0:s}/{1:s}.pkl".format(samples_folder, comparison)
            comp_samples, comp_log_probs, comp_best_samples, comp_best_log_probs = read_samples(run_file, fwm)

            # calculate similarities
            p_comp_target, p_comp_target_w = calculate_similarity_image_given_image(comp_data, obj_samples,
                                                                                    obj_log_probs)
            p_target_comp, p_target_comp_w = calculate_similarity_image_given_image(obj_data, comp_samples,
                                                                                    comp_log_probs)
            p_comp_target_MAP, p_comp_target_MAP_w = calculate_similarity_image_given_image(comp_data, obj_best_samples,
                                                                                            obj_best_log_probs)
            p_target_comp_MAP, p_target_comp_MAP_w = calculate_similarity_image_given_image(obj_data, comp_best_samples,
                                                                                            comp_best_log_probs)
            df.loc[i] = [obj, comparison,
                         p_comp_target, p_target_comp, (p_comp_target + p_target_comp) / 2.0,
                         p_comp_target_w, p_target_comp_w, (p_comp_target_w + p_target_comp_w) / 2.0,
                         p_comp_target_MAP, p_target_comp_MAP, (p_comp_target_MAP + p_target_comp_MAP) / 2.0,
                         p_comp_target_MAP_w, p_target_comp_MAP_w, (p_comp_target_MAP_w + p_target_comp_MAP_w) / 2.0]
            i += 1

    predictions = psql.sqldf("select d1.Target as Target, d1.Comparison as Comparison1, d2.Comparison as Comparison2, "
                             "d1.p_comp_target > d2.p_comp_target as I3D_pcomp_Prediction, "
                             "d1.p_target_comp > d2.p_target_comp as I3D_ptarget_Prediction, "
                             "d1.p_avg > d2.p_avg as I3D_pavg_Prediction, "
                             "d1.p_comp_target_MAP > d2.p_comp_target_MAP as I3D_pcomp_MAP_Prediction, "
                             "d1.p_target_comp_MAP > d2.p_target_comp_MAP as I3D_ptarget_MAP_Prediction, "
                             "d1.p_avg_MAP > d2.p_avg_MAP as I3D_pavg_MAP_Prediction, "
                             "d1.p_comp_target_w > d2.p_comp_target_w as I3D_pcomp_w_Prediction, "
                             "d1.p_target_comp_w > d2.p_target_comp_w as I3D_ptarget_w_Prediction, "
                             "d1.p_avg_w > d2.p_avg_w as I3D_pavg_w_Prediction, "
                             "d1.p_comp_target_MAP_w > d2.p_comp_target_MAP_w as I3D_pcomp_MAP_w_Prediction, "
                             "d1.p_target_comp_MAP_w > d2.p_target_comp_MAP_w as I3D_ptarget_MAP_w_Prediction, "
                             "d1.p_avg_MAP_w > d2.p_avg_MAP_w as I3D_pavg_MAP_w_Prediction "
                             "from df as d1, df as d2 where d1.Target = d2.Target and d1.Comparison < d2.Comparison",
                             env=locals())
    # write to disk
    open('I3D_AllComparisons_{0:s}_{1:s}_ModelPredictions.txt'.format(hypothesis, run_date), 'w').write(predictions.to_string())
    open('I3D_AllComparisons_{0:s}_{1:s}_ModelDistances.txt'.format(hypothesis, run_date), 'w').write(df.to_string())
