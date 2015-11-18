"""
Inferring 3D Shape from 2D Images

This script is for calculating similarities based on our model (Infer3DShape)
between stimuli in BDAoOSS experiment.

Created on Oct 5, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

import pandas as pd
import pandasql as psql
import cPickle as pkl

from shape_maxn import *
import vision_forward_model as vfm

def calculate_prob(data, samples, probs):
    """
    Calculate probability of observing data from samples with given probs.
        This function calculates prob(obj2|obj1) where obj2 and
        obj1 are both images for our purposes. Here obj2
        corresponds to the data parameter. samples are from
        the posterior p(h|obj1). we can estimate p(obj2|obj1) as
            p(obj2|obj1) ~ sum_h p(obj2|h)p(h|obj1)
    :param data: Observed data.
    :param samples: Samples.
    :param probs: Probability of samples.
    :return: probability of observing data from samples
    """
    probs /= np.sum(probs)
    sim = 0.0
    for sample, prob in zip(samples, probs):
        sample.ll = None
        sim += (prob * sample.likelihood(data))
    return sim

if __name__ == "__main__":
    fwm = vfm.VisionForwardModel()

    data_folder = "./data/stimuli20150624_144833"
    samples_folder = "./results/bdaoossShapeMaxD"

    df = pd.DataFrame(index=np.arange(0, 8), columns=['Target', 'Comparison', 'p_comp_target', 'p_target_comp'])

    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    transformations = ['t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_mf_d1', 't2_mf_d2', 't2_rp_d1', 't2_rp_d2']

    i = 0
    for obj in objects:
        print(obj)
        # load the target object data and samples
        obj_run = pkl.load(open("{0:s}/{1:s}.pkl".format(samples_folder, obj)))
        obj_data = np.load('{0:s}/{1:s}.npy'.format(data_folder, obj))

        obj_best_samples = obj_run.best_samples.samples
        for sample in obj_best_samples:
            sample.forward_model = fwm

        obj_probs = obj_run.best_samples.probs

        for transformation in transformations:
            comparison = "{0:s}_{1:s}".format(obj, transformation)
            comp_run = pkl.load(open("{0:s}/{1:s}.pkl".format(samples_folder, comparison)))
            comp_data = np.load('{0:s}/{1:s}.npy'.format(data_folder, comparison))

            comp_best_samples = comp_run.best_samples.samples
            for sample in comp_best_samples:
                sample.forward_model = fwm

            comp_probs = comp_run.best_samples.probs

            p_comp_target = calculate_prob(comp_data, obj_best_samples, obj_probs)
            p_target_comp = calculate_prob(obj_data, comp_best_samples, comp_probs)
            df.loc[i] = [obj, comparison, p_comp_target, p_target_comp]
            i += 1

    # calculate model predictions
    predictions = psql.sqldf("select d1.Comparison as Comparison1, d2.Comparison as Comparison2, "
                             "(d1.p_comp_target / (d1.p_comp_target + d2.p_comp_target)) as I3D_pcomp_Prediction, "
                             "(d1.p_target_comp / (d1.p_target_comp + d2.p_target_comp)) as I3D_ptarget_Prediction, "
                             "((d1.p_comp_target + d1.p_target_comp) / "
                             "(d1.p_comp_target + d1.p_target_comp + d2.p_comp_target + d2.p_target_comp)) "
                             "as I3D_avg_Prediction "
                             "from df as d1, df as d2 "
                             "where d1.Target = d2.Target and d1.Comparison<d2.Comparison", env=locals())

    # write to disk
    open('I3D_ModelPredictions.txt', 'w').write(predictions.to_string())
    open('../../R/BDAoOSS_Synthetic/I3D_ModelPredictions.txt', 'w').write(predictions.to_string())
    open('I3D_ModelDistances.txt', 'w').write(df.to_string())

