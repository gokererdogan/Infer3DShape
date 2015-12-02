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

def calculate_prob(data, samples, probs, integrate_over_viewpoint=False, use_only_MAP=False):
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
    :param integrate_over_viewpoint: Integrate out viewpoint. If False, we calculate likelihood from the
            default viewpoint.
    :param use_only_MAP: Use only the MAP sample
    :return: Two different probability estimates:
                1) Probability of observing data from samples calculated integrating out viewpoint
                2) Probability of observing data from samples calculated picking the best viewpoint
    """
    probs /= np.sum(probs)

    if use_only_MAP:
        map_id = np.argmax(probs)
        sample = samples[map_id]
        return _prob_image_given_hypothesis(data, sample, integrate_over_viewpoint)
    else:
        sim_mean = 0.0
        sim_max = 0.0
        for sample, prob in zip(samples, probs):
            sample.ll = None
            ll_mean, ll_max = _prob_image_given_hypothesis(data, sample, integrate_over_viewpoint)
            sim_mean += (prob * ll_mean)
            sim_max += (prob * ll_max)
        return sim_mean, sim_max

def _prob_image_given_hypothesis(image, hypothesis, integrate_over_viewpoint=False):
    """
    Calculate probability of image given hypothesis, p(I|H). We can integrate over viewpoint
    if we desire. In that case, we calculate p(I|H) = sum_theta p(I, theta| H) where we assume
    a uniform distribution over theta. Instead of integrating, we can pick the viewpoint with
    the maximum likelihood. This function returns both.
    :param image:
    :param hypothesis:
    :param integrate_over_viewpoint:
    :return: sum_theta p(I, theta|H), max_theta p(I,theta|H)
    """
    if not integrate_over_viewpoint:
        # if we are not integrating over viewpoint, mean and max does not mean anything.
        hypothesis.ll = None
        ll = hypothesis.likelihood(image)
        return ll, ll

    assert hypothesis.viewpoint is not None
    tot_ll = 0.0
    max_ll = 0.0
    # rotate the hypothesis and calculate likelihood for each viewpoint.
    x, y, z = hypothesis.viewpoint[0]
    d = np.sqrt(x**2 + y**2)
    for theta in range(0, 360, 2):
        nx = d * np.cos(theta * np.pi / 180.0)
        ny = d * np.sin(theta * np.pi / 180.0)
        hypothesis.viewpoint[0] = (nx, ny, z)
        # we want the likelihood to be calculated from scratch
        hypothesis.ll = None
        ll = hypothesis.likelihood(image)
        tot_ll += ll
        if ll > max_ll:
            max_ll = ll

    # average total likelihood (we have 180 viewpoints)
    tot_ll /= 180.0

    return tot_ll, max_ll


if __name__ == "__main__":
    fwm = vfm.VisionForwardModel()

    single_view = True

    if single_view:
        append_str = '_single_view'
        integrate_over_viewpoint = True
        fwm = vfm.VisionForwardModel(render_size=(200, 200))
    else:
        append_str = ''
        integrate_over_viewpoint = False
        fwm = vfm.VisionForwardModel(render_size=(100, 100))

    data_folder = "./data/stimuli20150624_144833"
    samples_folder = "./results/bdaoossShapeMaxD/20151018"

    df = pd.DataFrame(index=np.arange(0, 8), columns=['Target', 'Comparison',
                                                      'p_comp_target', 'p_comp_target_max',
                                                      'p_comp_target_MAP', 'p_comp_target_MAP_max',
                                                      'p_target_comp', 'p_target_comp_max',
                                                      'p_target_comp_MAP', 'p_target_comp_MAP_max'])

    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    transformations = ['t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_mf_d1', 't2_mf_d2', 't2_rp_d1', 't2_rp_d2']

    i = 0
    for obj in objects:
        print(obj)
        # load the target object data and samples
        obj_run = pkl.load(open("{0:s}/{1:s}.pkl".format(samples_folder, obj)))
        obj_data = np.load('{0:s}/{1:s}{2:s}.npy'.format(data_folder, obj, append_str))

        obj_best_samples = obj_run.best_samples.samples
        for sample in obj_best_samples:
            sample.forward_model = fwm

        obj_probs = obj_run.best_samples.probs

        for transformation in transformations:
            print("\t{0:s}".format(transformation))
            comparison = "{0:s}_{1:s}".format(obj, transformation)
            comp_run = pkl.load(open("{0:s}/{1:s}.pkl".format(samples_folder, comparison)))
            comp_data = np.load('{0:s}/{1:s}{2:s}.npy'.format(data_folder, comparison, append_str))

            comp_best_samples = comp_run.best_samples.samples
            for sample in comp_best_samples:
                sample.forward_model = fwm

            comp_probs = comp_run.best_samples.probs

            p_comp_target, p_comp_target_max = calculate_prob(comp_data, obj_best_samples, obj_probs,
                                                              integrate_over_viewpoint, use_only_MAP=False)
            p_comp_target_MAP, p_comp_target_MAP_max = calculate_prob(comp_data, obj_best_samples, obj_probs,
                                                                      integrate_over_viewpoint, use_only_MAP=True)
            p_target_comp, p_target_comp_max = calculate_prob(obj_data, comp_best_samples, comp_probs,
                                                              integrate_over_viewpoint, use_only_MAP=False)
            p_target_comp_MAP, p_target_comp_MAP_max = calculate_prob(obj_data, comp_best_samples, comp_probs,
                                                                      integrate_over_viewpoint, use_only_MAP=True)
            df.loc[i] = [obj, comparison, p_comp_target, p_comp_target_max, p_comp_target_MAP, p_comp_target_MAP_max,
                         p_target_comp, p_target_comp_max, p_target_comp_MAP, p_target_comp_MAP_max]
            i += 1

    # calculate model predictions
    predictions = psql.sqldf("select d1.Comparison as Comparison1, d2.Comparison as Comparison2, "
                             "(d1.p_comp_target / (d1.p_comp_target + d2.p_comp_target)) as I3D{0:s}_pcomp_Prediction, "
                             "(d1.p_comp_target_max / (d1.p_comp_target_max + d2.p_comp_target_max)) "
                             "as I3D{0:s}_pcomp_max_Prediction, "
                             "(d1.p_comp_target_MAP / (d1.p_comp_target_MAP + d2.p_comp_target_MAP)) "
                             "as I3D{0:s}_pcomp_MAP_Prediction, "
                             "(d1.p_comp_target_MAP_max / (d1.p_comp_target_MAP_max + d2.p_comp_target_MAP_max)) "
                             "as I3D{0:s}_pcomp_MAP_max_Prediction, "
                             "(d1.p_target_comp / (d1.p_target_comp + d2.p_target_comp)) "
                             "as I3D{0:s}_ptarget_Prediction, "
                             "(d1.p_target_comp_max / (d1.p_target_comp_max + d2.p_target_comp_max)) "
                             "as I3D{0:s}_ptarget_max_Prediction, "
                             "(d1.p_target_comp_MAP / (d1.p_target_comp_MAP + d2.p_target_comp_MAP)) "
                             "as I3D{0:s}_ptarget_MAP_Prediction, "
                             "(d1.p_target_comp_MAP_max / (d1.p_target_comp_MAP_max + d2.p_target_comp_MAP_max)) "
                             "as I3D{0:s}_ptarget_MAP_max_Prediction, "
                             "((d1.p_comp_target + d1.p_target_comp) / "
                             "(d1.p_comp_target + d1.p_target_comp + d2.p_comp_target + d2.p_target_comp)) "
                             "as I3D{0:s}_avg_Prediction, "
                             "((d1.p_comp_target_max + d1.p_target_comp_max) / "
                             "(d1.p_comp_target_max + d1.p_target_comp_max + "
                             "d2.p_comp_target_max + d2.p_target_comp_max)) "
                             "as I3D{0:s}_avg_max_Prediction, "
                             "((d1.p_comp_target_MAP + d1.p_target_comp_MAP) / "
                             "(d1.p_comp_target_MAP + d1.p_target_comp_MAP + "
                             "d2.p_comp_target_MAP + d2.p_target_comp_MAP)) "
                             "as I3D{0:s}_avg_MAP_Prediction, "
                             "((d1.p_comp_target_MAP_max + d1.p_target_comp_MAP_max) / "
                             "(d1.p_comp_target_MAP_max + d1.p_target_comp_MAP_max + "
                             "d2.p_comp_target_MAP_max + d2.p_target_comp_MAP_max)) "
                             "as I3D{0:s}_avg_MAP_max_Prediction "
                             "from df as d1, df as d2 "
                             "where d1.Target = d2.Target and d1.Comparison<d2.Comparison".format(append_str), env=locals())

    # write to disk
    open('I3D{0:s}_ModelPredictions.txt'.format(append_str), 'w').write(predictions.to_string())
    # open('../../R/BDAoOSS_Synthetic/I3D{0:s}_ModelPredictions.txt'.format(append_str), 'w').write(predictions.to_string())
    open('I3D{0:s}_ModelDistances.txt'.format(append_str), 'w').write(df.to_string())

