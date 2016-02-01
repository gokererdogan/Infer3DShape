"""
Inferring 3D Shape from 2D Images

This file contains the script for calculating predictions of the 3D ideal observer model that assumes 3D shape is
extracted perfectly from 2D images.
WARNING: For pickle to run properly (import the necessary modules), run this script from the root Infer3DShape folder.

Created on Feb 1, 2016

Goker Erdogan
https://github.com/gokererdogan/
"""

from Infer3DShape.similarity.calculate_similarity import *

if __name__ == "__main__":
    import numpy as np
    import cPickle as pkl
    import pandas as pd
    import pandasql as psql
    import vision_forward_model as vfm

    fwm = vfm.VisionForwardModel()

    data_folder = "./data/stimuli20150624_144833"

    # load the 3D shapes for the stimuli
    gt_shapes = pkl.load(open(data_folder + '/shapes_single_view.pkl'))

    objects = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10']
    transformations = ['t1_cs_d1', 't1_cs_d2', 't2_ap_d1', 't2_ap_d2', 't2_mf_d1', 't2_mf_d2', 't2_rp_d1', 't2_rp_d2']
    comparisons = {o: [o + '_' + t for t in transformations] for o in objects}

    df = pd.DataFrame(index=np.arange(0, 8 * len(objects)), columns=['Target', 'Comparison',
                                                                     'p_comp_target', 'p_target_comp', 'p_avg'])

    i = 0
    for obj in objects:
        print(obj)
        # load the target object data and samples
        obj_data = np.load('{0:s}/{1:s}_single_view.npy'.format(data_folder, obj))

        obj_sample = gt_shapes[obj]
        obj_sample.forward_model = fwm

        for comparison in comparisons[obj]:
            print("\t{0:s}".format(comparison))
            comp_data = np.load('{0:s}/{1:s}_single_view.npy'.format(data_folder, comparison))

            comp_sample = gt_shapes[comparison]
            comp_sample.forward_model = fwm

            p_comp_target, discard = calculate_similarity(comp_data, [obj_sample], [0.0])
            p_target_comp, discard = calculate_similarity(obj_data, [comp_sample], [0.0])
            df.loc[i] = [obj, comparison, p_comp_target, p_target_comp, (p_comp_target + p_target_comp) / 2.0]
            i += 1

    predictions = psql.sqldf("select d1.Comparison as Comparison1, d2.Comparison as Comparison2, "
                             "d1.p_comp_target > d2.p_comp_target as I3D_pcomp_GT_Prediction, "
                             "d1.p_target_comp > d2.p_target_comp as I3D_ptarget_GT_Prediction, "
                             "d1.p_avg > d2.p_avg as I3D_pavg_GT_Prediction "
                             "from df as d1, df as d2 where d1.Target = d2.Target and d1.Comparison < d2.Comparison",
                             env=locals())
    # write to disk
    open('I3D_GT_ModelPredictions.txt', 'w').write(predictions.to_string())
    open('I3D_GT_ModelDistances.txt', 'w').write(df.to_string())
