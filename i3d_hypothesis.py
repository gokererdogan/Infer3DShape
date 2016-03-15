"""
Inferring 3D Shape from 2D Images

This file contains the base hypothesis class for Infer3DShape package.
All hypothesis classes are derived from this base class.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""

from copy import deepcopy

import mcmclib.hypothesis as hyp
import Infer3DShape.i3d_likelihood as ll

# assuming that pixels ~ unif(0,1), expected variance of a pixel difference is 1/6
LL_VARIANCE = 0.0001 # in squared pixel distance
MAX_PIXEL_VALUE = 177.0 # this is usually 256.0 but in our case because of the lighting in our renders, it is lower
# sigma for the Gaussian filter used in likelihood_pixel_gaussian_filtered
LL_FILTER_SIGMA = 2.0


class I3DHypothesis(hyp.Hypothesis):
    """Base hypothesis class for hypotheses in Infer3DShape package.

    This hypothesis class brings together the common attributes and methods used by all hypothesis classes.

    Attributes:
        forward_model (VisionForwardModel): Forward model for rendering hypothesis
        viewpoint (list of 3-tuple): Viewpoints (given in spherical coordinates) from which the object is viewed.
            Each element is (r, theta, phi) where
                r is the distance to origin, theta is the angle in the xy plane and phi is the angle away from the
                z axis. Note that we are using the mathematics (not the physics) convention here.
            If not provided, viewpoint from forward_model is used.
        params (dict): A dictionary of parameters (generally parameters related to likelihood calculations)
        primitive_type (string): Type of primitive shape is made up from. It can be either CUBE or TUBE at
            the moment. This information is used by VisionForwardModel to render the object.
    """
    def __init__(self, forward_model, viewpoint=None, params=None, primitive_type='CUBE'):
        hyp.Hypothesis.__init__(self)
        self.forward_model = forward_model
        self.viewpoint = viewpoint
        self.primitive_type = primitive_type
        self.params = params
        # if params is not provided, use the default values
        if self.params is None:
            self.params = {'LL_VARIANCE': LL_VARIANCE, 'MAX_PIXEL_VALUE': MAX_PIXEL_VALUE,
                           'LL_FILTER_SIGMA': LL_FILTER_SIGMA}
        else:
            # if any of the parameters not in params, assign default values
            if 'LL_VARIANCE' not in self.params.keys():
                self.params['LL_VARIANCE'] = LL_VARIANCE
            if 'MAX_PIXEL_VALUE' not in self.params.keys():
                self.params['MAX_PIXEL_VALUE'] = MAX_PIXEL_VALUE
            if 'LL_FILTER_SIGMA' not in self.params.keys():
                self.params['LL_FILTER_SIGMA'] = LL_FILTER_SIGMA

    def _calculate_log_likelihood(self, data=None):
        """Calculates log likelihood of hypothesis given data.

        Overrides the method from base Hypothesis class. Right now, we use a pixel based likelihood model that assumes
        Gaussian noise.

        Parameters:
            data (numpy.ndarray): Observed image

        Returns:
            float: log likelihood
        """
        return ll.log_likelihood_pixel(self, data, self.params['MAX_PIXEL_VALUE'], self.params['LL_VARIANCE'])

    def convert_to_positions_sizes(self):
        """Convert the shape hypothesis to lists of positions and sizes of each part.

        This method is called by forward models to get positions and sizes of each part of the shape hypothesis. This
            method should be overridden in child classes.

        Returns:
            (list, list): A tuple of two lists for positions and sizes respectively
        """
        raise NotImplementedError()

    def copy(self):
        viewpoint_copy = deepcopy(self.viewpoint)
        return I3DHypothesis(forward_model=self.forward_model, viewpoint=viewpoint_copy, params=self.params)

    def __getstate__(self):
        """Return object state for pickling.

        We cannot pickle VisionForwardModel instances, so we need to get rid of them before pickling.

        Returns:
            dict: Object data without forward model
        """
        # we cannot pickle VTKObjects, so get rid of them.
        return {k: v for k, v in self.__dict__.iteritems() if k != 'forward_model'}

