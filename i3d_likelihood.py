"""
Inferring 3D Shape from 2D Images

This file contains implementations of various likelihood functions.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np
import scipy.ndimage as spi


def log_likelihood_pixel_gaussian_filtered(shape, data, max_pixel_value, ll_variance, filter_sigma):
    prediction = shape.forward_model.render(shape)
    prediction = spi.gaussian_filter(prediction, filter_sigma)
    mse = np.sum(np.square((prediction - data) / max_pixel_value)) / prediction.size
    ll = -mse / (2 * ll_variance)
    return ll


def log_likelihood_pixel(shape, data, max_pixel_value, ll_variance):
    """Calculate the Gaussian log likelihood in pixel space.

    Calculates the probability of observing `data` given hypothesis `shape` according to a Gaussian model.

    Parameters:
        shape (I3DHypothesis): Shape hypothesis
        data (numpy.ndarray): Observed image
        max_pixel_value (float): Maximum pixel intensity value in images. Used for normalization.
        ll_variance (float): Variance of the Gaussian.

    Returns:
        (float): log probability log p(data|shape).
    """
    prediction = shape.forward_model.render(shape)
    mse = np.sum(np.square((prediction - data) / max_pixel_value)) / prediction.size
    ll = -mse / (2 * ll_variance)
    return ll

