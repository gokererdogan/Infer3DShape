"""
Inferring 3D Shape from 2D Images

This file contains implementations of various likelihood functions.

Created on Dec 2, 2015

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np
import scipy.ndimage as spi


def log_likelihood_pixel_gaussian_filtered(prediction, data, max_pixel_value, ll_variance, filter_sigma):
    prediction = spi.gaussian_filter(prediction, filter_sigma)
    mse = np.sum(np.square((prediction - data) / max_pixel_value)) / prediction.size
    ll = -mse / (2 * ll_variance)
    return ll


def log_likelihood_pixel(prediction, data, max_pixel_value, ll_variance):
    mse = np.sum(np.square((prediction - data) / max_pixel_value)) / prediction.size
    ll = -mse / (2 * ll_variance)
    return ll

