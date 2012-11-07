#!/usr/bin/env python
# encoding: utf-8
"""
    functions to create spektral features or do any other feature transforms

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import __builtin__
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def get_k_best(scores, k):
    """get indices for the k best features depending on the scores"""
    assert not (scores < 0).any()
    assert len(scores) >= k
    scores[np.isnan(scores)] = 0
    return np.argsort(scores)[-k:]

def get_spectral_features(spectra, resolution, use_intensity=True,
                                               spec_type='ir',
                                               kernel_widths=1):
    """bining after convolution

        combine several binings if kernel_width is a list of widths
    """
    if not isinstance(kernel_widths, list):
        kernel_widths = [kernel_widths]
    combined = np.zeros((len(spectra),0))
    for k_width in kernel_widths:
        as_vectors = _place_waves_in_vector(spectra, resolution, use_intensity, spec_type)
        as_vectors = gaussian_filter(as_vectors, [0, k_width], 0)
        bined = _bining(as_vectors, k_width)
        combined = np.hstack((combined, bined))
    features = {}
    for i, molid in enumerate(spectra):
        features[molid] = combined[i]
    assert(len(spectra) == len(features))
    return features

def _place_waves_in_vector(spectra, resolution, use_intensity, spec_type):
    """from gaussian we only get the wavenumbers, place them in vector for convolution"""
    all_freq = __builtin__.sum([spectra[molid]['freq'] for molid in spectra], [])
    max_freq = np.max(all_freq)

    x = np.zeros((len(spectra), int(np.ceil(max_freq/resolution)) + 1))
    for i, molid in enumerate(spectra):
        idx = np.round(np.array(spectra[molid]['freq']) / resolution).astype(int)
        if use_intensity:
            x[i, idx] = spectra[molid][spec_type]
        else:
            x[i, idx] = 1
    return x

def _bining(vectors, kernel_width):
    """divide the *continous* spectrum into bins"""
    factor = vectors.shape[1] / kernel_width
    rest = vectors.shape[1] % kernel_width
    if rest:
        return np.mean(vectors[:,:-rest].reshape((vectors.shape[0], factor, -1)), axis=2)
    else:
        return np.mean(vectors.reshape((vectors.shape[0], factor, -1)), axis=2)
