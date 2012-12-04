#!/usr/bin/env python
# encoding: utf-8
"""

    this file will train models for different parameters and store
    them in pickles for later plotting

    * one pickle always contains a dict of models for different feature thresholds and glomeruli
    * each model is annotated with its settings
    * for each pickle we also write a json file containing the settings in a human readable format

    TODO: maybe make it work on job files in a folder (batch mode)

    List of parameters that can be explored with this file

    * parameters for each ML method (e.g. C for the SVR)
    * different features
    * spectral features with or without use_intensity
    * feature selection threshold
    * different feature selection methods
    * depth of a tree
    * combination of different spectral bands (kernel_widths)
    * per sample normalization


Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""


"""
    !!! move everything in small functions, with many options this script might
    !!! become very complex later
"""
import sys
import os
import json
import pickle
import time
from collections import defaultdict
from master.libs import read_data_lib as rdl
from master.libs import features_lib as flib
from master.libs import learning_lib as llib
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.preprocessing import normalize
import numpy as np
reload(rdl)
reload(flib)
reload(llib)

ml_methods = {'forest': RandomForestRegressor,
              'svr': llib.MySVR,
              'svr_ens': llib.SVREnsemble}

def prepare_features(config):
    """load and prepare the features, either conventional or spectral"""

    if config['features']['type'] == 'conventional':
        feature_file = os.path.join(config['data_path'], 'conventional_features',
                                    config['features']['descriptor'] + '.csv')
        features = flib.read_feature_csv(feature_file)
    elif config['features']['type'] == 'spectral':
        feature_file = os.path.join(config['data_path'], 'spectral_features',
                                    config['features']['descriptor'], 'parsed.pckl')
        spectra = pickle.load(open(feature_file))
        features = flib.get_spectral_features(spectra, config['features']['resolution'],
                                              spec_type=config['features']['spec_type'],
                                              use_intensity=config['features']['use_intensity'],
                                              kernel_widths=config['features']['kernel_width'])
    features = flib.remove_invalid_features(features)
    if config['features']['properties_to_add']:
        flib.add_molecule_properties(features, config['features']['properties_to_add'])
    if config['features']['normalize']:
        features = flib.normalize_features(features)
    return features


def load_data_targets(config, features):
    """load the targets for a glomerulus"""
    door2id = json.load(open(os.path.join(config['data_path'], 'door2id.json')))
    csv_path = os.path.join(config['data_path'], 'response_matrix.csv')
    cas_numbers, glomeruli, rm = rdl.load_response_matrix(csv_path, door2id)
    glom_idx = glomeruli.index(config['glomerulus'])

    # select molecules available for the glomerulus
    targets , tmp_cas_numbers = rdl.get_avail_targets_for_glom(rm, cas_numbers, glom_idx)
    molids = [str(door2id[cas_number][0]) for cas_number in tmp_cas_numbers]
    assert len(molids) == len(targets)

    # for some of them the spectra are not available
    avail = [i for i in range(len(molids)) if molids[i] in features]
    targets = np.array([targets[i] for i in avail])
    data = np.array([features[molids[i]] for i in avail])
    assert targets.shape[0] == data.shape[0]
    return data, targets, molids


def get_selection_score(config, data, targets):
    """select k_best features"""
    # feature selection
    if config['feature_selection']['method'] == 'linear':
        sel_scores, _ = f_regression(data, targets)
    elif config['feature_selection']['method'] == 'forest':
        rfr_sel = RandomForestRegressor(compute_importances=True, random_state=0)
        sel_scores = rfr_sel.fit(data, targets).feature_importances_
    return sel_scores


def run_runner(config, data, targets, get_models=False):
    """docstring for run"""

    res = {}
    if config['features']['normalize_samples']:
        data = normalize(data, norm='l2', axis=1, copy=True)
    if config['randomization_test']:
        np.random.seed()
        map(np.random.shuffle, data.T)

    # train models and get results
    for method in config['methods']:
        method_params = config['methods'][method]
        for p in method_params:
            if isinstance(method_params[p], unicode):
                method_params[p] = str(method_params[p])
        regressor = ml_methods[method](**method_params)
        regressor.fit(data, targets)
        res[method] = {'train_score': regressor.score(data, targets),
                       'gen_score': regressor.oob_score_}
        if get_models:
            res[method]['model'] = regressor
    return res


if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    features = prepare_features(config)
    res = run_runner(config, features)
    timestamp = time.strftime("%d%m%Y_%H%M%S", time.localtime())
    json.dump(res, open(os.path.join(config['results_path'], timestamp + '.json'), 'w'))
    json.dump(config, open(os.path.join(config['results_path'], timestamp + '_config.json'), 'w'))

