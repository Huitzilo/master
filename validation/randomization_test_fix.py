#!/usr/bin/env python
# encoding: utf-8
"""
    do a randomization test on a feature_selection-preprocessing-model combination

    by shuffling the data within columns and the re-evaluating the result N times.
    The idea is that the result should be much worse for shuffled data because
    the observations should now be meaningless and not helpful to predict any
    target values. It would only perform equally well if the unshuffled
    observations were already meaningless (without target related structure)

    this version of the script is for the case where we don't do any
    parameter search, instead we just train the model with fixed values

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import json
import sys
import os
import time
from master.libs import run_lib
from master.libs import read_data_lib as rdl
import numpy as np
import pylab as plt
reload(run_lib)

plt.close('all')

# repeat paramsearch N times with randomization_test set to true
# only for one descriptor
# record all the results from the different runs
sc = json.load(open(sys.argv[1]))
config = sc['runner_config_content']
config['data_path'] = os.path.join(os.path.dirname(__file__), '..', 'data')
all_glom_path = os.path.join(config['data_path'], 'all_glomeruli.json')
sc['glomeruli'] = json.load(open(all_glom_path))
json.dump(sc, open(os.path.join(sc['outpath'], 'sc_config.json'), 'w'))
assert len(config['methods']) == 1  # only one method a time
method = config['methods'].keys()[0]

# load the features
features = run_lib.prepare_features(config)
n_features = len(features[features.keys()[0]])
config['feature_selection']['k_best'] = n_features

print('true run')
config['randomization_test'] = False
res = {}
for glomerulus in sc['glomeruli']:
    config['glomerulus'] = glomerulus
    data, targets, molids = run_lib.load_data_targets(config, features)
    tmp_res = run_lib.run_runner(config, data, targets, get_models=True)
    res[glomerulus] = tmp_res[method]['gen_score']
json.dump(res, open(os.path.join(sc['outpath'], 'true.json'), 'w'))

# add shuffle data to the config and run the runner for N times
config['randomization_test'] = True
for i in range(sc['n_repetitions']):
    print('randomized run nr: {}'.format(i+1))
    res = {}
    for glomerulus in sc['glomeruli']:
        config['glomerulus'] = glomerulus
        data, targets, molids = run_lib.load_data_targets(config, features)
        tmp_res = run_lib.run_runner(config, data, targets, get_models=True)
        res[glomerulus] = tmp_res[method]['gen_score']
    run_name = 'run_' + time.strftime("%d%m%Y_%H%M%S", time.localtime())
    json.dump(res, open(os.path.join(sc['outpath'], run_name + '.json'), 'w'))

