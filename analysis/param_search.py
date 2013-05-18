#!/usr/bin/env python
# encoding: utf-8
'''
This is used to perform the parameter search.

An example config file is given in the config folder as param_search_base.json

* to run only some of the conventional descriptors but not all add a list
  containing the descriptors to the config
    {"descriptors": ["ATOMCENTRED_FRAGMENTS", "BLA_SUPER_DESCRIPTOR"]}
* to combine all conventional features into one huge descriptor add 'all' to list
    {"descriptors": ["all", "ATOMCENTRED_FRAGMENTS", "BLA_SUPER_DESCRIPTOR"]}

The results produced by this script can be analyzed using

    `analysis/plot/plot_paramsearch.py`

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''
import glob
import sys
import os
import json
import pickle
from master.libs import run_lib
from master.libs import features_lib as flib
from master.libs import utils
import numpy as np
import copy
reload(run_lib)
reload(flib)

def paramsearch(sc):
    configs = []
    # search config
    config = sc['runner_config_content']
    config['data_path'] = os.path.join(os.path.dirname(__file__), '..', 'data')

    if 'k_best' in sc:
        k_best_set = True
    else:
        k_best_set = False

    if config['features']['type'] == 'conventional':
        files = glob.glob(os.path.join(config['data_path'], 'conventional_features', '*.csv'))
        for f in files:
            desc = os.path.splitext(os.path.basename(f))[0]
            if 'descriptors' in sc and not desc in sc['descriptors']:
                continue
            config['features']['descriptor'] = desc
            config['run_name'] = desc
            configs.append(copy.deepcopy(config))
        if 'descriptors' in sc and 'all' in sc['descriptors']:
            print('using all descriptors together')
            config['features']['descriptor'] = 'all'
            config['run_name'] = 'all'
            configs.append(copy.deepcopy(config))
    elif config['features']['type'] == 'spectral':
        config['features']['descriptor'] = 'large_base'
        for kwidth in sc['kernel_widths']:
            config['features']['kernel_width'] = kwidth
            config['features']['bin_width'] = int(np.min(kwidth) * sc['sigma_l_ratio'])
            config['run_name'] = repr(kwidth)
            configs.append(copy.deepcopy(config))
    else:
        assert False

    for config in configs:

        cur_outpath = os.path.join(sc['outpath'], config['run_name'] + '.json')
        sc['runner_config_content'] = config
        # if result file already exists, load it to append new glomeruli
        if os.path.exists(cur_outpath):
            print('load existing results from: {}'.format(config['run_name']))
            res = json.load(open(cur_outpath))["res"]
        else:
            res = {sel: {} for sel in sc['selection']}

        # load the features
        print('preparing features..')
        features = run_lib.prepare_features(config)
        n_features = len(features[features.keys()[0]])

        # set k_best according to the number of available features
        max_expo = int(np.floor(np.log2(n_features))) + 1

        if not k_best_set:
            sc['k_best'] = [2**i for i in range(max_expo)]
            if sc['k_best'][-1] != n_features:
                sc['k_best'] += [n_features]
        else:
            sc['k_best'] = [n_features]

        # and the forest regularization parameter (als n_features dependend)
        sc['forest'] = range(2, max_expo + 2, 2)

        print 'working on: ', config['run_name']
        for selection in sc['selection']:
            print selection
            config['feature_selection']['method'] = selection
            for glomerulus in sc['glomeruli']:
                if not glomerulus in res[selection]:
                    res[selection][glomerulus] = {}
                config['glomerulus'] = glomerulus
                print('param search for {}..'.format(glomerulus))
                res[selection][glomerulus] = run_lib.do_paramsearch(sc, config, features, res[selection][glomerulus])
                print('param search for {} done'.format(glomerulus))
                if sc['get_models']:
                    cur_outpath_pckl = os.path.splitext(cur_outpath)[0] + '_' + glomerulus + '.pckl'
                    pickle.dump(res, open(cur_outpath_pckl, 'w'))
                json_res = utils.nested_remove_keys(copy.deepcopy(res), 'model')
                json.dump({'sc': sc, 'res': json_res}, open(cur_outpath+ '_' + glomerulus, 'w'))
    return 'fertig: ' + glomerulus

if __name__ == '__main__':
    sc = json.load(open(sys.argv[1]))
    paramsearch(sc)
