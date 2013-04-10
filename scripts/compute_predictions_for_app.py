#!/usr/bin/env python
# encoding: utf-8
"""
compute the predictions and save them for later usage

depends on the code from my master thesis

    github.com/dedan/master

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os

#!/usr/bin/env python
# encoding: utf-8
'''

compute predictions to compare the different models for one glomerulus

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''
import sys
import os
import json
import pickle
import itertools as it
from master.libs import run_lib as rl
from master.libs import utils
import numpy as np
import pybel
import copy
reload(rl)

outpath = '/Users/dedan/projects/odor_app/models'
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
mol_file = '/Users/dedan/projects/master/data/molecules.sdf'
molid2smile = json.load(open(os.path.join(data_path, 'molid2smile.json')))
gloms = json.load(open(os.path.join(data_path, 'all_glomeruli.json')))
print mol_file
molecules = pybel.readfile('sdf', mol_file)
id2desc = {}
for m in molecules:
    data = dict(m.data)
    id2desc[m.data['CdId']] = { 'name': data.get('Name', '')
                               , 'CID': data.get('PubChem CID', '')}
base_config = {
    "data_path": data_path,
    "feature_selection": {
        "k_best": "max",
        "method": "linear"
        },
    "features": {
        "normalize": True,
        "properties_to_add": []
    },
    "methods": {
        "svr": {
            "C": 1.0,
            "n_folds": 50
      }
    },
    "randomization_test": False
}

feat_config = {
    "haddad": {
        "type": "conventional",
        "descriptor": "haddad_desc",
    },
    "all": {
        "type": "conventional",
        "descriptor": "all",
    },
    "eva_100": {
      "type": "spectral",
      "kernel_width": 100,
      "bin_width": 150,
      "use_intensity": False,
      "spec_type": "ir",
    }
}
method = base_config['methods'].keys()[0]


# compute molecules available for all descriptors
cache = {}
for k, config in feat_config.items():
    base_config['features'].update(config)
    cache[k] = {"features": rl.prepare_features(base_config)}
all_mols = [r['features'].keys() for r in cache.values()]
mol_intersection = set(all_mols[0]).intersection(*all_mols[1:])
for k in cache:
    cache[k]['features'] = {m: cache[k]['features'][m] for m in mol_intersection}

res = {n: {g: {} for g in gloms} for n in feat_config}
for glom in gloms:
    print('{}\n'.format(glom))
    base_config.update({'glomerulus': glom, 'data_path': data_path})

    dtm = {}
    for name, config in feat_config.items():
        base_config['features'].update(config)
        data, targets, molids = rl.load_data_targets(base_config, cache[name]['features'])
        dtm[name] = {
            'data': data,
            'targets': targets,
            'molids': molids
        }

    for name, data in dtm.items():

        # fit model
        print('working on model: {}'.format(name))
        base_config['feature_selection']['k_best'] = data['data'].shape[1]
        print("use {} molecules for training".format(data['data'].shape[0]))
        tmp_res = rl.run_runner(base_config, data['data'], data['targets'], get_models=True)

        res[name][glom]['predictions'] = []
        molid2oob = {}
        for i, molid in enumerate(data['molids']):
            molid2oob[molid] = {}
            molid2oob[molid]['oob_mean'] = round(float(np.mean([x[0] for x in tmp_res[method]['model'].all_predictions[i]])), 3)
            molid2oob[molid]['oob_var'] = round(float(np.var([x[0] for x in tmp_res[method]['model'].all_predictions[i]])), 3)

        preds = tmp_res[method]['model'].predict(np.array(cache[name]['features'].values()))
        for i, (m, p) in enumerate(zip(cache[name]['features'].keys(), preds)):
            tmp = copy.deepcopy(id2desc[m])
            tmp.update({
                'prediction': round(float(p), 3),
                'molid': m,
                'smile': molid2smile[m] if m in molid2smile else ''
            })

            if m in data['molids']:
                tmp['target'] = round(float(data['targets'][data['molids'].index(m)]), 3)
                tmp['oob_prediction'] = molid2oob[m]
            res[name][glom]['predictions'].append(tmp)
        if tmp_res[method]['gen_score'] > 0:
            res[name][glom]['score'] = round(float(tmp_res[method]['gen_score']), 3)
        else:
            res[name][glom]['score'] = 0

        print('model genscore: {:.2f}\n'.format(tmp_res[method]['gen_score']))

pickle.dump(dict(res), open(os.path.join(outpath, 'predictions.pkl'), 'w'))
