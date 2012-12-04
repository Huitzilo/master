#!/usr/bin/env python
# encoding: utf-8
'''

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''
import glob
import sys
import os
import json
from master.libs import run_lib
from master.libs import features_lib as flib
from master.libs import read_data_lib as rdl
from master.libs import plot_lib as plib
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
reload(run_lib)
reload(flib)
reload(plib)

plt.close('all')
active_thresh = 0.5
outpath = '/Users/dedan/projects/master/results/prediction'
glomeruli = ['Or10a', 'Or42b', 'Or47b']

config = json.load(open(sys.argv[1]))
door2id = json.load(open(os.path.join(config['data_path'], 'door2id.json')))
daniel_set = json.load(open(os.path.join(config['data_path'], 'daniel_set.json')))
daniel_set_molid = [door2id[cas][0] for cas in daniel_set]
features = run_lib.prepare_features(config)

for glom in glomeruli:

    config['glomerulus'] = glom
    data, targets, molids = run_lib.load_data_targets(config, features)

    fig = plt.figure(figsize=(5,5))
    active_targets = np.where(targets > active_thresh)[0]
    plib.structure_plot(fig, [molids[i] for i in active_targets],
                             targets[active_targets])
    fig.suptitle(glom)
    fig.savefig(os.path.join(outpath, glom + '.png'))
plt.show()

sel_scores = run_lib.get_selection_score(config, data, targets)
data = flib.select_k_best(data, sel_scores, config['feature_selection']['k_best'])
tmp_res = run_lib.run_runner(config, data, targets, get_models=True)
model = tmp_res['svr_ens']['model']

molids_to_predict = list(set(daniel_set_molid).difference(molids))
data_to_predict = np.array([features[m] for m in molids_to_predict if len(features[m]) != 0 ])
molids_to_predict = np.array([m for m in molids_to_predict if len(features[m]) != 0 ])
data_to_predict = flib.select_k_best(data_to_predict, sel_scores, config['feature_selection']['k_best'])
assert len(data_to_predict) == len(molids_to_predict)
predictions = model.predict(data_to_predict)

fig = plt.figure()
active_predictions = np.where(predictions > active_thresh)[0]
plib.structure_plot(fig, [molids_to_predict[i] for i in active_predictions],
                         predictions[active_predictions])

fig = plt.figure()
ax = fig.add_subplot(211)
ax.hist(targets)
ax.set_xlim([0,1])
ax = fig.add_subplot(212)
ax.hist(predictions)
ax.set_xlim([0,1])
plt.show()

