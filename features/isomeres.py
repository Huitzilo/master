#!/usr/bin/env python
# encoding: utf-8
"""
measure for the different features the average similarity vs. stereoisomere similarity

We did this to see which descriptor is susceptible to 3D molecule configurations.
The problem is that in the experiments of the DoOr database they often used
mixtures of chemicals

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import json, os, csv, glob, json
import feature_lib as fl
import pylab as plt
import numpy as np
from collections import defaultdict
import __builtin__
from scipy.spatial.distance import pdist
reload(fl)

features_path = '/Users/dedan/projects/master/data/features/'
out_path = '/Users/dedan/projects/master/results/features'
N_BINS = 100

# mapping from the CAS number to jan's mol ID
door2id = json.load(open(os.path.join(features_path, 'door2id.json')))

if os.path.exists(os.path.join(features_path, 'features.json')):
    print('read previously computed features from: %s JSON' % features_path)
    features = json.load(open(os.path.join(features_path, 'features.json')))
else:
    print('read features from csv, normalize and save them')
    features = fl.normalize_features(fl.read_feature_csvs(features_path))
    json.dump(features, open(os.path.join(features_path, 'features.json'), 'w'))

# identify molecules with and without isomeres
molecules = [molids[0] for molids in door2id.values() if len(molids) == 1]
isomeres = [molids for molids in door2id.values() if len(molids) > 1]

# create similarity histogram plots for all descriptors
for f_space in features:

    fig = plt.figure()
    print 'working on: ', f_space
    ax = fig.add_subplot(111)
    mol_fspace = fl.get_features_for_molids(features[f_space], molecules)
    ax.hist(pdist(mol_fspace), bins=N_BINS, log=True)

    isomere_distances = np.array([])
    for isomere in isomeres:
        iso_fspace = fl.get_features_for_molids(features[f_space], isomere)
        if iso_fspace.any():
            isomere_distances = np.hstack([isomere_distances, pdist(iso_fspace)])

    print isomere_distances.shape
    _, _, patches = ax.hist(isomere_distances, bins=N_BINS, log=True)
    for patch in patches:
        plt.setp(patch, color="r")
    fig.savefig(os.path.join(out_path, f_space + '_histo.png'))
plt.close('all')
