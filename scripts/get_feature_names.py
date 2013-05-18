#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import os
import numpy as np
from collections import defaultdict
import glob
import csv
import json


def prepare_features(config):
    """load and prepare the features, either conventional or spectral"""
    features_path = '/Users/dedan/projects/master/data/conventional_features/'
    all_features = defaultdict(lambda: np.array([]))
    feature_names = []
    feature_files = glob.glob(os.path.join(features_path, '*.csv'))
    for feature_file in feature_files:
        if 'haddad' in feature_file or 'saito' in feature_file:
            continue
        features, header = read_feature_csv(feature_file)
        feature_names.extend(header)
        for key, value in features.items():
            all_features[key] = np.hstack((all_features[key], value))
    assert len(all_features[key]) == len(feature_names)

    max_len = max([len(val) for val in all_features.values()])
    for key in list(all_features.keys()):
        if len(all_features[key]) < max_len:
            del all_features[key]

    valid = np.var(np.array(all_features.values()), axis=0) != 0.
    for feature in all_features.keys():
        all_features[feature] = all_features[feature][valid]
    feature_names = [h for i, h in enumerate(feature_names) if valid[i]]
    assert len(all_features[feature]) == len(feature_names)
    return feature_names


def read_feature_csv(feature_file):
    """read one feature CSV into a dictionary structure

    csvs have molid in the 1st column, identifiere in 2nd and then features
    """
    features = {}
    molid_idx = 0
    identifiere_idx = 1
    feature_start_idx = 2
    features = defaultdict(list)

    with open(feature_file) as f:
        reader = csv.reader(f)
        header = reader.next()

        for row in reader:
            if 'Error' in row[identifiere_idx]:
                continue
            mol = row[molid_idx]
            data_str = ','.join(row[feature_start_idx:])
            features[mol] = np.fromstring(data_str, dtype=float, sep=',')
        header = header[feature_start_idx:]
    mols = features.keys()
    for i in range(len(mols) - 1):
        assert(len(features[mols[i]]) == len(features[mols[i+1]]))
    return features, header

if __name__ == '__main__':
    config = {"features": {
        "type": "conventional",
        "descriptor": "all",
        "normalize": False,
        "properties_to_add": []
    }}

    header = prepare_features(config)
    json.dump({i: name for i, name in enumerate(header)}, open('feature_names.json', 'w'))

