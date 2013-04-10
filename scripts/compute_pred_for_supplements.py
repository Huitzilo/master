#!/usr/bin/env python
# encoding: utf-8
"""
compute the predictions and save them for later usage

depends on the code from my master thesis

    github.com/dedan/master

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import json
import os
import numpy as np
from master.libs import read_data_lib as rdl

outfile = '/Users/dedan/projects/odor_app/static/data/response_matrix'
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
csvfile = os.path.join(data_path, 'response_matrix.csv')
molid2smile = json.load(open(os.path.join(data_path, 'molid2smile.json')))
door2id = json.load(open(os.path.join(data_path, 'door2id.json')))

cas_numbers, glomeruli, rm = rdl.load_response_matrix(csvfile)

rm_dict = {}
for i_glom, glom in enumerate(glomeruli):
    rm_dict[glom] = []
    for i_cas, cas in enumerate(cas_numbers):
        rm_dict[glom].append({
            'CAS': cas,
            'response': rm[i_cas, i_glom] if not np.isnan(rm[i_cas, i_glom]) else 'nan',
            'smile': molid2smile[door2id[cas][0]] if door2id[cas] else ''
        })
json.dump(rm_dict, open(outfile + '.json', 'w'))

out_csv = open(outfile + '.csv', 'w')
out_csv.write(', ' + ', '.join(glomeruli) + '\n')
for cas, rm_row in zip(cas_numbers, rm):
    smile = molid2smile[door2id[cas][0]] if door2id[cas] else 'smile not available - CAS: {}'.format(cas)
    out_csv.write(smile + ', ' + ', '.join([str(r) for r in rm_row.tolist()]) + '\n')
