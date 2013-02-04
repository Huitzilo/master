#!/usr/bin/env python
# encoding: utf-8
"""
    plot: regularization on x axis, number of k_best features on y

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import sys
import os
import json
import numpy as np
import pylab as plt
import itertools as it
from master.libs import plot_lib as plib
from master.libs import read_data_lib as rdl
from master.libs import utils
import matplotlib.gridspec as gridspec
reload(plib)
reload(rdl)

config = json.load(open(sys.argv[1]))
outpath = os.path.join(config['inpath'], 'plots')
if not os.path.exists(outpath):
    os.mkdir(outpath)

# variables for results
plt.close('all')
search_res, max_overview, sc, _ = rdl.read_paramsearch_results(config['inpath'],
                                                               p_selection=config.get('selection', {}))

if config['plot_param_space']:
    for desc in search_res:
        fig = plt.figure(figsize=(7,10))
        plib.plot_search_matrix(fig, search_res[desc], sc, config['glomeruli'])
        fig.savefig(os.path.join(outpath, desc + '.' + config['format']))

# descriptor method performance plots
fig = plt.figure(figsize=(15,5))
plib.new_descriptor_performance_plot(fig, max_overview, sc,
                                     config.get('glomeruli', []), config['boxplot'])
fig.subplots_adjust(bottom=0.3)
fig.savefig(os.path.join(outpath, 'desc_comparison.' + config['format']))


# descriptor comparison plot for svr lin
desc2comp = ['haddad_desc', 'saito_desc', 'all', 'vib_50' ]#, 'vib_100']
mn = desc2comp
gs = gridspec.GridSpec(len(mn)-1, len(mn)-1)
gs.update(wspace=0.2, hspace=0.2)
cur_max = max_overview['forest']['forest']
for m1, m2 in it.combinations(mn, 2):
    ax = plt.subplot(gs[mn.index(m1), mn.index(m2)-1])
    desc_idx1 = cur_max['desc_names'].index(m1)
    desc_idx2 = cur_max['desc_names'].index(m2)
    ax.plot(cur_max['p_selection'][desc_idx2, :], cur_max['p_selection'][desc_idx1, :], 'x')
    ax.plot([0, 1], [0, 1], color='0.5')
    ax.axis('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    if mn.index(m1) == (mn.index(m2)-1):
        ax.set_ylabel(m1)
    if mn.index(m1) == 0:
        ax.set_title(m2)
    if not (mn.index(m1) == 0 and mn.index(m2) == 1):
        ax.set_yticks([])
    if not (mn.index(m1) == (len(mn)-2) and mn.index(m2) == (len(mn)-1)):
        ax.set_xticks([])
plt.savefig(os.path.join(outpath, 'descriptor_comparison_forest.png'))

# ML method comparison plot
colors = ['#95D4EC', '#C092DD', '#86E66C', '#F75454', '#FBCF39']
fig = plt.figure()
ax = fig.add_subplot(111)
desc1_collect, desc2_collect = [], []
for i, desc in enumerate(desc2comp):
    desc_idx1 = max_overview['svr']['linear']['desc_names'].index(desc)
    desc_idx2 = max_overview['forest']['forest']['desc_names'].index(desc)
    desc1_collect.extend(max_overview['svr']['linear']['p_selection'][desc_idx1, :])
    desc2_collect.extend(max_overview['forest']['forest']['p_selection'][desc_idx2, :])
    ax.plot(max_overview['svr']['linear']['p_selection'][desc_idx1, :],
            max_overview['forest']['forest']['p_selection'][desc_idx2, :],
            '.', color=colors[i], markersize=12,
            label=desc)
ax.plot([0, 0.8], [0, 0.8], color='0.5')
ax.plot([0, 0.8], [0, 0.], color='0.5')
ax.plot([0, 0.], [0, 1.], color='0.5')
ax.add_patch(plt.Rectangle((-1,-1),1,1, alpha=0.4))
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_xlabel('SVR')
ax.set_ylabel('forest')
utils.simple_axis(ax)
# ax.legend(loc='lower right')
fig.savefig(os.path.join(outpath, 'best_method_comparison.' + config['format']))

assert len(desc1_collect) == len(desc2_collect)
ratio = np.sum(np.array(desc1_collect) > np.array(desc2_collect)) / len(desc1_collect)
print('svr better than rfr in {:.2f} \% of the cases'.format(ratio))

if utils.run_from_ipython():
    plt.show()
