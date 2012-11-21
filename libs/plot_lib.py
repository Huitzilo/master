#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import numpy as np
import pylab as plt

def descriptor_performance_plot(fig, max_overview, search_res, sc):
    """compare performance of different descriptors for several glomeruli"""
    for i_meth, method in enumerate(max_overview):

        for i_sel, selection in enumerate(max_overview[method]):

            data = max_overview[method][selection]['max']
            sort_x = np.argsort(np.mean(data, axis=0))
            sort_y = np.argsort(np.mean(data, axis=1))
            data = data[sort_y, :]
            data = data[:, sort_x]

            plot_x = (i_sel * len(max_overview) + i_meth + 1) * 3 - 2
            ax1 = fig.add_subplot(6, 3, plot_x)
            ax1.imshow(data, interpolation='nearest')
            ax1.set_xticks(range(len(sc['glomeruli'])))
            ax1.set_xticklabels([sc['glomeruli'][i] for i in sort_x], rotation='45')
            ax1.set_aspect('auto')
            ax1.set_title('{}_{}.'.format(method, selection))

            ax = fig.add_subplot(6, 3, plot_x + 1, sharey=ax1)
            ax.barh(np.arange(len(search_res.keys())) - 0.5,
                   np.mean(data, axis=1),
                   xerr=np.std(data, axis=1), height=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax1.set_yticks(range(len(search_res.keys())))
            ax1.set_yticklabels([search_res.keys()[i] for i in sort_y])
            ax.set_xlabel('average descriptor score')
            plt.setp(ax.get_yticklabels(), visible=False)

            ax = fig.add_subplot(6, 3, plot_x + 2)
            bins = np.arange(0, 1, 0.05)
            ax.hist(data.flatten(), bins=bins, color='b')
            ax.set_xlim([0, 1])
            ax.set_xlabel('overall method score histogram')
            plt.hold(True)
            ax.hist(data[-3:,:].flatten(), bins=bins, color='r')
            ax.set_xlim([0, 1])
            ax.set_xlabel('overall method score histogram')
            fig.subplots_adjust(hspace=0.35, wspace=0.02)


def feature_selection_comparison_plot(fig, max_overview, sc):
    """plot comparison between linear and forest feature selection"""
    for i_meth, method in enumerate(max_overview):
        ax = fig.add_subplot(2, len(max_overview), i_meth + 1)
        flat_lin = max_overview[method]['linear']['max'].flatten()
        flat_for = max_overview[method]['forest']['max'].flatten()
        counts_lin, _ = np.histogram(flat_lin, bins=len(sc['k_best']))
        counts_for, _ = np.histogram(flat_for, bins=len(sc['k_best']))

        # scatter plot of max values
        ax.plot(flat_lin, flat_for, 'x')
        plt.axis('scaled')
        ax.set_xticks([0, ax.get_xticks()[-1]])
        ax.set_yticks([0, ax.get_yticks()[-1]])
        ax.set_title(method)
        ax.set_xlabel('linear')
        if i_meth == 0:
            ax.set_ylabel('forest')
        ax.plot([0, 1], [0, 1], '-', color='0.6')

        # k_best histogram plot
        ax = fig.add_subplot(2, len(max_overview), i_meth + 4)
        ax.bar(range(len(sc['k_best'])), counts_lin, color='r', label='linear')
        plt.hold(True)
        ax.bar(range(len(sc['k_best'])), -counts_for, color='b', label='forest')
        ax.set_xticks(np.arange(len(sc['k_best'])) + .5)
        ax.set_xticklabels(sc['k_best'], rotation='90', ha='left')
        fig.subplots_adjust(hspace=0.4)


def plot_search_matrix(fig, desc_res, sc, methods):
    """docstring for plot_search_matrix"""
    for i_sel, selection in enumerate(sc['selection']):
        for i_glom, glom in enumerate(desc_res[selection]):
            for i_meth, method in enumerate(methods):
                mat = desc_res[selection][glom][method]
                ax_idx = i_meth * len(sc['glomeruli']) * 2 + len(sc['glomeruli']) * i_sel + i_glom + 1
                ax = fig.add_subplot(6, len(sc['glomeruli']), ax_idx)
                ax.imshow(mat, interpolation='nearest')
                if i_sel + i_meth == 0:
                    ax.set_title(glom, rotation='45')
                if i_glom == 0:
                    ax.set_ylabel('{}\n{}'.format(method,selection))
                ax.set_yticks([])
                ax.set_xticks([])
    fig.subplots_adjust(hspace=0.4, wspace=0.3)