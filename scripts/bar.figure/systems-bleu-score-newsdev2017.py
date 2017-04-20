#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

rcParams['grid.linestyle'] = ':'


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/4, height + 0.5, '%s' % float(height), rotation=90, fontsize=10)


def drawBarChartPoseRatio():
    n_groups = 1

    '''
                                                 bleu-4 / bleu-sbp-5
                                     newsdev2017       cwmt2011       cwmt2009
    2-2 NiuSMT.biglm               : 14.20	17.59	30.90	31.29	31.50	32.61
    3 NMT-UM-baseline              : 17.40	20.44	37.71	36.85	39.75	38.88
    4 NMT-UM-bpe                   : 17.60	20.91	37.30	36.66	39.90	39.43
    4-1 NMT-UM-bpe-finetune        : 17.70	20.86	37.80	37.18	40.00	39.53
    5 NMT-UM-bpe-finetune-synthetic: 17.89	21.44	36.21	36.26	38.46	38.78
    5-1	ensemble-4 &4-1&5          : 18.41	21.81	38.65	38.03	40.76	41.89
    5-2	avg-4&4-1&5                : 18.24	21.61	38.28	37.67	40.88	40.39
    9 NMT-NEU-bpe (nematus)        : 16.73	20.73	34.06	36.51	40.27	39.94
    '''
    
    smt_um_biglm = (17.59)
    nmt_um_baseline = (20.44)
    nmt_um_bpe = (20.91)
    nmt_um_bpe_finetune = (20.86)
    nmt_um_bpe_finetune_syn = (21.44)
    nmt_um_ensemble = (21.81)
    nematus = (20.73)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    #opacity = 0.4
    opacity = 1

    #plt.grid()
    yminorLocator = MultipleLocator(1)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.yaxis.grid(True, which='minor')


    rects1 = plt.bar(index, nmt_um_baseline, bar_width / 2, alpha=opacity, color='m', label='$li-baseline$')
    rects2 = plt.bar(index + bar_width / 2, nmt_um_bpe, bar_width / 2, alpha=opacity, color='g', label='$li-bpe$')
    rects3 = plt.bar(index + bar_width, nmt_um_bpe_finetune, bar_width / 2, alpha=opacity, color='c', label='$li-bpe-fine$')
    rects4 = plt.bar(index + 1.5 * bar_width, nmt_um_bpe_finetune_syn, bar_width / 2, alpha=opacity, color='b', label='$li-bpe-fine-syn$')
    rects5 = plt.bar(index + 2 * bar_width, nmt_um_ensemble, bar_width / 2, alpha=opacity, color='r', label='$li-ensemble$')
    rects6 = plt.bar(index + 2.5 * bar_width, nematus, bar_width / 2, alpha=0.4, color='r', label='$ben-nematus$')
    rects7 = plt.bar(index + 3 * bar_width, smt_um_biglm, bar_width / 2, alpha=0.2, color='r', label='$li-smt-biglm$')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)
    autolabel(rects7)


    #plt.xlabel('Category', fontsize=16)
    plt.ylabel('BLEU-SBP-5', fontsize=16)
    #plt.title('Scores by group and Category')

    # plt.xticks(index - 0.2+ 2*bar_width, ('balde', 'bunny', 'dragon', 'happy', 'pillow'))
    plt.xticks(index - 0.2 + 2.5 * bar_width, ('newsdev2017',), fontsize = 16)

    plt.yticks(fontsize=14)  # change the num axis size

    plt.ylim(17, 23)  # The ceil
    plt.legend()
    #plt.tight_layout()
    plt.show()


drawBarChartPoseRatio()
