#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pylab as plt
import csv
import pandas as pd
import CoolProp.CoolProp as CP
import math

from analysis_dw import trial_dw
from analysis_up import trial_up

from matplotlib import rc

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


import matplotlib.pyplot as plt
import time
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times New Roman']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# rc('figure', figsize=[12,4])
# rc('figure', figsize=[12,5])
rc('font', size=30)

#################################################################################
@contextmanager
def plotcontext(L=88, R=6.28, step=0.1, savepath=None):
    def CircleXToY(x, R):
        # 完整的1/4 圆的个数
        QuarterCircle_Num = int(x / R)
        # 最后一个不完整的1/4圆x方向大小;
        xr = (x - QuarterCircle_Num * R)
        # 计算y方向的大小
        if QuarterCircle_Num % 4 in [1, 3]:
            xr = R - xr
        yr = math.sqrt(R ** 2 - xr ** 2)
        # 计算y方向的符号
        if QuarterCircle_Num % 4 in [0, 3]:
            yr = - yr
        return yr

    x = np.arange(0.0, L, step)

    df = pd.DataFrame({'x': x})
    # 对于每个x求出在蛇形管的对应的y
    df['y'] = df['x'].apply(lambda e: CircleXToY(e, R))

    CenterX = np.arange(0, L-0.3*R, 2 * R)
    oddeven = np.array(range(len(CenterX)))
    Circle = pd.DataFrame({'centerx': CenterX,
                           'centery': np.zeros(len(CenterX)),
                           'oddeven': oddeven,
                           'dx': np.ones(len(CenterX),dtype=float)*0.3*R})
    Circle['dy'] = Circle.apply(lambda row: math.sqrt(R**2-row['dx']**2) if row['oddeven']%2==1 else -math.sqrt(R**2-row['dx']**2), axis=1)

    plt.figure(figsize=(12, 9))
    grid = plt.GridSpec(12, 5, wspace=1.0, hspace=0.0)

    # 下图
    main_ax = plt.subplot(grid[4:12, 0:5])

    yield

    plt.xticks(np.arange(0, L, 0.01))
    plt.setp(main_ax.get_xticklabels(), fontsize=20)
    #plt.xlabel('x')
    plt.grid(True, axis='x', linestyle="-.", color="black", linewidth="2")

    ax1 = plt.subplot(grid[0:4, 0:5], sharex=main_ax)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.plot(df['x'], df['y'], lw=5)
    for idx,e in Circle.iterrows():
        plt.arrow(e['centerx'], e['centery'], e['dx'], e['dy'], length_includes_head=True, head_width=0.05,
                  head_length=0.05, fc='r', ec='b')
    plt.text(0.0, 0.0, 'R', fontsize=20)
    plt.scatter(CenterX, np.zeros(len(CenterX)), s=20, color='black')

    # 纵轴grid线,对应过0点
    plt.xticks(np.arange(R, L, 2 * R))
    plt.grid(True, axis='x', linestyle="-.", color="black", linewidth="2")
    #plt.ylabel('y')
    if savepath!=None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

with plotcontext(savepath='test.pdf'):

#################################################################################
    d1 = trial_up('./data/7.7MPa-0.68kgh-UP-22-32A/7.7MPa-0.69kgh-22.8A-0.88V-26.3du-UP-2PM-.csv',
                  './data/7.7MPa-0.68kgh-UP-22-32A/7.7MPa-0.69kgh-22.8A-0.88V-25.3du-UP-2T.csv', I=22.8)
    d2 = trial_up('./data/7.7MPa-0.68kgh-UP-22-32A/7.7MPa-0.69kgh-24.8A-0.97V-25.3du-UP-2pm.csv',
                  './data/7.7MPa-0.68kgh-UP-22-32A/7.7MPa-0.69kgh-24.8A-0.88V-25.3du-UP-2T.csv', I=24.8)
    d3 = trial_up('./data/7.7MPa-0.68kgh-UP-22-32A/7.7MPa-0.69kgh-27.5A-1.07V-25.3du-UP-2-PM-0.5du-T4min.csv',
                  './data/7.7MPa-0.68kgh-UP-22-32A/7.7MPa-0.69kgh-27.5A-1.07V-25.3du-UP-2T.csv', I=27.5)
    d4 = trial_up('./data/7.7MPa-0.68kgh-UP-22-32A/7.7MPa-0.69kgh-29.6A-1.16V-25.3du-UP-2PM-1du-T5min.csv',
                  './data/7.7MPa-0.68kgh-UP-22-32A/7.7MPa-0.69kgh-29.6A-1.16V-25.3du-UP-2T.csv', I=29.6)

    pup1 = plt.plot(d1.x_d, d1.Tw_i, marker='s', markerfacecolor='black', color='black', markeredgecolor='black', linestyle='solid', label='DW')
    pup1f = plt.plot(d1.x_d, d1.T_i-273.15, color='black', linestyle=':', linewidth=1.5, label='')

    pup2 = plt.plot(d2.x_d, d2.Tw_i, marker='o', markerfacecolor='red', color='black', markeredgecolor='red', linestyle='solid', label='DW')
    pup2f = plt.plot(d1.x_d, d2.T_i-273.15, color='red', linestyle=':', linewidth=1.5, label='')

    pup3 = plt.plot(d3.x_d, d3.Tw_i, marker='v', markerfacecolor='blue', color='black', markeredgecolor='blue', linestyle='solid', label='DW')
    pup3f = plt.plot(d3.x_d, d3.T_i-273.15, color='blue', linestyle=':', linewidth=1.5, label='')

    pup4 = plt.plot(d4.x_d, d4.Tw_i, marker='^', markerfacecolor='green', color='black', markeredgecolor='green', linestyle='solid', label='DW')
    pup4f = plt.plot(d4.x_d, d4.T_i-273.15, color='green', linestyle=':', linewidth=1.5, label='')

    ptpc = plt.plot(d4.x_d, d4.T_pseudo_x, color='black', linestyle="-.", linewidth=1.5, label='T$_{pc}$')
    plt.xticks(fontsize=30)
    plt.xlabel('$x/d$')
    plt.ylabel('$T_{w,i}, T_{f}/^{\circ}C$')
    plt.xlim(0,90)
    plt.ylim(20,130)
    plt.legend([pup1[0], pup2[0], pup3[0], pup4[0], pup1f[0], ptpc[0]],
               ("59.2 kW/m$^2$ - UP", "70.5 kW/m$^2$ - UP", "87.6 kW/m$^2$ - UP", "102.6 kW/m$^2$ - UP", "T$_f$ -UP", "T$_{pc}$", "DW","DW", "DW", "DW", "T$_f$ -DW",),
               loc='best', ncol=2, prop={'size': 6})

# plt.legend(loc='upper center', bbox_to_anchor=(0.5,0.95),ncol=5,fancybox=True,shadow=True,prop={'size': 10})
