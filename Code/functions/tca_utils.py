#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:10:47 2018

@author: Corentin Guérinot

Some useful functions for TCA
"""

import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, parafac
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import scipy as sci
import seaborn as sns
import statannot as sta


def raster_plot(tensor):
    """[summary]
    
    [description]
    
    Arguments:
        tensor {[type]} -- [description]
    """

    fig = plt.figure(figsize=(30, 15))
    rand = np.random.randint(0, tensor.shape[2], 5)

    for i in range(5):
        
        fig.add_subplot(1, 5, i+1)
        plt.imshow(tensor[:, :, rand[i]], cmap='coolwarm', aspect='auto')
        
    plt.colorbar()
    
    
def make_map(roi_tensor, neuron_factor):
    """[summary]
    
    [description]
    
    Arguments:
        roi_tensor {[type]} -- [description]
        neuron_factor {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    
    roi_map = np.zeros([512, 512])
    for n in range(roi_tensor.shape[2]):
        roi_map += neuron_factor[n] * roi_tensor[:, :, n]
        
    return roi_map


def normalize(v):
    """[summary]
    
    [description]
    
    Arguments:
        v {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


def rec_err(true_tensor, pred_tensor):
    """[summary]
    
    [description]
    
    Arguments:
        true_tensor {[type]} -- [description]
        pred_tensor {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    err = norm(true_tensor - pred_tensor) / norm(true_tensor)
    
    # baseline = true_tensor.mean(axis=2)
    # base_tensor = np.zeros_like(true_tensor)
    
    # for k in range(true_tensor.shape[2]):
    #     base_tensor[:, :, k] = baseline
        
    # err = np.linalg.norm(true_tensor - pred_tensor)**2/np.linalg.norm(true_tensor - base_tensor)**2
    
    return err


def norm(tensor, order=2, axis=None):
    """Computes the l-`order` norm of tensor

    Parameters
    ----------
    tensor : ndarray
    order : int
    axis : int or tuple

    Returns
    -------
    float or tensor
        If `axis` is provided returns a tensor.
    """
    # handle difference in default axis notation
    if axis == ():
        axis = None

    if order == 'inf':
        return np.max(np.abs(tensor), axis=axis)
    if order == 1:
        return np.sum(np.abs(tensor), axis=axis)
    elif order == 2:
        return np.sqrt(np.sum(tensor**2, axis=axis))
    else:
        return np.sum(np.abs(tensor)**order, axis=axis)**(1/order)


def norm_factors(factors):
    """[summary]
    
    [description]
    
    Arguments:
        factors {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    
    norm_list = []
    
    for factor in factors:
        
        rank = factor.shape[1]
        norm_factor = np.zeros_like(factor)
        
        for i in range(rank):
            norm_factor[:, i] = normalize(factor[:, i])
            
        norm_list.append(norm_factor)
    
    return norm_list


def norm_tensor(acti):
    """[summary]
    
    [description]
    
    Arguments:
        acti {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    
    N, T, _ = acti.shape
    norm_acti = np.zeros_like(acti)
    max_amp = np.nanmax(acti, axis=(0, 1))

    for t in range(T):
        for n in range(N):
            norm_acti[n, t, :] = np.divide(acti[n, t, :] + 1, max_amp + 1)

    return norm_acti


def day_limits(day, drop_trial):
    """[summary]
    
    [description]
    
    Arguments:
        day {[type]} -- [description]
        drop_trial {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    limit_day = []
    
    day_drop = np.delete(day, drop_trial)
    tic = 1
    
    for i, _ in enumerate(day_drop):
        
        if day_drop[i] > tic:
            limit_day.append(i)
            tic += 1
        
    return limit_day


def make_score(learning, beh):
    """[summary]
    
    [description]
    
    Arguments:
        learning {[type]} -- [description]
        beh {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    
    n_blocks = len(beh) // 20
    idx = 0
    learn_score = []

    for _ in range(n_blocks):

        score = 0
        trial = 0

        while trial < 20:

            if learning[idx] > 0:

                if beh[idx] == 1:
                    score += 5
                if beh[idx] == 3:
                    score += 5

            trial += 1
            idx += 1

        learn_score.append(score)
    
    return n_blocks, learn_score


def get_limits(n_blocks, drop_trial):
    """[summary]
    
    [description]
    
    Arguments:
        n_blocks {[type]} -- [description]
        drop_trial {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    limit_block = []
    offset_list = []
    right = 0

    for block in range(n_blocks):

        left = right
        right = left + 20
        local_offset = 0

        for trial in drop_trial:
            if trial >= block*20:
                if trial < (block + 1)*20:
                    local_offset += 1

        right += - local_offset

        offset_list.append(local_offset)
        limit_block.append([left, right])
        
    return limit_block


def convert_color(learn_score):
    """[summary]
    
    [description]
    
    Arguments:
        learn_score {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    learn_color = []

    for score in learn_score:
        if score == 0:
            learn_color.append('grey')
        elif (score > 0) & (score <= 35):
            learn_color.append('maroon')
        elif (score > 35) & (score <= 50):
            learn_color.append('darkred')
        elif score == 55:
            learn_color.append('red')
        elif score == 60:
            learn_color.append('orangered')
        elif score == 65:
            learn_color.append('darkorange')
        elif score == 70:
            learn_color.append('orange')
        elif score == 75:
            learn_color.append('goldenrod')
        elif score == 80:
            learn_color.append('gold')
        elif score == 85:
            learn_color.append('yellowgreen')
        elif score == 90:
            learn_color.append('lawngreen')
        elif score == 95:
            learn_color.append('lime')
        elif score == 100:
            learn_color.append('limegreen')
  
    return learn_color


def behaviorgram(beh_mat, drop_trial):
    """[summary]
    
    [description]
    
    Arguments:
        beh_mat {[type]} -- [description]
        drop_trial {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    
    beh = beh_mat[:, 1]
    learning = beh_mat[:, 2]
    day = beh_mat[:, 3]
    
    n_blocks, learn_score = make_score(learning, beh)
    
    limit_day = day_limits(day, drop_trial)
    limit_block = get_limits(n_blocks, drop_trial)
    learn_color = convert_color(learn_score)
    
    return learn_color, limit_block, limit_day


def give_order(factors):
    """[summary]
    
    [description]
    
    Arguments:
        factors {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    rank = factors[0].shape[1]
    onset = []

    for r in range(1, rank):
        comp = factors[1][:, r]
        thres = 2 * np.std(comp) + comp[0]
        t = 0
        while comp[t] < thres:
            if t < 284:
                t += 1
            else:
                break
        onset.append(t)

    order = []
    ti = np.copy(onset)
    onset.sort()

    for t in onset:
        i = 0
        while t != ti[i]:
            i += 1
        order.append(i)
    
    return order


def ord_fact(factors, order):
    """[summary]
    
    [description]
    
    Arguments:
        factors {[type]} -- [description]
        order {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    ord_factors = []

    for factor in factors:
        ord_factor = np.zeros_like(factor)
        ord_factor[:, 0] = factor[:, 0]
        for i, o in enumerate(order):
            ord_factor[:, i+1] = factor[:, o+1]
        ord_factors.append(ord_factor)
        
    return ord_factors


def reco(acti, factors, rank, nidx=None, kidx=None):
    """[summary]
    
    [description]
    
    Arguments:
        acti {[type]} -- [description]
        factors {[type]} -- [description]
        rank {[type]} -- [description]
    
    Keyword Arguments:
        nidx {[type]} -- [description] (default: {None})
        kidx {[type]} -- [description] (default: {None})
    """
    
    N, T, K = acti.shape
    norm_acti = norm_tensor(acti)
    
    pred_tensor = tl.kruskal_to_tensor(factors)
    factors = ord_fact(factors, give_order(factors))
    
    if nidx is None:
        _, nidx = max((factors[0][:, rank][i], i) for i in range(N))
    if kidx is None:
        _, kidx = max((factors[2][:, rank][i], i) for i in range(K))
     
    plt.plot(pred_tensor[nidx, :, kidx], color='orangered', linewidth=2, label='Model')
    plt.plot(norm_acti[nidx, :, kidx], color='blue', linewidth=1, label='True')
    plt.xlabel('Time (s)', {'fontsize': 'medium', 'fontweight' : 'bold'})
    plt.ylabel('Normalized df/f0', {'fontsize': 'medium', 'fontweight' : 'bold'})
    plt.locator_params(nbins=T//30, steps=[1, 3, 5, 10], min_n_ticks=T//30)
    plt.fill_betweenx([0, 1], 105, 135, facecolor='red', alpha=0.3, label='Odor Pres.')

    time_index = list(np.arange(0, T//15 + 1, 2))
    plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270], time_index)
    plt.legend(loc=1)
    
    r2 = round(r2_score(pred_tensor[nidx, :, kidx], norm_acti[nidx, :, kidx]), 3)
    plt.text(x=0, y=0.9, s='R2 = {0}'.format(r2))
    
    plt.show()
    
    
def make_box_df(factors, sel, b, color_df):
    """[summary]
    
    [description]
    
    Arguments:
        factors {[type]} -- [description]
        sel {[type]} -- [description]
        b {[type]} -- [description]
        color_df {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    
    lims = b[1]
    block_list = []
    for i, lim in enumerate(lims):
        a, b = lim
        for _ in range(a, b):
            block_list.append(i)

    order = give_order(factors)
    factors = ord_fact(factors, order)

    box_df = pd.DataFrame(index=color_df.index, columns=['Odor', 'Block', 'Trial Factor'])

    box_df['Odor'] = color_df['Odor']
    box_df.replace('red', 'S +', inplace=True)
    box_df.replace('black', 'S -', inplace=True)

    box_df['Reward'] = color_df['Reward']
    box_df.replace('purple', 'Reward', inplace=True)
    box_df.replace('grey', 'No Reward', inplace=True)

    box_df['Day'] = color_df['Day']
    for i, day in enumerate(color_df['Day'].unique()):
        box_df.replace(day, i, inplace=True)

    box_df['Behavior'] = color_df['Behavior']
    box_df.replace('black', 'Miss', inplace=True)
    box_df.replace('blue', 'CR', inplace=True)
    box_df.replace('yellow', 'FA', inplace=True)
    box_df.replace('red', 'Hit', inplace=True)

    box_df['Block'] = pd.Series(block_list)

    box_df['Trial Factor'] = pd.Series(factors[2][:, sel])
    
    return box_df


def box_zoom(factors, sel, hue, b, color_df, palette=None, stat=False):
    """[summary]
    
    [description]
    
    Arguments:
        factors {[type]} -- [description]
        sel {[type]} -- [description]
        hue {[type]} -- [description]
        b {[type]} -- [description]
        color_df {[type]} -- [description]
    
    Keyword Arguments:
        palette {list} -- [description] (default: {['red', 'black']})
        stat {bool} -- [description] (default: {False})
    """

    box_df = make_box_df(factors, sel, b, color_df)
    n_blocks = len(b[1])
    if palette is None:
        palette = ['red', 'black']
    
    plt.rcParams['figure.figsize'] = 14, 6
    #fig = plt.figure(figsize=(14, 6))

    ax = sns.boxplot(x="Block", y="Trial Factor", hue=hue, data=box_df,
                     palette=palette, dodge=False,
                     linewidth=2, fliersize=2, width=.3)
    xmin, xmax, ymin, ymax = ax.axis()

    if stat:
        c1, c2 = box_df[hue].unique()[:2]
        for i in range(n_blocks):
            if len(box_df[hue][box_df['Block'] == i].unique()) < 2:
                continue
            sta.add_stat_annotation(ax, data=box_df, x="Block", y="Trial Factor", hue=hue,
                                    boxPairList=[((i, c1), (i, c2))], test='t-test',
                                    textFormat='star', loc='inside', fontsize='large',
                                    lineYOffsetAxesCoord=0.05, linewidth=0, verbose=0)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax*1.15)

    plt.show()


def seq_parafac(input_tensor, max_rank, nb_trial, pred_df, tol=1e-06, mode='non-negative'):
    """[summary]
    
    [description]
    
    Arguments:
        input_tensor {[type]} -- [description]
        max_rank {[type]} -- [description]
        nb_trial {[type]} -- [description]
        pred_df {[type]} -- [description]
    
    Keyword Arguments:
        mode {str} -- [description] (default: {'non-negative'})
    
    Returns:
        [type] -- [description]
    """

    rank_err = []
    rank_sim = []
    err_list = []
    sim_list = []
    spa_list = []
    
    odor_acc = []
    odor_std = []
    rew_acc = []
    rew_std = []
    
    base_tensor = np.zeros_like(input_tensor)
    for k in range(input_tensor.shape[2]):
        base_tensor[:, :, k] = input_tensor.mean(axis=2)

    for rank in np.arange(1, max_rank+1):

        pred_fac = []
        min_err = 1
        min_idx = 0

        for trial in range(nb_trial):

            print('Trial', trial)
            rank_err.append(rank)
        
            if mode == 'non-negative':
                pred_fac.append(non_negative_parafac(input_tensor, rank=rank, n_iter_max=5000,
                                                     init='random', tol=tol))
            else:
                pred_fac.append(parafac(input_tensor, rank=rank, n_iter_max=5000,
                                        init='random', tol=tol))
            pred_tensor = tl.kruskal_to_tensor(pred_fac[trial])
            err = rec_err(input_tensor, pred_tensor)
            err_list.append(err)
            
            nb_nonzero = 0
            tot_size = 0
            
            for i in range(len(pred_fac[trial])):
                nb_nonzero += np.count_nonzero(np.round(pred_fac[trial][i], 2))
                tot_size += pred_fac[trial][i].size
            
            spa = 1 - nb_nonzero / tot_size
            spa_list.append(spa)
            
            X, y_odor, y_rew = shuffle(pred_fac[trial][2], pred_df['Odor'].tolist(),
                                       pred_df['Reward'].tolist())

            clf = RandomForestClassifier(n_estimators=50, max_depth=None,
                                         min_samples_split=2, max_features='sqrt')

            X = StandardScaler().fit_transform(X)
            
            odor_acc.append(cross_val_score(clf, X, y_odor, cv=8).mean())
            odor_std.append(cross_val_score(clf, X, y_odor, cv=8).std())
            rew_acc.append(cross_val_score(clf, X, y_rew, cv=8).mean())
            rew_std.append(cross_val_score(clf, X, y_rew, cv=8).std())

            if err < min_err:
                min_err = err
                min_idx = trial

        for trial in range(nb_trial):
            
            if trial == min_idx:
                continue
                
            rank_sim.append(rank)

            sim_list.append(tt.kruskal_align(tt.tensors.KTensor(pred_fac[min_idx]), 
                                             tt.tensors.KTensor(pred_fac[trial]), 
                                             permute_U=True, permute_V=True))
            
    
    err_df = pd.DataFrame(data=np.transpose([rank_err, err_list]), 
                          columns=['Rank', 'Reconstruction Error'])
    sim_df = pd.DataFrame(data=np.transpose([rank_sim, sim_list]), 
                          columns=['Rank', 'Similarity'])
    spa_df = pd.DataFrame(data=np.transpose([rank_err, spa_list]), 
                          columns=['Rank', 'Sparsity'])
    odor_df = pd.DataFrame(data=np.transpose([rank_err, odor_acc, odor_std]),
                           columns=['Rank', 'Accuracy - Odor Prediction', 'Std - Odor Prediction'])
    rew_df = pd.DataFrame(data=np.transpose([rank_err, rew_acc, rew_std]),
                          columns=['Rank', 'Accuracy - Reward Prediction',
                                   'Std - Reward Prediction'])
    
    return err_df, sim_df, spa_df, odor_df, rew_df


def factorplot(factors, roi_tensor, b=None, balance=True, color='k', shaded=None, order=False, path=None):
    """[summary]
    
    [description]
    
    Arguments:
        factors {[type]} -- [description]
        roi_tensor {[type]} -- [description]
    
    Keyword Arguments:
        b {list} -- [description] (default: {[0]})
        balance {bool} -- [description] (default: {True})
        color {str} -- [description] (default: {'k'})
        shaded {list} -- [description] (default: {[7,9]})
        order {bool} -- [description] (default: {False})
    """
    
    # factors is the list of 3 factor matrices - Kruskal form of tensor
    
    if balance:
        # Compute norms along columns for each factor matrix
        norms = [sci.linalg.norm(f, axis=0) for f in factors]

        # Multiply norms across all modes
        lam = sci.multiply.reduce(norms) ** (1/3)

        # Update factors
        factors = [f * (lam / fn) for f, fn in zip(factors, norms)]
    
    if order:
        factors = ord_fact(factors, give_order(factors))

    # rank is the number of components of TCA - as well as the number of columns in factor matrices
    rank = factors[0].shape[1]
    T = factors[1].shape[0]
    K = factors[2].shape[0]

    #behaviorgram
    if b is not None:
        learn_color, limit_block, limit_day = b
        top = np.max(factors[2])

    if shaded is None:
        shaded = [7, 9]
    
    # initiate the plotting object
    fig, axarr = plt.subplots(rank, 3, sharex='col', figsize=(15, rank*3))

    # for each of the component
    for r in range(rank):

        # plot neuron factors on the ROI map
        roi_map = make_map(roi_tensor, factors[0][:, r])
        axarr[r, 0].imshow(roi_map, vmin=0, vmax=np.max(factors[0]), cmap='hot')
        # plot time factors as a lineplot
        axarr[r, 1].plot(np.arange(1, T+1), factors[1][:, r], color='k', linewidth=2)
        axarr[r, 1].locator_params(nbins=T//30, steps=[1, 3, 5, 10], min_n_ticks=T//30)
        axarr[r, 1].fill_betweenx([0, np.max(factors[1])+.01], 15*shaded[0],
                                  15*shaded[1], facecolor='red', alpha=0.5)
        # plot trial factors as a scatter plot
        axarr[r, 2].scatter(np.arange(1, K+1), factors[2][:, r], c=color)
        axarr[r, 2].locator_params(nbins=K//20, steps=[1, 2, 5, 10], min_n_ticks=K//20)
        
        if b is not None:
            for i, block in enumerate(limit_block):
                axarr[r, 2].fill_betweenx([1.05 * top, 1.25 * top], block[0], block[1], 
                                          facecolor=learn_color[i], alpha=1)
            
            for limit in limit_day:
                axarr[r, 2].axvline(limit, 0.75, 1, linewidth=2, color='black')
        
        for i in [1, 2]:

            # format axes
            axarr[r, i].spines['top'].set_visible(False)
            axarr[r, i].spines['right'].set_visible(False)

            # remove xticks on all but bottom row
            if r != rank-1:
                plt.setp(axarr[r, i].get_xticklabels(), visible=False)
        
        axarr[r, 0].tick_params(axis='both', which='both', bottom=False, top=False,
                                labelbottom=False, right=False, left=False, labelleft=False)

    # set titles for top row and legend for bottom row
    axarr[0, 0].set_title('Neuron Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
    axarr[0, 1].set_title('Temporal Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
    axarr[0, 2].set_title('Trial Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
    
    axarr[rank-1, 0].set_xlabel('ROI map', {'fontsize': 'large', 'fontweight' : 'bold',
                                            'verticalalignment' : 'top'})

    time_index = list(np.arange(0, T//15 + 1, 2))
    time_index.insert(0, 1)
    axarr[rank-1, 1].set_xlabel('Time (s)', {'fontsize': 'large', 'fontweight' : 'bold'})
    axarr[rank-1, 1].set_xticklabels(time_index)
    
    trial_index = list(np.arange(0, K//20 + 2))
    trial_index.insert(0, 1)
    axarr[rank-1, 2].set_xlabel('Block', {'fontsize': 'large', 'fontweight' : 'bold'})
    axarr[rank-1, 2].set_xticklabels(trial_index)
    
    # link y-axes within columns
    for i in range(3):
        yl = [a.get_ylim() for a in axarr[:, i]]
        y0, y1 = min([y[0] for y in yl]), max([y[1] for y in yl])
        _ = [a.set_ylim((y0, y1)) for a in axarr[:, i]]

    # format y-ticks
    for r in range(rank):
        for i in range(3):
            # only two labels
            axarr[r, i].set_ylim(np.round(axarr[r, i].get_ylim(), 2))
            axarr[r, i].set_yticks([0, np.round(axarr[r, i].get_ylim(), 2)[1]])

    # make so that plots are tightly presented
    plt.tight_layout()

    if path is not None:
        plt.savefig(path)

    plt.show(fig)
    

def factorplot_singlecomp(factors, roi_tensor, b=None, balance=True, color='k', shaded=None):
    """[summary]
    
    [description]
    
    Arguments:
        factors {[type]} -- [description]
        roi_tensor {[type]} -- [description]
    
    Keyword Arguments:
        b {[type]} -- [description] (default: {None})
        balance {bool} -- [description] (default: {True})
        color {str} -- [description] (default: {'k'})
        shaded {[type]} -- [description] (default: {None})
        order {bool} -- [description] (default: {False})
    """
    
    # factors is the list of 3 factor matrices - Kruskal form of tensor
    
    if balance:
        # Compute norms along columns for each factor matrix
        norms = [sci.linalg.norm(f, axis=0) for f in factors]

        # Multiply norms across all modes
        lam = sci.multiply.reduce(norms) ** (1/3)

        # Update factors
        factors = [f * (lam / fn) for f, fn in zip(factors, norms)]

    T = factors[1].shape[0]
    K = factors[2].shape[0]

    #behaviorgram
    if b is not None:
        learn_color, limit_block, limit_day = b
        top = np.max(factors[2])

    if shaded is None:
        shaded = [7, 9]
    
    # initiate the plotting object
    _, axarr = plt.subplots(1, 3, sharex='col', figsize=(15, 3))

    # plot neuron factors on the ROI map
    roi_map = make_map(roi_tensor, factors[0])
    axarr[0].imshow(roi_map, vmin=0, vmax=np.max(factors[0]), cmap='hot')
    
    # plot time factors as a lineplot
    axarr[1].plot(np.arange(1, T+1), factors[1], color='k', linewidth=2)
    axarr[1].locator_params(nbins=T//30, steps=[1, 3, 5, 10], min_n_ticks=T//30)
    axarr[1].fill_betweenx([0, np.max(factors[1])+.01], 15*shaded[0],
                           15*shaded[1], facecolor='red', alpha=0.5)
    # plot trial factors as a scatter plot
    axarr[2].scatter(np.arange(1, K+1), factors[2], c=color)
    axarr[2].locator_params(nbins=K//20, steps=[1, 2, 5, 10], min_n_ticks=K//20)
    
    if b is not None:
        for i, block in enumerate(limit_block):
            axarr[2].fill_betweenx([1.05 * top, 1.25 * top], block[0], block[1], 
                                   facecolor=learn_color[i], alpha=1)
        
        for limit in limit_day:
            axarr[2].axvline(limit, 0.75, 1, linewidth=2, color='black')
    
    for i in [1, 2]:

        # format axes
        axarr[i].spines['top'].set_visible(False)
        axarr[i].spines['right'].set_visible(False)
    
    axarr[0].tick_params(axis='both', which='both', bottom=False, top=False,
                         labelbottom=False, right=False, left=False, labelleft=False)

    # set titles for top row and legend for bottom row
    axarr[0].set_title('Neuron Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
    axarr[1].set_title('Temporal Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
    axarr[2].set_title('Trial Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
    
    axarr[0].set_xlabel('ROI map',
                        {'fontsize': 'large', 'fontweight' : 'bold', 'verticalalignment' : 'top'})

    time_index = list(np.arange(0, T//15 + 1, 2))
    time_index.insert(0, 1)
    axarr[1].set_xlabel('Time (s)', {'fontsize': 'large', 'fontweight' : 'bold'})
    axarr[1].set_xticklabels(time_index)
    
    trial_index = list(np.arange(0, K//20 + 2))
    trial_index.insert(0, 1)
    axarr[2].set_xlabel('Block', {'fontsize': 'large', 'fontweight' : 'bold'})
    axarr[2].set_xticklabels(trial_index)
    
    _, ymax1 = np.round(axarr[1].get_ylim(), 2)
    axarr[1].set_ylim((0, ymax1))
    axarr[1].set_yticks([0, ymax1])
    _, ymax2 = np.round(axarr[2].get_ylim(), 2)
    axarr[2].set_ylim((0, ymax2))
    axarr[2].set_yticks([0, ymax2])

    # make so that plots are tightly presented
    plt.tight_layout()
    plt.show()
    