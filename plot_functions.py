import pandas.core.common as com
from pandas.core.index import Index
from pandas.tools import plotting
from pandas.plotting import scatter_matrix

import itertools
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import confusion_matrix


def plot_signal_background(data1, data2, column=None, grid=True,
                      xlabelsize=None, xrot=None, ylabelsize=None,
                      yrot=None, ax=None, sharex=False,
                      sharey=False, figsize=None,
                      layout=None, bins=10, **kwds):


    if 'alpha' not in kwds:
        kwds['alpha'] = 0.5

    if column is not None:
        if not isinstance(column, (list, np.ndarray, Index)):
            column = [column]
        data1 = data1[column]
        data2 = data2[column]

    data1 = data1._get_numeric_data()
    data2 = data2._get_numeric_data()
    naxes = len(data1.columns)

    fig, axes = plotting._subplots(naxes=naxes, ax=ax, squeeze=False,
                                   sharex=sharex,
                                   sharey=sharey,
                                   figsize=figsize,
                                   layout=layout)
    _axes = plotting._flatten(axes)

    for i, col in enumerate(com._try_sort(data1.columns)):
        ax = _axes[i]
        low = min(data1[col].min(), data2[col].min())
        high = max(data1[col].max(), data2[col].max())
        ax.hist(data1[col].dropna().values,
                bins=bins, range=(low,high), **kwds)
        ax.hist(data2[col].dropna().values,
                bins=bins, range=(low,high), **kwds)
        ax.set_title(col)
        ax.grid(grid)

    plotting._set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot,
                              ylabelsize=ylabelsize, yrot=yrot)
    fig.subplots_adjust(wspace=0.3, hspace=0.7)

    return axes


def plot_roc_curve(fpr, tpr, auc, figname):

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.grid()
    ax.text(0.6, 0.3, 'ROC AUC Score: {0:.4f}'.format(auc),
            bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))

    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    fig.savefig(figname)
    print 'ROC curve saved as {}'.format(figname)

    return None


def plot_scatter_matrix(X, figname):
    ## THIS FUNCTION CURRENTLY DOESNT DO WHAT IT SHOULD
    ##

    # need to resample DataFrame
    df = X.ix[random.sample(X.index, 100)]
    # df =

    plt.figure()
    sm = scatter_matrix(df, figsize=(20,20), alpha=0.4, s=60, c=['y','r'])
    plt.savefig(figname)
    print 'Scatter matrix saved as {}'.format(figname)

    return None


def plot_confusion_matrix(y_test, y_pred, classes, figname, normalise=False, cmap=plt.cm.Blues):

    cm = confusion_matrix(y_test, y_pred)
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print 'Normalised confusion matrix'
    else:
        print 'Non-normalised confusion matrix'

    print cm

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='w' if cm[i, j] > thresh else 'k')

    plt.tight_layout(pad=1.4)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(figname)
    print 'Confusion matrix saved as {}'.format(figname)

    return None

def plot_features(booster, imp_type, figname):

    fig = plt.figure(figsize=(12,7))
    axes = fig.add_subplot(111)

    if imp_type == 'weight':
        xgb.plot_importance(booster, ax=axes, height=0.2, xlim=None, ylim=None,
                title='', xlabel='F score', ylabel='Features', importance_type='weight')
    elif imp_type == 'gain':
        xgb.plot_importance(booster, ax=axes, height=0.2, xlim=None, ylim=None,
                title='', xlabel='F score', ylabel='Features', importance_type='gain')

    fig.savefig(figname)
    print 'Feature importance saved as {}'.format(figname)

    return None

def plot_correlation_matrix(data, figname, **kwds):

    corrmat = data.corr(**kwds)

    fig, ax = plt.subplots(ncols=1, figsize=(6,5))

    opts = {'cmap': plt.get_cmap("RdBu"),
            'vmin': -1, 'vmax': +1}
    heatmap1 = ax.pcolor(corrmat, **opts)
    plt.colorbar(heatmap1, ax=ax)

    ax.set_title("")

    labels = corrmat.columns.values
    for ax in (ax,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
        ax.set_yticklabels(labels, minor=False)

    plt.tight_layout()
    fig.savefig(figname)
    print 'Correlation matrix saved as {}'.format(figname)

