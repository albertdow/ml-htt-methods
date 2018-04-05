# Usage:
#     python train.py --sig_sample powheg --mode xgb_multi --channel tt

import random
import uproot
import ROOT
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from scipy import interp
from root_numpy import array2root
import json
from pandas.core.groupby import GroupBy
# import seaborn as sns
from mlglue.tree import tree_to_tmva, BDTxgboost, BDTsklearn
import xgboost2tmva

from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# custom modules
import plot_functions as pf
import load_functions as lf
import fit_functions as ff


parser = argparse.ArgumentParser()

parser.add_argument('--mode', action='store', default='sklearn_ttsplit',
    help='training procedure (default train_test_split)')
parser.add_argument('--channel', action='store', default='mt',
    help='channels to train on')
parser.add_argument('--sig_sample', action='store', default='powheg',
    help='''ggh signal sample to run on (default powheg)\n
    choose powheg for n_jets < 2 | (n_jets >= 2 & mjj < 300)\n
    choose JHU for n_jets >=2 & mjj > 300''')

opt = parser.parse_args()

train_data = pd.read_hdf('data/dataset_cpsm_{}_{}.hdf5'.format(opt.channel, opt.sig_sample))
print train_data.shape

if opt.mode == 'sklearn_ttsplit':

    ff.fit_ttsplit(train_data, opt.channel, opt.sig_sample)

if opt.mode == 'sklearn_sssplit':

    ff.fit_sssplit(train_data, 4, opt.channel, opt.sig_sample)

if opt.mode == 'gbc_ttsplit':

    ff.fit_gbc_ttsplit(train_data, opt.channel, opt.sig_sample)

if opt.mode == 'keras_multi':

    ff.fit_keras(train_data, opt.channel, opt.sig_sample)

if opt.mode == 'xgb_multi':

    ff.fit_multiclass_ttsplit(train_data, opt.channel, opt.sig_sample)

