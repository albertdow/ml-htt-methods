# Usage:
#     python make_dataset.py -c --sig_sample powheg --mode xgb_multi --channel tt

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

parser.add_argument('-c', action='store_true', default=False,
        dest='apply_selection', help='apply the preselection (default False)')
parser.add_argument('--mode', action='store', default='sklearn_ttsplit',
        help='training procedure (default train_test_split)')
parser.add_argument('--channel', action='store', default='mt',
        help='channels to train on')
parser.add_argument('--sig_sample', action='store', default='powheg',
        help='''ggh signal sample to run on (default powheg)\n
        choose powheg for n_jets < 2 | (n_jets >= 2 & mjj < 300)\n
        choose JHU for n_jets >=2 & mjj > 300''')

opt = parser.parse_args()

## To create dataset for chosen channel & sig_sample (i.e. boosted or dijet cat)

print '\nCreate dataeset for {} channel with {} sig samples\n'.format(opt.channel, opt.sig_sample)
if opt.apply_selection:
    sig_files = lf.load_files('./filelist/sig_{}_files.dat'.format(opt.sig_sample))
    bkg_files = lf.load_files('./filelist/bkgs_files.dat')
else:
    opt.sig_sample = 'JHU'
    sig_files = lf.load_files('./filelist/sig_{}_files.dat'.format(opt.sig_sample))
    bkg_files = lf.load_files('./filelist/full_mc_files.dat')
data_files = lf.load_files('./filelist/{}_data_files.dat'.format(opt.channel))

# this file contains information about the xsections, lumi and event numbers
params_file = json.load(open('Params_2016_smsummer16.json'))
lumi = params_file['MuonEG']['lumi']

# cut_features will only be used for preselection
# and then dropped again
if opt.channel == 'tt':
    cut_features = [
            'mva_olddm_medium_1', 'mva_olddm_medium_2',
            'mva_olddm_loose_1', 'mva_olddm_loose_2',
            'antiele_1', 'antimu_1', 'antiele_2', 'antimu_2',
            'leptonveto', 'trg_doubletau',
            'mjj',
            ]

elif opt.channel == 'mt':
    cut_features = [
            'iso_1',
            'mva_olddm_medium_2',
            'antiele_2', 'antimu_2',
            'leptonveto',
            'trg_singlemuon', 'trg_mutaucross',
            'os',
            'mjj'
            ]

elif opt.channel == 'et':
    cut_features = [
            'iso_1',
            'mva_olddm_medium_2',
            'antiele_2', 'antimu_2',
            'leptonveto',
            'trg_singleelectron',
            'os',
            'mjj'
            ]

elif opt.channel == 'em':
    cut_features = [
            'iso_1',
            'iso_2',
            'leptonveto',
            'trg_muonelectron',
            'os',
            'mjj'
            ]

# features to train on
# apart from 'wt' - this is used for weights
# still need to multipy 'wt' by the scaling factor
# coming from the xsection

if opt.mode in ['keras_multi', 'xgb_multi']:

    features = [
        'pt_1', 'pt_2', 'eta_1', 'eta_2', 'dphi',
        'mt_1', 'mt_2', 'mt_lep',
        'm_vis', 'm_sv', 'pt_tt', 'eta_tt',
        'met', 'met_dphi_1', 'met_dphi_2',
        'n_jets', 'n_bjets',
        'pt_vis',
        'phi_1', 'phi_2',
        'wt', # for training/validation weights
        'gen_match_1', 'gen_match_2', # for splitting DY into separate processes

        # add more features similar to KIT take tt for now
        # 'mjj', 'jdeta',
        # 'jpt_1', 'jeta_1', 'jphi_1',
        # 'jphi_2',
        # 'jdphi',
        ]
else:
    features = [
            'pt_1', 'pt_2', 'eta_1', 'eta_2', 'dphi',
            'mt_1', 'mt_2', 'mt_lep',
            'm_vis', 'm_sv', 'pt_tt', 'eta_tt',
            'met', 'met_dphi_1', 'met_dphi_2',
            'n_jets', 'n_bjets',
            'pt_vis',
            'wt',
            ]
if opt.channel == 'em':
    features.append('pzeta')

class_dict = {
    'ggh': ['GluGluToHToTauTau_M-125',
        'GluGluH2JetsToTauTau_M125_CPmixing_maxmix',
        'GluGluH2JetsToTauTau_M125_CPmixing_sm',
        'GluGluH2JetsToTauTau_M125_CPmixing_pseudoscalar'],
    'qqh': ['VBFHToTauTau_M-125',
        'VBFHToWWTo2L2Nu_M-125',
        'VBFHiggs0Mf05ph0_M-125',
        'VBFHiggs0M_M-125',
        'VBFHiggs0PM_M-125'],
    'dy': ['DYJetsToLL_M-10-50-LO',
        'DY1JetsToLL-LO',
        'DY2JetsToLL-LO',
        'DY3JetsToLL-LO',
        'DY4JetsToLL-LO',
        'DYJetsToLL-LO-ext1',
        'DYJetsToLL-LO-ext2'],
    'w': [ 'W1JetsToLNu-LO',
        'W2JetsToLNu-LO-ext',
        'W2JetsToLNu-LO',
        'W3JetsToLNu-LO-ext',
        'W3JetsToLNu-LO',
        'W4JetsToLNu-LO-ext1',
        'W4JetsToLNu-LO-ext2',
        'W4JetsToLNu-LO',
        'WGToLNuG-ext',
        'WGToLNuG',
        'WGstarToLNuEE',
        'WGstarToLNuMuMu',
        'WJetsToLNu-LO-ext',
        'WJetsToLNu-LO',
        'WminusHToTauTau_M-125',
        'WplusHToTauTau_M-125'],
    'tt': ['TT'],
    'qcd': ['TauB','TauC',
        'TauD','TauE',
        'TauF','TauG',
        'TauHv2','TauHv3',
        'SingleMuonB','SingleMuonC',
        'SingleMuonD','SingleMuonE',
        'SingleMuonF','SingleMuonG',
        'SingleMuonHv2','SingleMuonHv3',
        'SingleElectronB','SingleElectronC',
        'SingleElectronD','SingleElectronE',
        'SingleElectronF','SingleElectronG',
        'SingleElectronHv2','SingleElectronHv3',
        'MuonEGB','MuonEGC',
        'MuonEGD','MuonEGE',
        'MuonEGF','MuonEGG',
        'MuonEGHv2','MuonEGHv3'],
    'misc': ['EWKWMinus2Jets_WToLNu-ext1','EWKWMinus2Jets_WToLNu-ext2',
        'EWKWMinus2Jets_WToLNu','EWKWPlus2Jets_WToLNu-ext1',
        'EWKWPlus2Jets_WToLNu-ext2','EWKWPlus2Jets_WToLNu',
        'EWKZ2Jets_ZToLL-ext','EWKZ2Jets_ZToLL',
        'EWKZ2Jets_ZToNuNu-ext','EWKZ2Jets_ZToNuNu',
        'WWTo1L1Nu2Q','WZJToLLLNu',
        'WZTo1L1Nu2Q','WZTo1L3Nu',
        'VVTo2L2Nu-ext1','VVTo2L2Nu',
        'WZTo2L2Q','ZZTo2L2Q','ZZTo4L-amcat',
        'GluGluHToWWTo2L2Nu_M-125','ZHToTauTau_M-125',
        'T-tW','T-t','Tbar-tW','Tbar-t']
    }


# directory of the files (usually /vols/cms/)
path = '/vols/cms/akd116/Offline/output/SM/2018/Mar19'

ggh = []
for sig in sig_files:
    print sig
    sig_tmp = lf.load_mc_ntuple(
            '{}/{}_{}_2016.root'.format(path, sig, opt.channel),
            'ntuple',
            features,
            opt.sig_sample,
            opt.channel,
            cut_features,
            apply_cuts=opt.apply_selection
            )
    ## need to multiply event weight by
    ## (XS * Lumi) / #events
    xs_tmp = params_file[sig]['xs']
    events_tmp = params_file[sig]['evt']
    sig_tmp['process'] = sig
    sig_tmp['wt'] = sig_tmp['wt'] * (xs_tmp * lumi)/events_tmp

    ## test multi_classes
    if opt.mode in ['keras_multi', 'xgb_multi']:
        for key, value in class_dict.iteritems():
            if sig in value:
                sig_tmp['multi_class'] = key
        # for key, value in class_weight_dict.iteritems():
        #     if sig_tmp['multi_class'].iloc[0] == key:
        #         sig_tmp['wt'] = value * sig_tmp['wt'] * (xs_tmp * lumi)/events_tmp
    # else:
        # sig_tmp['wt'] = sig_tmp['wt'] * (xs_tmp * lumi)/events_tmp

    ggh.append(sig_tmp)

ggh = pd.concat(ggh, ignore_index=True)

# pf.plot_correlation_matrix(
#         ggh.drop(['wt'], axis=1),
#         'ggh_{}_{}_correlation_matrix.pdf'.format(opt.channel, opt.sig_sample))

bkgs_tmp = []
for bkg in bkg_files:
    print bkg
    bkg_tmp = lf.load_mc_ntuple(
            '{}/{}_{}_2016.root'.format(path, bkg, opt.channel),
            'ntuple',
            features,
            opt.sig_sample,
            opt.channel,
            cut_features,
            apply_cuts=opt.apply_selection
            )
    ztt_tmp = pd.DataFrame()
    zl_tmp = pd.DataFrame()
    zj_tmp = pd.DataFrame()
    zll_tmp = pd.DataFrame()

    ## need to multiply event weight by
    ## (XS * Lumi) / #events
    xs_tmp = params_file[bkg]['xs']
    events_tmp = params_file[bkg]['evt']
    if len(bkg_tmp) >= 1:
        bkg_tmp['process'] = bkg
        bkg_tmp['wt'] = bkg_tmp['wt'] * (xs_tmp * lumi)/events_tmp

        if opt.mode in ['keras_multi', 'xgb_multi']:
            for key, value in class_dict.iteritems():
                if bkg in value:
                    bkg_tmp['multi_class'] = key
            # for key, value in class_weight_dict.iteritems():
            #     if bkg_tmp['multi_class'].iloc[0] == key:
            #         bkg_tmp['wt'] = value * bkg_tmp['wt'] * (xs_tmp * lumi)/events_tmp
        # else:
        #     bkg_tmp['wt'] = bkg_tmp['wt'] * (xs_tmp * lumi)/events_tmp

        ## need to genmatch for splitting DY into ZTT and ZLL
        if opt.mode in ['keras_multi', 'xgb_multi'] and bkg in [
            'DYJetsToLL_M-10-50-LO','DY1JetsToLL-LO','DY2JetsToLL-LO',
            'DY3JetsToLL-LO','DY4JetsToLL-LO','DYJetsToLL-LO-ext1',
            'DYJetsToLL-LO-ext2',]:

            if opt.channel == 'tt':
                ztt_tmp = bkg_tmp[(bkg_tmp['gen_match_1'] == 5) & (bkg_tmp['gen_match_2'] == 5)]
                ztt_tmp.reset_index(drop=True)
                ztt_tmp['multi_class'] = 'ztt'
                # print ztt_tmp.shape

                zl_tmp = bkg_tmp[
                    ((bkg_tmp['gen_match_1'] != 6) | (bkg_tmp['gen_match_2'] != 6))
                    & ((bkg_tmp['gen_match_1'] != 5) & (bkg_tmp['gen_match_2'] != 5))
                    ]
                zl_tmp.reset_index(drop=True)
                zl_tmp['multi_class'] = 'zll' ## zl -- > zll
                # print zl_tmp.shape

                zj_tmp = bkg_tmp[(bkg_tmp['gen_match_1'] == 6) | (bkg_tmp['gen_match_2'] == 6)]
                zj_tmp.reset_index(drop=True)
                zj_tmp['multi_class'] = 'misc' ## zj --> misc
                # print zj_tmp.shape

            if opt.channel in ['mt', 'et']:
                ztt_tmp = bkg_tmp[(bkg_tmp['gen_match_2'] == 5)]
                ztt_tmp.reset_index(drop=True)
                ztt_tmp['multi_class'] = 'ztt'
                # print ztt_tmp

                zl_tmp = bkg_tmp[(bkg_tmp['gen_match_2'] != 6) & (bkg_tmp['gen_match_2'] != 5)]
                zl_tmp.reset_index(drop=True)
                zl_tmp['multi_class'] = 'zll' ## zl --> zll
                # print zl_tmp

                zj_tmp = bkg_tmp[(bkg_tmp['gen_match_2'] == 6)]
                zj_tmp.reset_index(drop=True)
                zj_tmp['multi_class'] = 'misc' ## zj --> misc
                # print zj_tmp

            if opt.channel == 'em':
                ztt_tmp = bkg_tmp[(bkg_tmp['gen_match_1'] > 2) & (bkg_tmp['gen_match_2'] > 3)]
                ztt_tmp.reset_index(drop=True)
                ztt_tmp['multi_class'] = 'ztt'
                # print ztt_tmp

                zll_tmp = bkg_tmp[(bkg_tmp['gen_match_1'] < 3) | (bkg_tmp['gen_match_2'] < 4)]
                zll_tmp.reset_index(drop=True)
                zll_tmp['multi_class'] = 'zll'
                # print zll_tmp


            for zsplit in [ztt_tmp, zl_tmp, zj_tmp, zll_tmp]:
                bkgs_tmp.append(zsplit)
                # print len(bkgs_tmp)

        else:
            bkgs_tmp.append(bkg_tmp)

        # print len(bkgs_tmp)

bkgs = pd.concat(bkgs_tmp, ignore_index=True)

print bkgs.shape

qcd_tmp = []
for data in data_files:
    print data
    data_tmp = lf.load_data_ntuple(
            '{}/{}_{}_2016.root'.format(path, data, opt.channel),
            'ntuple',
            features,
            opt.sig_sample,
            opt.channel,
            cut_features,
            apply_cuts=opt.apply_selection
            )

    data_tmp['process'] = data
    if opt.mode in ['keras_multi', 'xgb_multi']:
        for key, value in class_dict.iteritems():
            if data in value:
                data_tmp['multi_class'] = key
        # for key, value in class_weight_dict.iteritems():
        #     if data_tmp['multi_class'].iloc[0] == key:
        #         data_tmp['wt'] = value * data_tmp['wt']
    qcd_tmp.append(data_tmp)
qcd = pd.concat(qcd_tmp, ignore_index=True)

# full background DataFrame
bkgs = pd.concat([bkgs, qcd], ignore_index=True)

print bkgs.shape


# pf.plot_signal_background(
#         ggh, bkgs, 'm_sv',
#         opt.channel, opt.sig_sample,
#         bins=100
        # )

# pf.plot_roc_cutbased(ggh, bkgs, 'm_sv', opt.channel, opt.sig_sample)

# pf.plot_correlation_matrix(
#         bkgs.drop(['wt'], axis=1),
#         'bkgs_{}_{}_correlation_matrix.pdf'.format(opt.channel, opt.sig_sample))


## for ggh vs rest discrimation
y_sig = pd.DataFrame(np.ones(ggh.shape[0]))
y_bkgs = pd.DataFrame(np.zeros(bkgs.shape[0]))
y = pd.concat([y_sig, y_bkgs])
y.columns = ['class']


X = pd.concat([ggh, bkgs])
X['deta'] = np.abs(X['eta_1'] - X['eta_2'])
X['class'] = y.values


# pf.plot_correlation_matrix(X, 'correlation_matrix.pdf')

if opt.apply_selection:
    X.to_hdf('data/dataset_cpsm_{}_{}.hdf5'
        .format(opt.channel, opt.sig_sample),
        key='X',
        mode='w')
else:
    X.to_hdf('data/dataset_full_cpsm_{}.hdf5'
        .format(opt.channel),
        key='X',
        mode='w')
