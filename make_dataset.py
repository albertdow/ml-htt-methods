# Usage:
#     python make_dataset.py -c --analysis cpsm --sig_sample powheg --split --mode xgb_multi --channel tt --kfold

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

def parse_arguments():
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
    parser.add_argument('--kfold', action='store_true', default=False,
        dest='kfold', help='apply kfold on dataset (default False)')
    parser.add_argument('--split', action='store_true', default=False,
        dest='split', help='split by powheg/JHU sample (default False)')
    parser.add_argument('--analysis', action='store', default='cpsm',
        dest='analysis', help='what analysis to make dataset for (default cpsm)')
    parser.add_argument('--use_jet_variables', action='store_true', default=False,
        dest='use_jet_variables', help='whether to use jet variables or not')
    parser.add_argument('--embedding', action='store_true', default=False,
        dest='use_embedding', help='Use embedded samples?')
    parser.add_argument('--ff', action='store_true', default=False,
        dest='ff', help='Use FF method?')


    return parser.parse_args()

def main(opt):
    ## To create dataset for chosen channel & sig_sample (i.e. boosted or dijet cat)

    print '\nCreate dataset for {} channel with {} sig samples\n'.format(opt.channel, opt.sig_sample)
    sig_files = lf.load_files('./filelist/sig_{}_files.dat'.format(opt.sig_sample))
    bkg_files = lf.load_files('./filelist/bkgs_files.dat')
    data_files = lf.load_files('./filelist/{}_data_files.dat'.format(opt.channel))

    # this file contains information about the xsections, lumi and event numbers
    params_file = json.load(open('Params_2016_smsummer16.json'))
    lumi = params_file['MuonEG']['lumi']

    # cut_features will only be used for preselection
    # and then dropped again
    if opt.channel == 'tt':
        cut_features = [
                'mva_olddm_tight_1', 'mva_olddm_tight_2',
                'mva_olddm_medium_1', 'mva_olddm_medium_2',
                'mva_olddm_loose_1', 'mva_olddm_loose_2',
                'antiele_1', 'antimu_1', 'antiele_2', 'antimu_2',
                'leptonveto', 'trg_doubletau',
                ]

    elif opt.channel == 'mt':
        cut_features = [
                'iso_1',
                'mva_olddm_tight_2',
                'antiele_2', 'antimu_2',
                'leptonveto',
                'trg_singlemuon', 'trg_mutaucross',
                'os',
                ]

    elif opt.channel == 'et':
        cut_features = [
                'iso_1',
                'mva_olddm_tight_2',
                'antiele_2', 'antimu_2',
                'leptonveto',
                'trg_singleelectron',
                'os',
                ]

    elif opt.channel == 'em':
        cut_features = [
                'iso_1',
                'iso_2',
                'leptonveto',
                'trg_muonelectron',
                'os',
                ]
    # if opt.analysis == 'cpsm' and not opt.use_jet_variables:
    #     cut_features.append('mjj')

    # features to train on
    # apart from 'wt' - this is used for weights
    # still need to multipy 'wt' by the scaling factor
    # coming from the xsection

    if opt.mode in ['keras_multi', 'xgb_multi']:
        if opt.analysis == 'cpsm':
            features = [
                'pt_1', 'pt_2', 'eta_1', 'eta_2',
                'mt_1', 'mt_2', 'mt_lep',
                'm_vis', 'm_sv', 'pt_tt',# 'eta_tt',
                'eta_h',
                'met', #'met_dphi_1', 'met_dphi_2',
                'n_jets', 'n_bjets',
                'pt_vis',
                'mjj','jdeta',
                'jpt_1','jpt_2',
                'dijetpt',
                ]
        if opt.channel == 'em':
            features.append('pzeta')
        if opt.use_jet_variables:
            features.extend([
                'mjj','jdeta', # use jet variables now
                'jpt_1','jpt_2',
                'dijetpt',
                ])

        if opt.analysis == 'sm':
            if opt.channel == 'tt':
                features = [
                        'm_sv','pt_1','pt_2','eta_1',
                        'phi_1','phi_2','jpt_1','jeta_1',
                        'jphi_1','jphi_2','jcsv_1','jcsv_2',
                        'jm_1','jmva_1','bpt_1','met',
                        'd0_1','beta_1','beta_2','bphi_1',
                        'bphi_2','bcsv_1','bcsv_2','n_jets',
                        'n_bjets','mt_1','pt_vis','pt_tt',
                        'mjj','jdeta','m_vis','dijetphi',
                        'dijetpt',
                        ]
            if opt.channel == 'mt':
                features = [
                        'm_sv','pt_1','pt_2','eta_2',
                        'jpt_2','jphi_2','jm_1','jm_2',
                        'bpt_1','bpt_2','beta_1','beta_2',
                        'dijetpt','met','bphi_1','bphi_2',
                        'bcsv_1','bcsv_2','n_jets','n_bjets',
                        'mt_1','mt_2','pt_vis','pt_tt',
                        'mjj','m_vis',
                        ]
            if opt.channel == 'et':
                features = [
                        'm_sv','pt_1','pt_2','jpt_1',
                        'jcsv_2','jm_1','bpt_1','bpt_2',
                        'beta_1','beta_2','bphi_1','bphi_2',
                        'met','bcsv_1','bcsv_2','n_jets',
                        'n_bjets','mt_1','mt_2','pt_vis',
                        'pt_tt','mjj','m_vis','dijetpt',
                        ]
            if opt.channel == 'em':
                features = [
                        'm_sv','pt_1','pt_2','jpt_1',
                        'jcsv_2','jm_1','bpt_1','bpt_2',
                        'beta_1','beta_2','bphi_1','bphi_2',
                        'met','bcsv_1','bcsv_2','n_jets',
                        'n_bjets','mt_1','mt_2','pt_vis',
                        'pt_tt','mjj','m_vis','dijetpt',
                        'pzeta',
                        ]

#                     'pt_1', 'pt_2', 'eta_1', 'eta_2', 'dphi',
#                     'mt_1', 'mt_2', 'mt_lep',
#                     'm_vis', 'm_sv', 'pt_tt',
#                     'met', 'met_dphi_1', 'met_dphi_2',
#                     'n_jets', 'n_bjets',
#                     'pt_vis',
#                     'phi_1', 'phi_2',
#                     # 'wt', # for training/validation weights
#                     # 'gen_match_1', 'gen_match_2', # for splitting DY into separate processes
#                     # 'event',
#                     # add more features similar to KIT take tt for now
#                     'mjj', 'jdeta',
#                     'jpt_1', 'jeta_1', 'jphi_1',
#                     'jphi_2',
#                     'jdphi',
#                     ]
    else:
        features = [
                'pt_1', 'pt_2', 'eta_1', 'eta_2', 'dphi',
                'mt_1', 'mt_2', 'mt_lep',
                'm_vis', 'm_sv', 'pt_tt', 'eta_tt',
                'met', 'met_dphi_1', 'met_dphi_2',
                'n_jets', 'n_bjets',
                'pt_vis',
                # 'wt',
                ]

    # auxiliary features for preselection
    features.extend((
            'wt', # weights
            'gen_match_1','gen_match_2', # split DY
            'event' # kfolding
            ))


    if opt.sig_sample == 'powheg' and opt.channel in ['et','mt']:
        class_dict = {
            'ggh': ['GluGluToHToTauTau_M-125',
                'GluGluH2JetsToTauTau_M125_CPmixing_sm'],
            'qqh': ['VBFHToTauTau_M-125'],
            'dy': ['DYJetsToLL_M-10-50-LO',
                'DY1JetsToLL-LO',
                'DY2JetsToLL-LO',
                'DY3JetsToLL-LO',
                'DY4JetsToLL-LO',
                'DYJetsToLL-LO-ext1',
                'DYJetsToLL-LO-ext2'],
            'fake': ['W1JetsToLNu-LO',
                'W2JetsToLNu-LO-ext',
                'W2JetsToLNu-LO',
                'W3JetsToLNu-LO-ext',
                'W3JetsToLNu-LO',
                'W4JetsToLNu-LO-ext1',
                'W4JetsToLNu-LO-ext2',
                'W4JetsToLNu-LO',
                'WJetsToLNu-LO-ext',
                'WJetsToLNu-LO',
                'SingleMuonB','SingleMuonC',
                'SingleMuonD','SingleMuonE',
                'SingleMuonF','SingleMuonG',
                'SingleMuonHv2','SingleMuonHv3',
                'SingleElectronB','SingleElectronC',
                'SingleElectronD','SingleElectronE',
                'SingleElectronF','SingleElectronG',
                'SingleElectronHv2','SingleElectronHv3',],
            'tt': ['TT'],
            'misc': ['EWKWMinus2Jets_WToLNu-ext1','EWKWMinus2Jets_WToLNu-ext2',
                'EWKWMinus2Jets_WToLNu','EWKWPlus2Jets_WToLNu-ext1',
                'EWKWPlus2Jets_WToLNu-ext2','EWKWPlus2Jets_WToLNu',
                'EWKZ2Jets_ZToLL-ext','EWKZ2Jets_ZToLL',
                'WWTo1L1Nu2Q','WZJToLLLNu',
                'WZTo1L1Nu2Q','WZTo1L3Nu',
                'VVTo2L2Nu-ext1','VVTo2L2Nu',
                'WZTo2L2Q','ZZTo2L2Q','ZZTo4L-amcat',
                'GluGluHToWWTo2L2Nu_M-125','ZHToTauTau_M-125',
                'T-tW','T-t','Tbar-tW','Tbar-t',
                'WminusHToTauTau_M-125','WplusHToTauTau_M-125',
                'WGToLNuG-ext',
                'WGToLNuG',
                'WGstarToLNuEE',
                'WGstarToLNuMuMu',
                'VBFHToWWTo2L2Nu_M-125',
                ],
            }
    if opt.sig_sample == 'powheg' and opt.channel in ['em']:
        class_dict = {
            'ggh': ['GluGluToHToTauTau_M-125',
                'GluGluH2JetsToTauTau_M125_CPmixing_sm'],
            'qqh': ['VBFHToTauTau_M-125'],
            'dy': ['DYJetsToLL_M-10-50-LO',
                'DY1JetsToLL-LO',
                'DY2JetsToLL-LO',
                'DY3JetsToLL-LO',
                'DY4JetsToLL-LO',
                'DYJetsToLL-LO-ext1',
                'DYJetsToLL-LO-ext2'],
            'qcd': ['MuonEGB','MuonEGC',
                'MuonEGD','MuonEGE',
                'MuonEGF','MuonEGG',
                'MuonEGHv2','MuonEGHv3'],
            'tt': ['TT'],
            'misc': ['EWKWMinus2Jets_WToLNu-ext1','EWKWMinus2Jets_WToLNu-ext2',
                'EWKWMinus2Jets_WToLNu','EWKWPlus2Jets_WToLNu-ext1',
                'EWKWPlus2Jets_WToLNu-ext2','EWKWPlus2Jets_WToLNu',
                'EWKZ2Jets_ZToLL-ext','EWKZ2Jets_ZToLL',
                'WWTo1L1Nu2Q','WZJToLLLNu',
                'WZTo1L1Nu2Q','WZTo1L3Nu',
                'VVTo2L2Nu-ext1','VVTo2L2Nu',
                'WZTo2L2Q','ZZTo2L2Q','ZZTo4L-amcat',
                'GluGluHToWWTo2L2Nu_M-125','ZHToTauTau_M-125',
                'T-tW','T-t','Tbar-tW','Tbar-t',
                'WminusHToTauTau_M-125','WplusHToTauTau_M-125',
                'WGToLNuG-ext',
                'WGToLNuG',
                'WGstarToLNuEE',
                'WGstarToLNuMuMu',
                'VBFHToWWTo2L2Nu_M-125',
                # w
                'W1JetsToLNu-LO',
                'W2JetsToLNu-LO-ext',
                'W2JetsToLNu-LO',
                'W3JetsToLNu-LO-ext',
                'W3JetsToLNu-LO',
                'W4JetsToLNu-LO-ext1',
                'W4JetsToLNu-LO-ext2',
                'W4JetsToLNu-LO',
                'WJetsToLNu-LO-ext',
                'WJetsToLNu-LO',
                ]
            }

    if opt.channel == 'tt':
        class_dict = {
            'ggh': ['GluGluToHToTauTau_M-125',
                'GluGluH2JetsToTauTau_M125_CPmixing_sm'],
            'qqh': ['VBFHToTauTau_M-125'],
            'dy': ['DYJetsToLL_M-10-50-LO',
                'DY1JetsToLL-LO',
                'DY2JetsToLL-LO',
                'DY3JetsToLL-LO',
                'DY4JetsToLL-LO',
                'DYJetsToLL-LO-ext1',
                'DYJetsToLL-LO-ext2'],
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
                'WWTo1L1Nu2Q','WZJToLLLNu',
                'WZTo1L1Nu2Q','WZTo1L3Nu',
                'VVTo2L2Nu-ext1','VVTo2L2Nu',
                'WZTo2L2Q','ZZTo2L2Q','ZZTo4L-amcat',
                'GluGluHToWWTo2L2Nu_M-125','ZHToTauTau_M-125',
                'T-tW','T-t','Tbar-tW','Tbar-t',
                # w
                'W1JetsToLNu-LO','W2JetsToLNu-LO-ext','W2JetsToLNu-LO',
                'W3JetsToLNu-LO-ext','W3JetsToLNu-LO','W4JetsToLNu-LO-ext1',
                'W4JetsToLNu-LO-ext2','W4JetsToLNu-LO','WGToLNuG-ext',
                'WGToLNuG','WGstarToLNuEE','WGstarToLNuMuMu',
                'WJetsToLNu-LO-ext','WJetsToLNu-LO','WminusHToTauTau_M-125',
                'WplusHToTauTau_M-125',
                # tt
                'TT',
                # vbf bkg
                'VBFHToWWTo2L2Nu_M-125',
                ]
            }

    if opt.sig_sample == 'JHU':
        if opt.channel in ['mt','et']:
            class_dict = {
                'ggh': ['GluGluToHToTauTau_M-125',
                    'GluGluH2JetsToTauTau_M125_CPmixing_sm'],
                'qqh': ['VBFHToTauTau_M-125'],
                'dy': ['DYJetsToLL_M-10-50-LO',
                    'DY1JetsToLL-LO',
                    'DY2JetsToLL-LO',
                    'DY3JetsToLL-LO',
                    'DY4JetsToLL-LO',
                    'DYJetsToLL-LO-ext1',
                    'DYJetsToLL-LO-ext2'],

                'fake': ['TauB','TauC',
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
                    'MuonEGHv2','MuonEGHv3',
                    'W1JetsToLNu-LO',
                    'W2JetsToLNu-LO-ext',
                    'W2JetsToLNu-LO',
                    'W3JetsToLNu-LO-ext',
                    'W3JetsToLNu-LO',
                    'W4JetsToLNu-LO-ext1',
                    'W4JetsToLNu-LO-ext2',
                    'W4JetsToLNu-LO',
                    'WJetsToLNu-LO-ext',
                    'WJetsToLNu-LO',
                    ],
                'misc': ['EWKWMinus2Jets_WToLNu-ext1','EWKWMinus2Jets_WToLNu-ext2',
                    'EWKWMinus2Jets_WToLNu','EWKWPlus2Jets_WToLNu-ext1',
                    'EWKWPlus2Jets_WToLNu-ext2','EWKWPlus2Jets_WToLNu',
                    'EWKZ2Jets_ZToLL-ext','EWKZ2Jets_ZToLL',
                    'WWTo1L1Nu2Q','WZJToLLLNu',
                    'WZTo1L1Nu2Q','WZTo1L3Nu',
                    'VVTo2L2Nu-ext1','VVTo2L2Nu',
                    'WZTo2L2Q','ZZTo2L2Q','ZZTo4L-amcat',
                    'GluGluHToWWTo2L2Nu_M-125','ZHToTauTau_M-125',
                    'T-tW','T-t','Tbar-tW','Tbar-t',
                    'WminusHToTauTau_M-125','WplusHToTauTau_M-125',
                    'WGToLNuG-ext',
                    'WGToLNuG',
                    'WGstarToLNuEE',
                    'WGstarToLNuMuMu',
                    # vbf bkg
                    'VBFHToWWTo2L2Nu_M-125',
                    ],
                }
        if opt.channel in ['em']:
            class_dict = {
                'ggh': ['GluGluToHToTauTau_M-125',
                    'GluGluH2JetsToTauTau_M125_CPmixing_sm'],
                'qqh': ['VBFHToTauTau_M-125'],
                'dy': ['DYJetsToLL_M-10-50-LO',
                    'DY1JetsToLL-LO',
                    'DY2JetsToLL-LO',
                    'DY3JetsToLL-LO',
                    'DY4JetsToLL-LO',
                    'DYJetsToLL-LO-ext1',
                    'DYJetsToLL-LO-ext2'],
                'tt': ['TT'],
                'qcd': [
                    'MuonEGB','MuonEGC',
                    'MuonEGD','MuonEGE',
                    'MuonEGF','MuonEGG',
                    'MuonEGHv2','MuonEGHv3'],
                'misc': ['EWKWMinus2Jets_WToLNu-ext1','EWKWMinus2Jets_WToLNu-ext2',
                    'EWKWMinus2Jets_WToLNu','EWKWPlus2Jets_WToLNu-ext1',
                    'EWKWPlus2Jets_WToLNu-ext2','EWKWPlus2Jets_WToLNu',
                    'EWKZ2Jets_ZToLL-ext','EWKZ2Jets_ZToLL',
                    'WWTo1L1Nu2Q','WZJToLLLNu',
                    'WZTo1L1Nu2Q','WZTo1L3Nu',
                    'T-tW','T-t','Tbar-tW','Tbar-t',
                    'VVTo2L2Nu-ext1','VVTo2L2Nu',
                    'WZTo2L2Q','ZZTo2L2Q','ZZTo4L-amcat',
                    'GluGluHToWWTo2L2Nu_M-125','ZHToTauTau_M-125',
                    'WminusHToTauTau_M-125','WplusHToTauTau_M-125',
                    'WGToLNuG-ext',
                    'WGToLNuG',
                    'WGstarToLNuEE',
                    'WGstarToLNuMuMu',
                    # w
                    'W1JetsToLNu-LO',
                    'W2JetsToLNu-LO-ext',
                    'W2JetsToLNu-LO',
                    'W3JetsToLNu-LO-ext',
                    'W3JetsToLNu-LO',
                    'W4JetsToLNu-LO-ext1',
                    'W4JetsToLNu-LO-ext2',
                    'W4JetsToLNu-LO',
                    'WJetsToLNu-LO-ext',
                    'WJetsToLNu-LO',
                    # vbf bkg
                    'VBFHToWWTo2L2Nu_M-125',
                    ],
                }

    # directory of the files (usually /vols/cms/)
    # path = '/vols/cms/akd116/Offline/output/SM/2018/Mar19'
    # path = '/vols/cms/akd116/Offline/output/SM/2018/Apr24_1'
    # path = '/vols/cms/akd116/Offline/output/SM/2018/May17_2'
    path = '/vols/cms/akd116/Offline/output/SM/2018/Jun22_2016_Danny'

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
                apply_cuts=opt.apply_selection,
                split_by_sample=opt.split
                )
        ## need to multiply event weight by
        ## (XS * Lumi) / #events
        xs_tmp = params_file[sig]['xs']
        events_tmp = params_file[sig]['evt']
        sig_tmp['process'] = sig
        sig_tmp['wt_xs'] = sig_tmp['wt'] * (xs_tmp * lumi)/events_tmp

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
                apply_cuts=opt.apply_selection,
                split_by_sample=opt.split
                )
        if opt.mode in ['keras_multi', 'xgb_multi'] and bkg in [
            'DYJetsToLL_M-10-50-LO','DY1JetsToLL-LO','DY2JetsToLL-LO',
            'DY3JetsToLL-LO','DY4JetsToLL-LO','DYJetsToLL-LO-ext1',
            'DYJetsToLL-LO-ext2',]:

            ztt_tmp = pd.DataFrame()
            zl_tmp = pd.DataFrame()
            zj_tmp = pd.DataFrame()
            zll_tmp = pd.DataFrame()

        if opt.channel in ['et','mt'] and opt.mode in ['keras_multi', 'xgb_multi'] and bkg in ['TT']:

            ttt_tmp = pd.DataFrame()
            ttj_tmp = pd.DataFrame()

        if opt.channel in ['et','mt'] and opt.mode in ['keras_multi', 'xgb_multi'] and bkg in ['VVTo2L2Nu-ext1','VVTo2L2Nu',]:

            vvt_tmp = pd.DataFrame()
            vvj_tmp = pd.DataFrame()

        ## need to multiply event weight by
        ## (XS * Lumi) / #events
        xs_tmp = params_file[bkg]['xs']
        events_tmp = params_file[bkg]['evt']
        if len(bkg_tmp) >= 1:
            bkg_tmp['process'] = bkg
            bkg_tmp['wt_xs'] = bkg_tmp['wt'] * (xs_tmp * lumi)/events_tmp

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

                    zl_tmp = bkg_tmp[
                        ((bkg_tmp['gen_match_1'] != 6) & (bkg_tmp['gen_match_2'] != 6))
                        & ((bkg_tmp['gen_match_1'] != 5) & (bkg_tmp['gen_match_2'] != 5))
                        ]
                    zl_tmp.reset_index(drop=True)
                    zl_tmp['multi_class'] = 'misc' ## zl -- > misc for tt training

                    zj_tmp = bkg_tmp[(bkg_tmp['gen_match_1'] == 6) | (bkg_tmp['gen_match_2'] == 6)]
                    zj_tmp.reset_index(drop=True)
                    zj_tmp['multi_class'] = 'misc' ## zj --> misc

                if opt.channel in ['mt', 'et']:
                    ztt_tmp = bkg_tmp[(bkg_tmp['gen_match_2'] == 5)] # and gen_match_1==4 for mt, gen_match_1==3 for et
                    ztt_tmp.reset_index(drop=True)
                    ztt_tmp['multi_class'] = 'ztt'

                    zl_tmp = bkg_tmp[(bkg_tmp['gen_match_2'] != 6) & (bkg_tmp['gen_match_2'] != 5)]
                    zl_tmp.reset_index(drop=True)
                    if opt.sig_sample == 'powheg':
                        zl_tmp['multi_class'] = 'zll' ## zl --> zll
                    else:
                        zl_tmp['multi_class'] = 'misc' ## zl --> misc

                    zj_tmp = bkg_tmp[(bkg_tmp['gen_match_2'] == 6)]
                    zj_tmp.reset_index(drop=True)
                    zj_tmp['multi_class'] = 'fake' ## zj --> fake


                if opt.channel == 'em':
                    ztt_tmp = bkg_tmp[(bkg_tmp['gen_match_1'] > 2) & (bkg_tmp['gen_match_2'] > 3)]
                    ztt_tmp.reset_index(drop=True)
                    ztt_tmp['multi_class'] = 'ztt'


                    zll_tmp = bkg_tmp[(bkg_tmp['gen_match_1'] < 3) | (bkg_tmp['gen_match_2'] < 4)]
                    zll_tmp.reset_index(drop=True)
                    zll_tmp['multi_class'] = 'misc'


                for zsplit in [ztt_tmp, zl_tmp, zj_tmp, zll_tmp]:
                    bkgs_tmp.append(zsplit)

            elif opt.channel in ['et','mt'] and opt.mode in ['keras_multi', 'xgb_multi'] and bkg in ['TT']:
                ttt_tmp = bkg_tmp[(bkg_tmp['gen_match_2'] == 5)]
                ttt_tmp.reset_index(drop=True)
                ttt_tmp['multi_class'] = 'tt'

                ttj_tmp = bkg_tmp[(bkg_tmp['gen_match_2'] != 5)]
                ttj_tmp.reset_index(drop=True)
                ttj_tmp['multi_class'] = 'fake'

                for fake_split in [ttt_tmp, ttj_tmp,]:
                    bkgs_tmp.append(fake_split)

            elif opt.channel in ['et','mt'] and opt.mode in ['keras_multi', 'xgb_multi'] and bkg in ['VVTo2L2Nu-ext1','VVTo2L2Nu',]:
                vvt_tmp = bkg_tmp[(bkg_tmp['gen_match_2'] == 5)]
                vvt_tmp.reset_index(drop=True)
                vvt_tmp['multi_class'] = 'misc'

                vvj_tmp = bkg_tmp[(bkg_tmp['gen_match_2'] != 5)]
                vvj_tmp.reset_index(drop=True)
                vvj_tmp['multi_class'] = 'fake'

                for fake_split in [vvt_tmp, vvj_tmp]:
                    bkgs_tmp.append(fake_split)
            else:
                bkgs_tmp.append(bkg_tmp)


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
                apply_cuts=opt.apply_selection,
                split_by_sample=opt.split
                )

        data_tmp['process'] = data
        data_tmp['wt_xs'] = data_tmp['wt']
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
    if opt.mode not in ['keras_multi','xgb_multi']:
        y_sig = pd.DataFrame(np.ones(ggh.shape[0]))
        y_bkgs = pd.DataFrame(np.zeros(bkgs.shape[0]))
        y = pd.concat([y_sig, y_bkgs])
        y.columns = ['class']


    X = pd.concat([ggh, bkgs])
    print X
    # X['deta'] = np.abs(X['eta_1'] - X['eta_2'])
    # X['class'] = y.values


    # pf.plot_correlation_matrix(X, 'correlation_matrix.pdf')


    # randomise the order of events
    # print X[(X.multi_class == 'ggh')]

    X = X.sample(
            frac=1,
            random_state=123456,
            ).reset_index(drop=True)

    # print X[(X.multi_class == 'ggh')]


    ## divide datasets
    if opt.kfold:


        # get even event numbers
        X_fold0 = X[(X['event'] % 2 == 0)]#.drop(['event'], axis=1)

        # get odd event numbers
        X_fold1 = X[(X['event'] % 2 == 1)]#.drop(['event'], axis=1)


        if opt.apply_selection:
            X_fold1.to_hdf('data_Jun22Danny/dataset_fold1_{}_{}_{}.hdf5' # odd event numbers
                .format(opt.analysis, opt.channel, opt.sig_sample),
                key='X_fold1',
                mode='w')
            X_fold0.to_hdf('data_Jun22Danny/dataset_fold0_{}_{}_{}.hdf5' # even event numbers
                .format(opt.analysis, opt.channel, opt.sig_sample),
                key='X_fold0',
                mode='w')

    else:
        if opt.apply_selection:
            X.to_hdf('data/dataset_{}_{}_{}.hdf5'
                .format(opt.analysis, opt.channel, opt.sig_sample),
                key='X',
                mode='w')
        else:
            X.to_hdf('data/dataset_full_{}_{}.hdf5'
                .format(opt.analysis, opt.channel),
                key='X',
                mode='w')

if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)
