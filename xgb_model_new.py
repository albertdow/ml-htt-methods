import random
import uproot
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from scipy import interp
from root_numpy import array2root
import json
# import seaborn as sns

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

# custom modules
import plot_functions as pf
import load_functions as lf
import fit_functions as ff


parser = argparse.ArgumentParser()

parser.add_argument('-s', action='store_true', default=False,
        dest='skip', help='skip training procedure (default False)')
parser.add_argument('--mode', action='store', default='sklearn_ttsplit',
        help='training procedure (default train_test_split)')
parser.add_argument('--channel', action='store', default='mt',
        help='channels to train on')
parser.add_argument('--sig_sample', action='store', default='powheg',
        help='''ggh signal sample to run on (default powheg)\n
        choose powheg for n_jets < 2 | (n_jets >= 2 & mjj < 300)\n
        choose JHU for n_jets >=2 & mjj > 300''')

opt = parser.parse_args()


if not opt.skip:
    print '\nTraining model on {} channel with {} sig samples\n'.format(opt.channel, opt.sig_sample)
    sig_files = lf.load_files('./filelist/{0}/{0}_sig_{1}_files.dat'.format(opt.channel, opt.sig_sample))
    bkg_files = lf.load_files('./filelist/{0}/{0}_bkgs_files.dat'.format(opt.channel))
    data_files = lf.load_files('./filelist/{0}/{0}_data_files.dat'.format(opt.channel))

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
                'mjj'
                ]

    elif opt.channel == 'mt':
        cut_features = [
                'iso_1',
                'mva_olddm_medium_2',
                'antiele_2', 'antimu_2',
                'leptonveto',
                'trg_singlemuon', 'trg_mutaucross',
                'os', 'mjj'
                ]

    elif opt.channel == 'et':
        cut_features = [
                'iso_1',
                'mva_olddm_medium_2',
                'antiele_2', 'antimu_2',
                'leptonveto',
                'trg_singleelectron',
                'os', 'mjj'
                ]

    elif opt.channel == 'em':
        cut_features = [
                'iso_1',
                'iso_2',
                'leptonveto',
                'trg_muonelectron',
                'os', 'mjj'
                ]

    # features to train on
    # apart from 'wt' - this is used for weights
    # still need to multipy 'wt' by the scaling factor
    # coming from the xsection
    features = [
            'pt_1', 'pt_2', 'eta_1', 'eta_2', 'dphi',
            'mt_1', 'mt_2', 'mt_lep',
            'm_vis', 'm_sv', 'pt_tt',
            'met', 'met_dphi_1', 'met_dphi_2',
            'n_jets', 'n_bjets',
            'wt',
            # 'pzeta', 'pt_vis'
            ]

    # directory of the files (usually /vols/cms)
    path = '/vols/cms/akd116/Offline/output/SM/2018/Feb13/'

    ggh = []
    for sig in sig_files:
        print sig
        sig_tmp = lf.load_mc_ntuple(
                path + sig + '.root',
                'ntuple',
                features,
                opt.sig_sample,
                opt.channel,
                cut_features
                )
        ## need to multiply event weight by
        ## (XS * Lumi) / #events
        xs_tmp = params_file[sig[:-8]]['xs']
        events_tmp = params_file[sig[:-8]]['evt']
        sig_tmp['wt'] = sig_tmp['wt'] * (xs_tmp * lumi)/events_tmp

        ggh.append(sig_tmp)

    ggh = pd.concat(ggh, ignore_index=True)
    print ggh


    # pf.plot_correlation_matrix(
    #         ggh.drop(['wt'], axis=1),
    #         'ggh_{}_{}_correlation_matrix.pdf'.format(opt.channel, opt.sig_sample))

    bkgs_tmp = []
    for bkg in bkg_files:
        print bkg
        bkg_tmp = lf.load_mc_ntuple(
                path + bkg + '.root',
                'ntuple',
                features,
                opt.sig_sample,
                opt.channel,
                cut_features
                )
        ## need to multiply event weight by
        ## (XS * Lumi) / #events
        xs_tmp = params_file[bkg[:-8]]['xs']
        events_tmp = params_file[bkg[:-8]]['evt']
        bkg_tmp['wt'] = bkg_tmp['wt'] * (xs_tmp * lumi)/events_tmp

        bkgs_tmp.append(bkg_tmp)
    bkgs = pd.concat(bkgs_tmp, ignore_index=True)

    qcd_tmp = []
    for data in data_files:
        print data
        data_tmp = lf.load_data_ntuple(
                path + data + '.root',
                'ntuple',
                features,
                opt.sig_sample,
                opt.channel,
                cut_features
                )

        qcd_tmp.append(data_tmp)
    qcd = pd.concat(qcd_tmp, ignore_index=True)

    # full background DataFrame
    bkgs = pd.concat([bkgs, qcd], ignore_index=True)
    print bkgs

    pf.plot_signal_background(
            ggh, bkgs, 'm_sv',
            opt.channel, opt.sig_sample,
            bins=50
            )


    # pf.plot_correlation_matrix(
    #         bkgs.drop(['wt'], axis=1),
    #         'bkgs_{}_{}_correlation_matrix.pdf'.format(opt.channel, opt.sig_sample))

    y_sig = pd.DataFrame(np.ones(ggh.shape[0]))
    y_bkgs = pd.DataFrame(np.zeros(bkgs.shape[0]))
    y = pd.concat([y_sig, y_bkgs])
    y.columns = ['class']


    ### COMMENT OUT IF WANNA
    ### TRY USING scale_pos_weight INSTEAD OF CLASS_WEIGHTS
    # class_weights = class_weight.compute_class_weight(
    #         'balanced',
    #         np.unique(np.array(y).ravel()),
    #         np.array(y).ravel()
    #         )

    # bkgs['wt'] = bkgs['wt'] * class_weights[0]
    # ggh['wt'] = ggh['wt'] * class_weights[1]


    X = pd.concat([ggh, bkgs])
    X['class'] = y.values

    w = np.array(X['wt'])
    # X = X.drop(['wt'], axis=1).reset_index(drop=True)



    # pf.plot_correlation_matrix(X, 'correlation_matrix.pdf')

    ## Just some test to correct scatter_matrix


    params = {}


############ SKLEARN WRAPPER ###########

    if opt.mode == 'sklearn_ttsplit':

        ff.fit_ttsplit(X, opt.channel, opt.sig_sample)



if opt.skip:
    if opt.mode == 'sklearn_ttsplit':
        with open('skl_{}_{}_xgb.pkl'.format(channel, sig_sample), 'r') as f:
            xgb_clf = pickle.load(f)
