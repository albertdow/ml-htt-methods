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

# from class_weight import create_class_weight

def create_feature_map(fmap_filename, features):
    outfile = open(fmap_filename, 'w')
    for i, feat in enumerate(features):
        outfile.write('{}\t{}\tq\n'.format(i, feat))
    outfile.close()


parser = argparse.ArgumentParser()

parser.add_argument('-s', action='store_true', default=False,
        dest='skip', help='skip training procedure (default False)')
parser.add_argument('--mode', action='store', default='ttsplit',
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
            'wt'
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


    # pf.plot_correlation_matrix(
    #         ggh.drop(['wt'], axis=1),
    #         'ggh_correlation_matrix.pdf')

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

    # for scale_pos_weight
    n_ratio = bkgs.shape[0]/ggh.shape[0]
    print n_ratio


    # pf.plot_correlation_matrix(
    #         bkgs.drop(['wt'], axis=1),
    #         'bkgs_correlation_matrix.pdf')

    y_sig = pd.DataFrame(np.ones(ggh.shape[0]))
    y_bkgs = pd.DataFrame(np.zeros(bkgs.shape[0]))
    y = pd.concat([y_sig, y_bkgs])
    y.columns = ['class']



    ### TRY USING scale_pos_weight INSTEAD OF CLASS_WEIGHTS
    class_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(np.array(y).ravel()),
            np.array(y).ravel()
            )

    bkgs['wt'] = bkgs['wt'] * class_weights[0]
    ggh['wt'] = ggh['wt'] * class_weights[1]



    X = pd.concat([ggh, bkgs])
    X['class'] = y.values

    w = np.array(X['wt'])
    X = X.drop(['wt'], axis=1).reset_index(drop=True)

    ##



    # pf.plot_correlation_matrix(X, 'correlation_matrix.pdf')

    ## Just some test to correct scatter_matrix

    # plt.figure()
    # sns.set(style='ticks')
    # df = X
    # df['classes'] = y.values
    # sns.pairplot(df.ix[random.sample(df.index, 1000)], hue='classes')
    # plt.savefig('pairplot.pdf')
    # plt.close()
    # pf.plot_scatter_matrix(X, 'scatter_matrix.pdf')


    params = {}

    # Stratified shuffle k fold
    if opt.mode == 'kfold':

        params['objective'] = 'binary:logistic'
        params['eta'] = 0.1
        params['max_depth'] = 5
        params['nthread'] = 4
        params['silent'] = 1
        params['eval_metric'] = ['auc','error','logloss']

        num_round = 1000
        stop_round = 50

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        folds = 10
        X = X.as_matrix()
        y = y.as_matrix().ravel()

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1234)
        i = 0

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            w_train = w[train_index]
            if y[test_index] == 0.0:
                w_test = w[test_index] / class_weights[0]
                print 'should be 0.0: ', y[test_index]
                print w[test_index]
                print w_test
            elif y[test_index] == 1.0:
                w_test = w[test_index] / class_weights[1]
                print 'should be 1.0: ', y[test_index]
                print w[test_index]
                print w_test

            xg_train = xgb.DMatrix(
                    X_train,
                    label=y_train,
                    missing=-9999,
                    weight=w_train
                    )

            xg_test = xgb.DMatrix(
                    X_test,
                    label=y_test,
                    missing=-9999,
                    weight=w_test
                    )

            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            bst = xgb.train(
                    params,
                    xg_train,
                    num_round,
                    early_stopping_rounds=stop_round,
                    evals=watchlist
                    )

            probas_ = bst.predict(xg_test)

            fpr, tpr, _ = roc_curve(y_test.ravel(), probas_)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, lw=1, alpha=0.3)
            #, label='ROC fold {0} (AUC = {1:.2f})'.format(i, roc_auc))

            i += 1

        ax.plot([0,1], [0,1], 'k--')

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(
                mean_fpr,
                mean_tpr,
                'b',
                label=r'Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(mean_auc, std_auc))

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color='grey',
                alpha=.2,
                label=r'$\pm$ 1 std deviation'
                )

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid()
        ax.legend(loc='lower right')
        fig.savefig('kfold_roc_10fold.pdf')




    # Train_test_split mode
    if opt.mode == 'ttsplit':

        # TO MAKE SURE INDICES ARE CORRECT
        X_train,X_test, y_train,y_test,w_train,w_test  = train_test_split(
                X.drop(['class'], axis=1),
                X['class'],
                w,
                test_size=0.33,
                random_state=1234
                )

        ## TRY USING scale_pos_weight INSTEAD OF CLASS_WEIGHTS

        for i, val in enumerate(y_test):
            if val == 0.0:
                w_test[i] = w_test[i] / class_weights[0]
            elif val == 1.0:
                w_test[i] = w_test[i] / class_weights[1]

        xg_train = xgb.DMatrix(
                X_train,
                label=y_train,
                missing=-9999,
                weight=w_train
                )

        xg_test = xgb.DMatrix(
                X_test,
                label=y_test,
                missing=-9999,
                weight=w_test
                )


        # params['booster'] = 'dart'
        params['objective'] = 'binary:logistic'
        params['subsample'] = 0.5
        params['eta'] = 0.01
        params['max_depth'] = 6
        params['max_delta_step'] = 1
        # params['scale_pos_weight'] = n_ratio
        params['gamma'] = 1
        params['nthread'] = 4
        params['silent'] = 1
        params['eval_metric'] = ['auc','error','logloss']

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
        num_round = 2000
        stop_round = 70
        bst = xgb.train(
                params,
                xg_train,
                num_round,
                early_stopping_rounds=stop_round,
                evals=watchlist
                )

        # feat_score_gain = bst.get_score(importance_type='gain')
        # plot_importance(features, feat_score_gain, 'features_gain.pdf')

        # feat_score_weight = bst.get_score(importance_type='weight')
        # plot_importance(features, feat_score_weight, 'features_weight.pdf')

        prediction = bst.predict(xg_test)
        # fig, ax = plt.subplots()
        # ax.hist(prediction)
        # ax.set_yscale('log', nonposy='clip')
        # fig.savefig('noclassw_ttsplit_output.pdf')



        ## ROC CURVE

        fpr, tpr, _ = roc_curve(y_test, prediction)
        auc = roc_auc_score(y_test, prediction)

        pf.plot_roc_curve(fpr, tpr, auc, 'ttsplit_roc.pdf')
        ## SAVE FOR SKIP

        with open('fpr.pkl', 'w') as f:
            pickle.dump(fpr, f)
        with open('tpr.pkl', 'w') as f:
            pickle.dump(tpr, f)
        with open('auc.pkl', 'w') as f:
            pickle.dump(auc, f)
        with open('pred.pkl', 'w') as f:
            pickle.dump(prediction, f)
        with open('X_train.pkl', 'w') as f:
            pickle.dump(X_train, f)
        with open('y_train.pkl', 'w') as f:
            pickle.dump(y_train, f)
        with open('X_test.pkl', 'w') as f:
            pickle.dump(X_test, f)
        with open('y_test.pkl', 'w') as f:
            pickle.dump(y_test, f)
        with open('w_test.pkl', 'w') as f:
            pickle.dump(w_test, f)
        with open('w_train.pkl', 'w') as f:
            pickle.dump(w_train, f)
        with open('booster.pkl', 'w') as f:
            pickle.dump(bst, f)

############ SKLEARN WRAPPER ###########
    ## USE THE SKLEARN WRAPPER
    ## FOR MLGLUE

    if opt.mode == 'sklearn_ttsplit':
        X_train,X_test, y_train,y_test,w_train,w_test  = train_test_split(
                X.drop(['class'], axis=1),
                X['class'],
                w,
                test_size=0.5,
                random_state=1234
                )

        ## COMMENT NEXT FIVE LINES IF
        ## TRY USING scale_pos_weight INSTEAD OF CLASS_WEIGHTS
        for i, val in enumerate(y_test):
            if val == 0.0:
                w_test[i] = w_test[i] / class_weights[0]
            elif val == 1.0:
                w_test[i] = w_test[i] / class_weights[1]

        params = {
                'objective':'binary:logistic',
                'max_depth':7,
                'learning_rate':0.01,
                'silent':1,
                'scale_pos_weight':1,
                'n_estimators':3000,
                'gamma':1.0,
                'subsample':0.7,
                'missing':-9999.0,
                'nthread':-1,
                'seed':1234
                }

        xgb_clf = xgb.XGBClassifier(**params)

        xgb_clf.fit(
                X_train,
                y_train,
                sample_weight = w_train,
                early_stopping_rounds=70,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_metric = 'logloss',
                verbose=True
                )

        # evals_result = xgb_clf.evals_result()

        y_predict = xgb_clf.predict(X_test)
        print classification_report(
                y_test,
                y_predict,
                target_names=["background", "signal"],
                sample_weight=w_test
                )

        y_pred = xgb_clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_pred[:,1])
        fpr, tpr, _ = roc_curve(y_test, y_pred[:,1])

        pf.plot_roc_curve(fpr, tpr, auc, 'sklearn_ttsplit_roc.pdf')

        # Define these so that I can use plot_output()
        xg_train = xgb.DMatrix(
                X_train,
                label=y_train,
                missing=-9999,
                weight=w_train
                )
        xg_test = xgb.DMatrix(
                X_test,
                label=y_test,
                missing=-9999,
                weight=w_test
                )
        pf.plot_output(xgb_clf.booster(), xg_train, xg_test, y_train, y_test, 'sklearn_output.pdf')

        ## SAVE FOR SKIP

        with open('fpr.pkl', 'w') as f:
            pickle.dump(fpr, f)
        with open('tpr.pkl', 'w') as f:
            pickle.dump(tpr, f)
        with open('auc.pkl', 'w') as f:
            pickle.dump(auc, f)
        with open('X_train.pkl', 'w') as f:
            pickle.dump(X_train, f)
        with open('y_train.pkl', 'w') as f:
            pickle.dump(y_train, f)
        with open('X_test.pkl', 'w') as f:
            pickle.dump(X_test, f)
        with open('y_test.pkl', 'w') as f:
            pickle.dump(y_test, f)
        with open('w_test.pkl', 'w') as f:
            pickle.dump(w_test, f)
        with open('w_train.pkl', 'w') as f:
            pickle.dump(w_train, f)
        with open('skl_xgb.pkl', 'w') as f:
            pickle.dump(xgb_clf, f)


if opt.skip:
    if opt.mode == 'ttsplit':
        with open('fpr.pkl', 'r') as f:
            fpr = pickle.load(f)
        with open('tpr.pkl', 'r') as f:
            tpr = pickle.load(f)
        with open('auc.pkl', 'r') as f:
            auc = pickle.load(f)
        with open('pred.pkl', 'r') as f:
            prediction = pickle.load(f)
        with open('X_train.pkl', 'r') as f:
            X_train = pickle.load(f)
        with open('y_train.pkl', 'r') as f:
            y_train = pickle.load(f)
        with open('X_test.pkl', 'r') as f:
            X_test = pickle.load(f)
        with open('y_test.pkl', 'r') as f:
            y_test = pickle.load(f)
        with open('w_test.pkl', 'r') as f:
            w_test = pickle.load(f)
        with open('w_train.pkl', 'r') as f:
            w_train = pickle.load(f)
        with open('booster.pkl', 'r') as f:
            bst = pickle.load(f)

        # print prediction
        # print prediction[y_test>0.5]

            # fig, ax = plt.subplots()
            # ax.hist(prediction)
            # ax.set_yscale('log', nonposy='clip')
            # fig.savefig('ttsplit_output.pdf')

            # fpr, tpr, _ = roc_curve(y_test, prediction)
            # auc = roc_auc_score(y_test, prediction)

            # pf.plot_roc_curve(fpr, tpr, auc, 'ttsplit_roc.pdf')

        # pf.plot_output(prediction, 'noclassw_output.pdf')

        xg_train = xgb.DMatrix(X_train, label=y_train, missing=-9999, weight=w_train)
        xg_test = xgb.DMatrix(X_test, label=y_test, missing=-9999, weight=w_test)
        # pf.plot_output(bst, xg_train, xg_test, y_train, y_test, 'output.pdf')

        # roc_fig = pf.plot_roc_curve(fpr, tpr, auc, 'ttsplit_roc.pdf')

        # y_pred = [round(value) for value in prediction]
        #     ## CHECK CLASSES ORDER..........
        # pf.plot_confusion_matrix(y_test, y_pred, w_test, classes=['background', 'signal'],
        #                 figname='non-normalised_weights_cm.pdf', normalise=False)

        # pf.plot_confusion_matrix(y_test, y_pred, w_test, classes=['background', 'signal'],
        #                 figname='normalised_weights_cm.pdf', normalise=True)

        # pf.plot_features(bst, 'weight', 'features_weight.pdf')


        # pf.plot_output(bst, xg_train, xg_test, y_train, y_test, 'noclassw_output.pdf')



    if opt.mode == 'sklearn_ttsplit':
        with open('skl_xgb.pkl', 'r') as f:
            model = pickle.load(f)

        features = X_test.columns
