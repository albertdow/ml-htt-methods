import random
import uproot
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from scipy import interp
# import seaborn as sns

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, recall_score, precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix

# custom modules
# from plot_functions import plot_signal_background
from plot_functions import plot_roc_curve
from plot_functions import plot_scatter_matrix
from plot_functions import plot_confusion_matrix
from plot_functions import plot_correlation_matrix
from plot_functions import plot_features

from load_functions import load_ntuple
from load_functions import load_files
# from class_weight import create_class_weight
# from data_class import Data
# from data_class import Sample

parser = argparse.ArgumentParser()

parser.add_argument('-s', action='store_true', default=False,
        dest='skip', help='skip training procedure (default False)')
parser.add_argument('--mode', action='store', default='ttsplit',
        help='training procedure (default train_test_split)')

opt = parser.parse_args()


if not opt.skip:
    ggh_file = load_files('ggh_files.txt')
    bkg_files = load_files('background_files.txt')

    path = '/vols/cms/akd116/Offline/output/SM/2018/Jan26/'

    # cut_features will only be used for preselection
    # and then dropped again
    cut_features = ['iso_1', 'mva_olddm_medium_2', 'antiele_2', 'antimu_2',
            'leptonveto', 'trg_singlemuon', 'trg_mutaucross']

    # features to train on
    features = ['pt_1', 'pt_2', 'eta_1', 'eta_2', 'dphi', 'm_vis',
            'met', 'met_dphi_1', 'met_dphi_2', 'pt_tt',
            'mt_1', 'mt_2', 'mt_lep', 'wt']


    print ggh_file[0]
    ggh = load_ntuple(path + ggh_file[0] + '.root','ntuple', features, cut_features)
    plot_correlation_matrix(ggh.drop(['wt'], axis=1), 'ggh_correlation_matrix.pdf')

    bkgs = []
    for bkg in bkg_files:
        print bkg
        bkg_tmp = load_ntuple(path + bkg + '.root','ntuple', features, cut_features)
        bkgs.append(bkg_tmp)
    bkgs = pd.concat(bkgs, ignore_index=False)
    plot_correlation_matrix(bkgs.drop(['wt'], axis=1), 'bkgs_correlation_matrix.pdf')



    y_sig = pd.DataFrame(np.ones(ggh.shape[0]))
    y_bkgs = pd.DataFrame(np.zeros(bkgs.shape[0]))
    y = pd.concat([y_sig, y_bkgs])
    y.columns = ['y']

    class_weights = class_weight.compute_class_weight('balanced',
                                                np.unique(np.array(y).ravel()),
                                                np.array(y).ravel())

    bkgs['wt'] = bkgs['wt'] * class_weights[0]
    ggh['wt'] = ggh['wt'] * class_weights[1]

    X = pd.concat([ggh, bkgs])
    w = np.array(X['wt'])
    X = X.drop(['wt'], axis=1).reset_index(drop=True)

    # plot_correlation_matrix(X, 'correlation_matrix.pdf')

    ## Just some test to correct scatter_matrix

    # plt.figure()
    # sns.set(style='ticks')
    # df = X
    # print y.values
    # df['classes'] = y.values

    # sns.pairplot(df.ix[random.sample(df.index, 1000)], hue='classes')
    # plt.savefig('pairplot.pdf')
    # plt.close()
    # plot_scatter_matrix(X, 'scatter_matrix.pdf')


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
            w_train, w_test = w[train_index], w[test_index]

            xg_train = xgb.DMatrix(X_train, label=y_train, missing=-9999, weight=w_train)
            xg_test = xgb.DMatrix(X_test, label=y_test, missing=-9999, weight=w_test)
            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            bst = xgb.train(params, xg_train, num_round, early_stopping_rounds=stop_round, evals=watchlist)

            probas_ = bst.predict(xg_test)

            fpr, tpr, _ = roc_curve(y_test.ravel(), probas_)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, lw=1, alpha=0.3)#, label='ROC fold {0} (AUC = {1:.2f})'.format(i, roc_auc))

            i += 1

        ax.plot([0,1], [0,1], 'k--')

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(mean_fpr, mean_tpr, 'b', label=r'Mean ROC (AUC = {0:.2f} $\pm$ {1:.2f})'.format(mean_auc, std_auc))

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std deviation')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid()
        ax.legend(loc='lower right')
        fig.savefig('kfold_roc_10fold.pdf')




    # Train_test_split mode
    if opt.mode == 'ttsplit':
        X_train,X_test, y_train,y_test,w_train,w_test  = train_test_split(X, y, w,
                test_size=0.5, random_state=1234)


        xg_train = xgb.DMatrix(X_train, label=y_train, missing=-9999, weight=w_train)
        xg_test = xgb.DMatrix(X_test, label=y_test, missing=-9999, weight=w_test)

        params['objective'] = 'binary:logistic'
        params['eta'] = 0.1
        params['max_depth'] = 5
        params['nthread'] = 4
        params['silent'] = 1
        params['eval_metric'] = ['auc','error','logloss']

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
        num_round = 1000
        stop_round = 50
        bst = xgb.train(params, xg_train, num_round, early_stopping_rounds=stop_round, evals=watchlist)

        # feat_score_gain = bst.get_score(importance_type='gain')
        # plot_importance(features, feat_score_gain, 'features_gain.pdf')

        # feat_score_weight = bst.get_score(importance_type='weight')
        # plot_importance(features, feat_score_weight, 'features_weight.pdf')

        prediction = bst.predict(xg_test)

        ## ROC CURVE

        fpr, tpr, _ = roc_curve(y_test, prediction)
        auc = roc_auc_score(y_test, prediction)

        with open('fpr.pkl', 'w') as f:
            pickle.dump(fpr, f)
        with open('tpr.pkl', 'w') as f:
            pickle.dump(tpr, f)
        with open('auc.pkl', 'w') as f:
            pickle.dump(auc, f)
        with open('pred.pkl', 'w') as f:
            pickle.dump(prediction, f)
        with open('y_test.pkl', 'w') as f:
            pickle.dump(y_test, f)
        with open('booster.pkl', 'w') as f:
            pickle.dump(bst, f)

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
        with open('y_test.pkl', 'r') as f:
            y_test = pickle.load(f)
        with open('booster.pkl', 'r') as f:
            bst = pickle.load(f)


    # roc_fig = plot_roc_curve(fpr, tpr, auc, 'ttsplit_roc.pdf')

    # y_pred = [round(value) for value in prediction]
    ## CHECK CLASSES ORDER..........
    # plot_confusion_matrix(y_test, y_pred, classes=['signal', 'background'],
    #                 figname='non-normalised_cm.pdf', normalise=False)

    # plot_confusion_matrix(y_test, y_pred, classes=['signal', 'background'],
    #                 figname='normalised_cm.pdf', normalise=True)

    # plot_features(bst, 'weight', 'features_weight.pdf')

