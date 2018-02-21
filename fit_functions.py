import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plot_functions as pf

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

def fit_ttsplit(X, channel, sig_sample):

    X_train,X_test, y_train,y_test,w_train,w_test  = train_test_split(
            X,
            X['class'],
            X['wt'],
            test_size=0.33,
            random_state=1234
            )

    ## SOME TESTS WITH WEIGHTS
    # w_train *= (sum(w) / sum(w_train))
    # w_test *= (sum(w) / sum(w_test))

    sum_wpos = sum(w_train[y_train == 1])
    sum_wneg = sum(w_train[y_train == 0])
    ratio = sum_wneg / sum_wpos

    X_train = X_train.drop(['wt', 'class'], axis=1).reset_index(drop=True)
    X_test = X_test.drop(['wt', 'class'], axis=1).reset_index(drop=True)

    # if channel == 'tt':

    if sig_sample == 'powheg':
        params = {
                'objective':'binary:logistic',
                'max_depth':2,
                'min_child_weight':1,
                'learning_rate':0.01,
                'silent':1,
                'scale_pos_weight':ratio,
                'n_estimators':2000,
                'gamma':1.0,
                'subsample':0.7,
                'colsample_bytree':0.8,
                'max_delta_step':1,
                'nthread':-1,
                'seed':1234
                }


    if sig_sample == 'JHU':
        params = {
                'objective':'binary:logistic',
                'max_depth':4,
                'min_child_weight':2,
                'learning_rate':0.1,
                'silent':1,
                'scale_pos_weight':ratio,
                'n_estimators':2000,
                'gamma':0.1,
                'subsample':0.8,
                'colsample_bytree':0.8,
                'max_delta_step':1,
                'nthread':-1,
                'seed':1234
                }

    xgb_clf = xgb.XGBClassifier(**params)

    xgb_clf.fit(
            X_train,
            y_train,
            sample_weight = w_train,
            early_stopping_rounds=50,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric = ['auc'],
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

    pf.plot_roc_curve(
            fpr, tpr, auc,
            '{}_{}_roc.pdf'.format(channel, sig_sample))

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

    pf.plot_output(
            xgb_clf.booster(),
            xg_train, xg_test,
            y_train, y_test,
            '{}_{}_output.pdf'.format(channel, sig_sample))

    pf.plot_features(
            xgb_clf.booster(),
            'weight',
            '{}_{}_features_weight.pdf'.format(channel, sig_sample))

    pf.plot_features(
            xgb_clf.booster(),
            'gain',
            '{}_{}_features_gain.pdf'.format(channel, sig_sample))


    y_prediction = xgb_clf.predict(X_test)

    pf.plot_confusion_matrix(
            y_test, y_prediction, w_test,
            classes=['background', 'signal'],
            figname='{}_{}_non-normalised_weights_cm.pdf'.format(channel, sig_sample),
            normalise=False)

    pf.plot_confusion_matrix(
            y_test, y_prediction, w_test,
            classes=['background', 'signal'],
            figname='{}_{}_normalised_weights_cm.pdf'.format(channel, sig_sample),
            normalise=True)

    ## SAVE FOR SKIP

    # with open('fpr.pkl', 'w') as f:
    #     pickle.dump(fpr, f)
    # with open('tpr.pkl', 'w') as f:
    #     pickle.dump(tpr, f)
    # with open('auc.pkl', 'w') as f:
    #     pickle.dump(auc, f)
    # with open('X_train.pkl', 'w') as f:
    #     pickle.dump(X_train, f)
    # with open('y_train.pkl', 'w') as f:
    #     pickle.dump(y_train, f)
    # with open('X_test.pkl', 'w') as f:
    #     pickle.dump(X_test, f)
    # with open('y_test.pkl', 'w') as f:
    #     pickle.dump(y_test, f)
    # with open('w_test.pkl', 'w') as f:
    #     pickle.dump(w_test, f)
    # with open('w_train.pkl', 'w') as f:
    #     pickle.dump(w_train, f)
    with open('skl_{}_{}_xgb.pkl'.format(channel, sig_sample), 'w') as f:
        pickle.dump(xgb_clf, f)

    return None
