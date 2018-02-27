import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plot_functions as pf
from scipy import interp

from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

def fit_ttsplit(X, channel, sig_sample):

    X = X.sample(frac=1).reset_index(drop=True)

    X_train,X_test, y_train,y_test,w_train,w_test  = train_test_split(
            X,
            X['class'],
            X['wt'],
            test_size=0.30,
            random_state=123456,
            )

    ## SOME TESTS WITH WEIGHTS
    # w_train *= (sum(w) / sum(w_train))
    # w_test *= (sum(w) / sum(w_test))

    sum_wpos = np.sum(w_train[y_train == 1])
    sum_wneg = np.sum(w_train[y_train == 0])
    ratio = sum_wneg / sum_wpos

    X_train = X_train.drop(['wt', 'class', 'eta_1', 'eta_2'], axis=1).reset_index(drop=True)
    X_test = X_test.drop(['wt', 'class', 'eta_1', 'eta_2'], axis=1).reset_index(drop=True)

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
                'seed':123456
                }


    if sig_sample == 'JHU':
        params = {
                'objective':'binary:logistic',
                'max_depth':9,
                'min_child_weight':1,
                'learning_rate':0.01,
                'silent':1,
                'scale_pos_weight':ratio,
                'n_estimators':2000,
                'gamma':2.0,
                'subsample':0.9,
                'colsample_bytree':0.9,
                # 'max_delta_step':1,
                'nthread':-1,
                'seed':123456
                }

    xgb_clf = xgb.XGBClassifier(**params)

    xgb_clf.fit(
            X_train,
            y_train,
            sample_weight = w_train,
            early_stopping_rounds=50,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric = ['mae', 'auc'],
            verbose=True
            )

    # evals_result = xgb_clf.evals_result()

    y_predict = xgb_clf.predict(X_test)
    print y_predict

    print classification_report(
            y_test,
            y_predict,
            target_names=["background", "signal"],
            sample_weight=w_test
            )


    y_pred = xgb_clf.predict_proba(X_test)

    print y_pred
    # proba_predict_train = xgb_clf.predict_proba(X_train)[:,1]
    # proba_predict_test = xgb_clf.predict_proba(X_test)[:,1]

    ## 15% of highest probablilty output

    # Make predictions for s and b

    auc = roc_auc_score(y_test, y_pred[:,1])
    print auc
    fpr, tpr, _ = roc_curve(y_test, y_pred[:,1])

    pf.plot_roc_curve(
            fpr, tpr, auc,
            '{}_{}_roc.pdf'.format(channel, sig_sample))

    # Define these so that I can use plot_output()
    xg_train = xgb.DMatrix(
            X_train,
            label=y_train,
            # missing=-9999,
            weight=w_train
            )
    xg_test = xgb.DMatrix(
            X_test,
            label=y_test,
            # missing=-9999,
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




def fit_sssplit(X, folds, channel, sig_sample):
    ## STRATIFIED SHUFFLE K FOLD

    sss = StratifiedShuffleSplit(n_splits=folds, test_size=0.3, random_state=123456)

    X = X.sample(frac=1).reset_index(drop=True)
    y = X['class']

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        print 'Fold {}/{}'.format(i+1, folds)
        X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        w_train, w_test = X_train['wt'], X_test['wt']

        X_train = X_train.drop(['wt', 'class'], axis=1).reset_index(drop=True)
        X_test = X_test.drop(['wt', 'class'], axis=1).reset_index(drop=True)

        sum_wpos = np.sum(w_train[y_train == 1])
        sum_wneg = np.sum(w_train[y_train == 0])
        ratio = sum_wneg / sum_wpos

        params = {
                'objective':'binary:logistic',
                'max_depth':3,
                'min_child_weight':10,
                'learning_rate':0.01,
                'silent':1,
                'scale_pos_weight':ratio,
                'n_estimators':2000,
                # 'gamma':0.1,
                'subsample':0.9,
                'colsample_bytree':0.9,
                # 'max_delta_step':1,
                'nthread':-1,
                'seed':123456
                }

        xgb_clf = xgb.XGBClassifier(**params)

        xgb_clf.fit(
                X_train,
                y_train,
                sample_weight = w_train,
                early_stopping_rounds=50,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_metric = ['mae', 'auc'],
                verbose=True
                )


        probas_ = xgb_clf.predict_proba(X_test)

        fpr, tpr, _ = roc_curve(y_test.ravel(), probas_[:,1])
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
        fig.savefig('{}fold_roc_{}_{}.pdf'.format(folds, channel, sig_sample))

    return None


def fit_gbc_ttsplit(X, channel, sig_sample):

    X = X.sample(frac=1).reset_index(drop=True)

    X_train,X_test, y_train,y_test,w_train,w_test  = train_test_split(
            X,
            X['class'],
            X['wt'],
            test_size=0.30,
            random_state=123456,
            )

    ## SOME TESTS WITH WEIGHTS
    # w_train *= (sum(w) / sum(w_train))
    # w_test *= (sum(w) / sum(w_test))

    sum_wpos = np.sum(w_train[y_train == 1])
    sum_wneg = np.sum(w_train[y_train == 0])
    ratio = sum_wneg / sum_wpos

    X_train = X_train.drop(['wt', 'class', 'eta_1', 'eta_2'], axis=1).reset_index(drop=True)
    X_test = X_test.drop(['wt', 'class', 'eta_1', 'eta_2'], axis=1).reset_index(drop=True)

    # if channel == 'tt':

    if sig_sample in ['powheg', 'JHU']:
        params = {
                'loss':'deviance',
                'max_depth':3,
                # 'min_child_weight':1,
                'learning_rate':0.1,
                'verbose':1,
                # 'scale_pos_weight':ratio,
                # 'min_samples_leaf':600,
                'n_estimators':100,
                'subsample':0.7,
                # 'colsample_bytree':0.8,
                # 'max_delta_step':1,
                'random_state':123456
                }


    # if sig_sample == 'JHU':
    #     params = {
    #             'objective':'binary:logistic',
    #             'max_depth':9,
    #             'min_child_weight':1,
    #             'learning_rate':0.01,
    #             'silent':1,
    #             'scale_pos_weight':ratio,
    #             'n_estimators':2000,
    #             'gamma':2.0,
    #             'subsample':0.9,
    #             'colsample_bytree':0.9,
    #             # 'max_delta_step':1,
    #             'nthread':-1,
    #             'seed':123456
    #             }

    gbc_clf = GradientBoostingClassifier(**params)

    gbc_clf.fit(
            X_train,
            y_train,
            sample_weight = w_train,
            )

    # evals_result = gbc_clf.evals_result()

    y_predict = gbc_clf.predict(X_test)
    print y_predict

    print classification_report(
            y_test,
            y_predict,
            target_names=["background", "signal"],
            sample_weight=w_test
            )


    decisions = gbc_clf.decision_function(X_test)

    # proba_predict_train = gbc_clf.predict_proba(X_train)[:,1]
    # proba_predict_test = gbc_clf.predict_proba(X_test)[:,1]

    ## 15% of highest probablilty output

    # Make predictions for s and b

    fpr, tpr, _ = roc_curve(y_test, decisions)
    roc_auc = auc(fpr,tpr)

    print roc_auc

#     pf.plot_roc_curve(
#             fpr, tpr, roc_auc,
#             'gbc_{}_{}_roc.pdf'.format(channel, sig_sample))

#     pf.compare_train_test(gbc_clf, X_train, y_train, X_test, y_test, 'gbc_{}_{}_output.pdf'.format(channel, sig_sample), bins=30)

    # Define these so that I can use plot_output()
    # xg_train = gbc.DMatrix(
    #         X_train,
    #         label=y_train,
    #         # missing=-9999,
    #         weight=w_train
    #         )
    # xg_test = gbc.DMatrix(
    #         X_test,
    #         label=y_test,
    #         # missing=-9999,
    #         weight=w_test
    #         )

    # pf.plot_output(
    #         gbc_clf.booster(),
    #         xg_train, xg_test,
    #         y_train, y_test,
    #         '{}_{}_output.pdf'.format(channel, sig_sample))

    # pf.plot_features(
    #         gbc_clf.booster(),
    #         'weight',
    #         '{}_{}_features_weight.pdf'.format(channel, sig_sample))

    # pf.plot_features(
    #         gbc_clf.booster(),
    #         'gain',
    #         '{}_{}_features_gain.pdf'.format(channel, sig_sample))


    # y_prediction = gbc_clf.predict(X_test)

    # pf.plot_confusion_matrix(
    #         y_test, y_prediction, w_test,
    #         classes=['background', 'signal'],
    #         figname='{}_{}_non-normalised_weights_cm.pdf'.format(channel, sig_sample),
    #         normalise=False)

    # pf.plot_confusion_matrix(
    #         y_test, y_prediction, w_test,
    #         classes=['background', 'signal'],
    #         figname='{}_{}_normalised_weights_cm.pdf'.format(channel, sig_sample),
    #         normalise=True)

    # ## SAVE FOR SKIP

    # # with open('fpr.pkl', 'w') as f:
    # #     pickle.dump(fpr, f)
    # # with open('tpr.pkl', 'w') as f:
    # #     pickle.dump(tpr, f)
    # # with open('auc.pkl', 'w') as f:
    # #     pickle.dump(auc, f)
    # # with open('X_train.pkl', 'w') as f:
    # #     pickle.dump(X_train, f)
    # # with open('y_train.pkl', 'w') as f:
    # #     pickle.dump(y_train, f)
    # # with open('X_test.pkl', 'w') as f:
    # #     pickle.dump(X_test, f)
    # # with open('y_test.pkl', 'w') as f:
    # #     pickle.dump(y_test, f)
    # # with open('w_test.pkl', 'w') as f:
    # #     pickle.dump(w_test, f)
    # # with open('w_train.pkl', 'w') as f:
    # #     pickle.dump(w_train, f)
    with open('skl_{}_{}_gbc.pkl'.format(channel, sig_sample), 'w') as f:
        pickle.dump(gbc_clf, f)

    return None


