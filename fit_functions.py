import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plot_functions as pf
from scipy import interp
from root_numpy import array2root

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import *
from keras.optimizers import Nadam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import np_utils


from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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
                'max_depth':3,
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
                'max_depth':7,
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


def fit_multiclass_ttsplit(X, channel, sig_sample):

    X = X.sample(frac=1).reset_index(drop=True)

    X_train,X_test, y_train,y_test,w_train,w_test  = train_test_split(
        X,
        X['multi_class'],
        X['wt'],
        test_size=0.30,
        random_state=123456,
        )

    ## FINISH THIS FOR CLASS WEIGHTS CALC
    # class_weights = compute_class_weights(X_train)
    # print class_weights
    # sum_w = X_train['wt'].sum()
    # print sum_w

    # data_gb = X_train.groupby('multi_class')
    # dict_data_gb = {x: data_gb.get_group(x) for x in data_gb.groups}
    # print dict_data_gb

    # class_weights = []
    # # calculate sum of event weights per category
    # sum_w_cat = []
    # for cat in X_train['multi_class']:
    #     if X_train['multi_class'] == cat:
    #         sum_w_cat.append(X_train['wt'])
    #         print 'individual', sum_w_cat
    #     print 'full cat', sum_w_cat

        # try:
        #     print 'category {}'.format(cat)
        #     weights = sum_w / sum_w_cat
        #     print weights
        #     class_weights.append(weights)
        # except ZeroDivisionError:
        #     'Cannot divide by zero'

    # print class_weights

    sum_w = X_train['wt'].sum()
    # print 'sum_w', sum_w
    sum_w_cat = X_train.groupby('multi_class')['wt'].sum()
    # print 'sum_w_cat', sum_w_cat
    class_weights = sum_w / sum_w_cat

    class_weight_dict = dict(class_weights)

    print class_weight_dict

    for i in w_train.index:
        for key, value in class_weight_dict.iteritems():
        # print 'before: ',index, row
            if y_train[i] == key:
                w_train.at[i] *= value
                # print 'after dividing by class_weight: ',index, row

    ## use one-hot encoding
    # encode class values as integers
    encoder_train = LabelEncoder()
    encoder_test = LabelEncoder()
    encoder_train.fit(y_train)

    y_train = encoder_train.transform(y_train)

    encoder_test.fit(y_test)
    y_test = encoder_test.transform(y_test)



    # test_class_weight = class_weight.compute_class_weight(
    #     'balanced', np.unique(encoded_Y), encoded_Y
    #     )
    # print test_class_weight

    # print 'original Y: ', X_train['multi_class'].head()
    # print 'one-hot y: ', y_train


    X_train = X_train.drop([
        'wt', 'process', 'multi_class', 'class',
        'gen_match_1', 'gen_match_2'
        ], axis=1).reset_index(drop=True)

    X_test = X_test.drop([
        'wt', 'process', 'multi_class', 'class',
        'gen_match_1', 'gen_match_2'
        ], axis=1).reset_index(drop=True)


    ## SOME TESTS WITH WEIGHTS
    # w_train *= (sum(w) / sum(w_train))
    # w_test *= (sum(w) / sum(w_test))



    # sum_wpos = np.sum(w_train[y_train == 1])
    # sum_wneg = np.sum(w_train[y_train != 1])
    # ratio = sum_wneg / sum_wpos

    # X_train = X_train.drop(['wt', 'class', 'eta_1', 'eta_2'], axis=1).reset_index(drop=True)
    # X_test = X_test.drop(['wt', 'class', 'eta_1', 'eta_2'], axis=1).reset_index(drop=True)

    # if channel == 'tt':

    # if sig_sample == 'powheg':
    #     params = {
    #             'objective':'multi:softprob',
    #             'max_depth':3,
    #             'min_child_weight':1,
    #             'learning_rate':0.01,
    #             'silent':1,
    #             # 'scale_pos_weight':ratio,
    #             'n_estimators':2000,
    #             'gamma':1.0,
    #             'subsample':0.7,
    #             'colsample_bytree':0.8,
    #             'max_delta_step':1,
    #             'nthread':-1,
    #             'seed':123456
    #             }


    if sig_sample in ['JHU','powheg']:
        if channel == 'tt':
            params = {
                    'objective':'multi:softprob',
                    'max_depth':8,
                    'min_child_weight':1,
                    'learning_rate':0.025,
                    'silent':1,
                    # 'scale_pos_weight':ratio,
                    'n_estimators':100,
                    'gamma':2.0,
                    'subsample':0.9,
                    'colsample_bytree':0.9,
                    'max_delta_step':1,
                    'nthread':-1,
                    'seed':123456
                    }

        if channel in ['mt']:
            params = {
                    'objective':'multi:softprob',
                    'max_depth':8,
                    'min_child_weight':1,
                    'learning_rate':0.025,
                    'silent':1,
                    # 'scale_pos_weight':ratio,
                    'n_estimators':100,
                    'gamma':2.0,
                    'subsample':0.9,
                    'colsample_bytree':0.9,
                    # 'max_delta_step':1,
                    'nthread':-1,
                    'seed':123456
                    }

        if channel in ['et']:
            params = {
                    'objective':'multi:softprob',
                    'max_depth':7,
                    'min_child_weight':1,
                    'learning_rate':0.025,
                    'silent':1,
                    # 'scale_pos_weight':ratio,
                    'n_estimators':100,
                    'gamma':2.0,
                    'subsample':0.9,
                    'colsample_bytree':0.9,
                    # 'max_delta_step':1,
                    'nthread':-1,
                    'seed':123456
                    }

        if channel == 'em':
            params = {
                    'objective':'multi:softprob',
                    'max_depth':8,
                    'min_child_weight':1,
                    'learning_rate':0.025,
                    'silent':1,
                    # 'scale_pos_weight':ratio,
                    'n_estimators':100,
                    'gamma':2.0,
                    'subsample':0.9,
                    'colsample_bytree':0.9,
                    'max_delta_step':1,
                    'nthread':-1,
                    'seed':123456
                    }

    xgb_clf = xgb.XGBClassifier(**params)

    xgb_clf.fit(
            X_train,
            y_train,
            sample_weight = w_train,
            early_stopping_rounds=10,
            eval_set=[(X_train, y_train, w_train), (X_test, y_test, w_test)],
            eval_metric = ['mlogloss'],
            verbose=True
            )

    # evals_result = xgb_clf.evals_result()

    y_predict = xgb_clf.predict(X_test)
    print 'true label: {},{},{}'.format(y_test[0],y_test[1],y_test[2])
    print 'predicted label: {},{},{}'.format(y_predict[0],y_predict[1],y_predict[2])


    print classification_report(
            y_test,
            y_predict,
            # target_names=["background", "signal"],
            target_names=list(encoder_test.classes_),
            sample_weight=w_test
            )


    y_pred = xgb_clf.predict_proba(X_test)
    print 'highest proba: {},{},{}'.format(max(y_pred[0]),max(y_pred[1]),max(y_pred[2]))

    # print y_pred
    # proba_predict_train = xgb_clf.predict_proba(X_train)[:,1]
    # proba_predict_test = xgb_clf.predict_proba(X_test)[:,1]

    ## 15% of highest probablilty output

    # Make predictions for s and b

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
    with open('multi_{}_{}_xgb.pkl'.format(channel, sig_sample), 'w') as f:
        pickle.dump(xgb_clf, f)


#     auc = roc_auc_score(y_test, y_pred[:,1])
#     print auc
#     fpr, tpr, _ = roc_curve(y_test, y_pred[:,1])

    # pf.plot_roc_curve(
    #         fpr, tpr, auc,
    #         'multi_{}_{}_roc.pdf'.format(channel, sig_sample))

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

    # pf.plot_output(
    #         xgb_clf.booster(),
    #         xg_train, xg_test,
    #         y_train, y_test,
    #         'multi_{}_{}_output.pdf'.format(channel, sig_sample))

    pf.plot_features(
            xgb_clf.booster(),
            'weight',
            'multi_{}_{}_features_weight.pdf'.format(channel, sig_sample))

    pf.plot_features(
            xgb_clf.booster(),
            'gain',
            'multi_{}_{}_features_gain.pdf'.format(channel, sig_sample))


    y_prediction = xgb_clf.predict(X_test)

    pf.plot_confusion_matrix(
            y_test, y_prediction, w_test,
            # classes=['background', 'signal'],
            classes=list(encoder_test.classes_),
            figname='multi_{}_{}_non-normalised_weights_cm.pdf'.format(channel, sig_sample),
            normalise=False)

    pf.plot_confusion_matrix(
            y_test, y_prediction, w_test,
            classes=list(encoder_test.classes_),
            figname='multi_{}_{}_normalised_weights_cm.pdf'.format(channel, sig_sample),
            normalise=True)

    return None


def fit_keras(X, channel, sig_sample):
    ### TEST A KERAS MODEL

    X = X.sample(frac=1).reset_index(drop=True)

    X_train,X_test, y_train,y_test,w_train,w_test  = train_test_split(
        X,
        X['multi_class'],
        X['wt'],
        test_size=0.30,
        random_state=123456,
        )


    ## use one-hot encoding
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y_train = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_train = np_utils.to_categorical(encoded_y_train, num_classes=7)
    encoder.fit(y_test)
    encoded_y_test = encoder.transform(y_test)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_test = np_utils.to_categorical(encoded_y_test, num_classes=7)

    # test_class_weight = class_weight.compute_class_weight(
    #     'balanced', np.unique(encoded_Y), encoded_Y
    #     )
    # print test_class_weight

    print 'original Y: ', X['multi_class'].head()
    print 'one-hot y: ', y_train[0]


    X_train = X_train.drop(['wt', 'multi_class', 'class'], axis=1).reset_index(drop=True)
    X_test = X_test.drop(['wt', 'multi_class', 'class'], axis=1).reset_index(drop=True)


    ## standard scaler
    columns = X_train.columns
    scaler = StandardScaler()
    np_scaled_train = scaler.fit_transform(X_train.as_matrix())
    scaled_train = pd.DataFrame(np_scaled_train)
    scaled_train.columns = columns

    np_scaled_test = scaler.fit_transform(X_test.as_matrix())
    scaled_test = pd.DataFrame(np_scaled_test)
    scaled_test.columns = columns


    # if channel == 'tt':

    # if sig_sample == 'powheg':
    #     params = {
    #             'objective':'binary:logistic',
    #             'max_depth':3,
    #             'min_child_weight':1,
    #             'learning_rate':0.01,
    #             'silent':1,
    #             'scale_pos_weight':ratio,
    #             'n_estimators':2000,
    #             'gamma':1.0,
    #             'subsample':0.7,
    #             'colsample_bytree':0.8,
    #             'max_delta_step':1,
    #             'nthread':-1,
    #             'seed':123456
    #             }

    ## how many features
    num_inputs = X_train.shape[1]
    ## how many classes
    num_outputs = 7

    model = Sequential()
    model.add(
        Dense(
            200,
            init='glorot_normal',
            activation='tanh',
            W_regularizer=l2(1e-4),
            input_dim=num_inputs
            )
        )
    model.add(
        Dense(
            200,
            init='glorot_normal',
            activation='tanh',
            W_regularizer=l2(1e-4),
            )
        )
    model.add(
        Dense(
            200,
            init='glorot_normal',
            activation='tanh',
            W_regularizer=l2(1e-4),
            )
        )
    model.add(
        Dense(
            num_outputs,
            init='glorot_normal',
            activation='softmax'
            )
        )
    model.compile(
        loss='mean_squared_error',
        optimizer=Nadam(),
        metrics=['acc']
        )


    ## add early stopping
    callbacks = []
    callbacks.append(
        EarlyStopping(patience=100)
        )

    model.summary()
    model.fit(
        # X_train,
        scaled_train,
        y_train,
        # class_weight=test_class_weight,
        # sample_weight=w_train,
        # validation_data=(X_test,y_test,w_test),
        validation_data=(scaled_test,y_test),#w_test),
        batch_size=32,
        epochs=10,
        shuffle=True,
        callbacks=callbacks
        )

    model.save('keras_model_{}_{}.h5'.format(channel, sig_sample))
    # model.save_weights('keras_model_weights_{}_{}.h5'.format(channel, sig_sample))

    return None



def write_score(data, model, channel, doSystematics):

    path = '/vols/cms/akd116/Offline/output/SM/2018/Mar18' # path of nominal ntuples

    # for full systematics need this:
    systematics = [
            'TSCALE_UP', 'TSCALE_DOWN', 'TSCALE0PI_UP', 'TSCALE0PI_DOWN', 'TSCALE1PI_UP',
            'TSCALE1PI_DOWN', 'TSCALE3PRONG_UP', 'TSCALE3PRONG_DOWN' , 'JES_UP', 'JES_DOWN',
            'EFAKE0PI_DOWN', 'EFAKE0PI_UP', 'EFAKE1PI_DOWN', 'EFAKE1PI_UP', 'MUFAKE0PI_DOWN' ,
            'MUFAKE0PI_UP', 'MUFAKE1PI_DOWN', 'MUFAKE1PI_UP', 'METUNCL_UP', 'METUNCL_DOWN',
            'METCL_UP', 'METCL_DOWN',

            # 'TSCALE_UP_1', 'TSCALE_UP_2', 'TSCALE_DOWN_2', 'TSCALE_UP_3', 'TSCALE_DOWN_3',
            # 'TSCALE_UP_0.5', 'TSCALE_DOWN_0.5', 'TSCALE_UP_1.5', 'TSCALE_DOWN_1.5', 'TSCALE_UP_2.5',
            # 'TSCALE_DOWN_2.5', 'BTAG_UP', 'BTAG_DOWN', 'BFAKE_UP', 'BFAKE_DOWN',
            # 'HF_UP', 'HF_DOWN', 'HFSTATS1_UP', 'HFSTATS1_DOWN', 'HFSTATS2_UP',
            # 'HFSTATS2_DOWN', 'CFERR1_UP', 'CFERR1_DOWN', 'CFERR2_UP', 'CFERR2_DOWN',
            # 'LF_UP', 'LF_DOWN', 'LFSTATS1_UP', 'LFSTATS1_DOWN', 'LFSTATS2_UP',
            # 'LFSTATS2_DOWN', 'MET_SCALE_UP', 'MET_SCALE_DOWN', 'MET_RES_UP', 'MET_RES_DOWN',
            ]


    if len(data) > 0:
        gb = data.groupby('process')
        df_dict = {x: gb.get_group(x) for x in gb.groups}

    score = []
    for key, value in df_dict.iteritems():
        print 'Writing into {}_{}_2016.root'.format(key, channel)
        value = value.drop(['process'], axis=1)
        if len(data) > 0:
            score = model.predict_proba(value)[:,1]
        else:
            score = np.array(0.0)

        score.dtype = [('mva_score', np.float32)]
        array2root(
            score,
            '{}/{}_{}_2016.root'.format(path, key, channel),
            'ntuple',
            mode = 'update'
            )


        if doSystematics:
            for systematic in systematics:
                print 'Writing into {}/{}_{}_2016.root'.format(systematic, key, channel)

                array2root(
                    score,
                    '{}/{}/{}_{}_2016.root'.format(path, systematic, key, channel),
                    'ntuple',
                    mode = 'update'
                    )

    return None




def write_score_multi(data, model, channel, doSystematics):
    ## START EDITING THIS

    path = '/vols/cms/akd116/Offline/output/SM/2018/Mar19'

    # for full systematics need this:
    systematics = [
            'TSCALE_UP', 'TSCALE_DOWN', 'TSCALE0PI_UP', 'TSCALE0PI_DOWN', 'TSCALE1PI_UP',
            'TSCALE1PI_DOWN', 'TSCALE3PRONG_UP', 'TSCALE3PRONG_DOWN' , 'JES_UP', 'JES_DOWN',
            'EFAKE0PI_DOWN', 'EFAKE0PI_UP', 'EFAKE1PI_DOWN', 'EFAKE1PI_UP', 'MUFAKE0PI_DOWN' ,
            'MUFAKE0PI_UP', 'MUFAKE1PI_DOWN', 'MUFAKE1PI_UP', 'METUNCL_UP', 'METUNCL_DOWN',
            'METCL_UP', 'METCL_DOWN',

            # 'TSCALE_UP_1', 'TSCALE_UP_2', 'TSCALE_DOWN_2', 'TSCALE_UP_3', 'TSCALE_DOWN_3',
            # 'TSCALE_UP_0.5', 'TSCALE_DOWN_0.5', 'TSCALE_UP_1.5', 'TSCALE_DOWN_1.5', 'TSCALE_UP_2.5',
            # 'TSCALE_DOWN_2.5', 'BTAG_UP', 'BTAG_DOWN', 'BFAKE_UP', 'BFAKE_DOWN',
            # 'HF_UP', 'HF_DOWN', 'HFSTATS1_UP', 'HFSTATS1_DOWN', 'HFSTATS2_UP',
            # 'HFSTATS2_DOWN', 'CFERR1_UP', 'CFERR1_DOWN', 'CFERR2_UP', 'CFERR2_DOWN',
            # 'LF_UP', 'LF_DOWN', 'LFSTATS1_UP', 'LFSTATS1_DOWN', 'LFSTATS2_UP',
            # 'LFSTATS2_DOWN', 'MET_SCALE_UP', 'MET_SCALE_DOWN', 'MET_RES_UP', 'MET_RES_DOWN',
            ]


    if len(data) > 0:
        gb = data.groupby('process')
        df_dict = {x: gb.get_group(x) for x in gb.groups}

    score = []
    for key, value in df_dict.iteritems():
        print 'Writing into {}_{}_2016.root'.format(key, channel)
        value = value.drop(['process'], axis=1)
        if len(data) > 0:
            # assign event to max score class
            # print model.predict_proba(value)
            # print len(model.predict(value))

            for index, ls in enumerate(model.predict_proba(value)):
                # print index
                # print ls
                score.append(max(ls))
                # print score

            np_score = np.array(score)
            cat = np.array(model.predict(value))

        else:
            np_score = np.array(0.0)
            cat = ''

        np_score.dtype = [('mva_score_Mar29_powheg', np.float32)]
        cat.dtype = [('mva_cat_Mar29_powheg', np.int)]
        array2root(
            np_score,
            '{}/{}_{}_2016.root'.format(path, key, channel),
            'ntuple',
            mode = 'update'
            )
        array2root(
            cat,
            '{}/{}_{}_2016.root'.format(path, key, channel),
            'ntuple',
            mode = 'update'
            )


        if doSystematics:
            for systematic in systematics:
                print 'Writing into {}/{}_{}_2016.root'.format(systematic, key, channel)

                array2root(
                    np_score,
                    '{}/{}/{}_{}_2016.root'.format(path, systematic, key, channel),
                    'ntuple',
                    mode = 'update'
                    )
                array2root(
                    cat,
                    '{}/{}/{}_{}_2016.root'.format(path, systematic, key, channel),
                    'ntuple',
                    mode = 'update'
                    )


    return None


def compute_class_weights(df):#, channel, sig_sample):
    # calculate sum of all event weights per category
    print df['wt']
    sum_w = df['wt'].sum()
    print sum_w

    class_weights = []
    # calculate sum of event weights per category
    for cat in df['multi_class']:
        sum_w_cat = df['wt'].sum()
        try:
            weights = sum_w / sum_w_cat
            return class_weights.append(weights)
        except ZeroDivisionError:
            'Cannot divide by zero'


