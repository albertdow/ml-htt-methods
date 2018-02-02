import uproot
import os
import pandas as pd
import numpy as np

def load_ntuple(data, tree, branch, cut_feats):
    # tfile = uproot.open(data)
    # events = tfile.get(tree)

    # iterator = events.iterate(branches)

    # df = events.pandas.df()

    iterator = uproot.iterate(data, tree, branches=branch+cut_feats)

    df = []
    for block in iterator:
        df_b = pd.DataFrame(block)
        df_b = df_b[(df_b['iso_1'] < 0.15) & (df_b['mva_olddm_medium_2'] > 0.5)
                & (df_b['antiele_2'] == True) & (df_b['antimu_2'] == True)
                & (df_b['leptonveto'] == False)
                & ((df_b['trg_singlemuon']*df_b['pt_1'] > 23) | (df_b['trg_mutaucross']*df_b['pt_1'] > 23))]
        df_b = df_b.drop(cut_feats, axis=1)
        df.append(df_b)

    df = pd.concat(df, ignore_index=True)

    return df

        # df.concat(df_b)


    # add iterate bit to make sure
    # i don't run out of memory

    # iterator = events.iterate()
    # for block in iterator:
    #     df = pd.DataFrame(block)
    #     yield df


def load_files(filelist):

    with open(filelist) as f:
        files = f.read().splitlines()
        file_names = [os.path.splitext(file)[0] for file in files]

    return file_names

