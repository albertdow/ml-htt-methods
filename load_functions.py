import uproot
import os
import pandas as pd
import numpy as np

def load_mc_ntuple(data, tree, branch, sig, channel, cut_feats, apply_cuts, split_by_sample):
    # LOAD MC NTUPLES AND APPLY BASELINE CUTS BY CHANNEL
    # need to do something for when df too large..


    try:
        iterator = uproot.iterate(data, tree, branches=branch+cut_feats)
    except IOError:
        print 'Tree/Branches not found'

    df = []
    try:
        for block in iterator:
            df_b = pd.DataFrame(block)

            if apply_cuts:
                if channel == 'tt':
                    df_b = df_b[
                            (df_b['pt_1'] > 40)
                            & (df_b['mva_olddm_tight_1'] > 0.5)
                            & (df_b['mva_olddm_tight_2'] > 0.5)
                            & (df_b['antiele_1'] == True)
                            & (df_b['antimu_1'] == True)
                            & (df_b['antiele_2'] == True)
                            & (df_b['antimu_2'] == True)
                            & (df_b['leptonveto'] == False)
                            & (df_b['trg_doubletau'] == True)
                            ]

                elif channel == 'mt':
                    df_b = df_b[
                            (df_b['iso_1'] < 0.15)
                            & (df_b['mt_1'] < 40) # was 70 but use 40
                            & (df_b['mva_olddm_tight_2'] > 0.5)
                            & (df_b['antiele_2'] == True)
                            & (df_b['antimu_2'] == True)
                            & (df_b['leptonveto'] == False)
                            & (df_b['pt_2'] > 20)
                            & ((df_b['trg_singlemuon']*df_b['pt_1'] > 23)
                                | (df_b['trg_mutaucross']*df_b['pt_1'] < 23))
                            ]

                elif channel == 'et':
                    df_b = df_b[
                            (df_b['iso_1'] < 0.1)
                            & (df_b['mt_1'] < 40)
                            & (df_b['mva_olddm_tight_2'] > 0.5)
                            & (df_b['antiele_2'] == True)
                            & (df_b['antimu_2'] == True)
                            & (df_b['leptonveto'] == False)
                            & (df_b['pt_2'] > 20)
                            & (df_b['trg_singleelectron'] == True)
                            ]

                elif channel == 'em':
                    df_b = df_b[
                            (df_b['iso_1'] < 0.15)
                            & (df_b['iso_2'] < 0.2)
                            & (df_b['pzeta'] > -50)
                            & (df_b['leptonveto'] == False)
                            & (df_b['trg_muonelectron'] == True)
                            ]
                else:
                    assert ValueError('Channel not in ["tt", "mt", "et", "em"]')

                ## TO SELECT THE SIGNAL SAMPLE ACCORDING TO
                ## CUTS APPLIED RELATING TO n_jets AND mjj
                if not split_by_sample:
                    df_b = df_b
                else:
                    if sig == 'powheg':
                            df_b = df_b[
                                    (df_b['n_jets'] < 2)
                                    | ((df_b['n_jets'] >= 2)
                                        & (df_b['mjj'] < 300))
                                    ]
                    elif sig == 'JHU':
                            df_b = df_b[
                                    ((df_b['n_jets'] >= 2)
                                    & (df_b['mjj'] > 300))
                                    ]
                    else:
                        assert ValueError('Signal sample not in ["powheg", "JHU"]')

                df_b = df_b[(df_b['m_sv'] > 0)] ## SOME m_sv ARE MISSING

            df_b = df_b.drop(cut_feats, axis=1)
            df.append(df_b)

        df = pd.concat(df, ignore_index=True)
    except IndexError:
        print 'zero events in ntuple'

    return df

def load_data_ntuple(data, tree, branch, sig, channel, cut_feats, apply_cuts, split_by_sample):
    ## THIS FUNCTION IS FOR READING IN SAME SIGN DATA (FOR mt, et, em CHANNELS)
    ## OR ANTIISOLATED (FOR tt CHANNEL) FOR THE QCD ESTIMATION

    try:
        iterator = uproot.iterate(data, tree, branches=branch+cut_feats)
    except IOError:
        print 'Tree/Branches not found'

    df = []
    for block in iterator:
        df_b = pd.DataFrame(block)

        if apply_cuts:
            if channel == 'tt':
                df_b = df_b[
                        (df_b['pt_1'] > 40)
                        & ((df_b['mva_olddm_loose_1'] > 0.5)
                            & (df_b['mva_olddm_tight_1'] < 0.5)
                            & (df_b['mva_olddm_medium_2'] > 0.5))
                        | ((df_b['mva_olddm_loose_2'] > 0.5)
                            & (df_b['mva_olddm_tight_2'] < 0.5)
                            & (df_b['mva_olddm_medium_1'] > 0.5))
                        & (df_b['antiele_1'] == True)
                        & (df_b['antimu_1'] == True)
                        & (df_b['antiele_2'] == True)
                        & (df_b['antimu_2'] == True)
                        & (df_b['leptonveto'] == False)
                        & (df_b['trg_doubletau'] == True)
                        ]

            elif channel == 'mt':
                df_b = df_b[
                        (df_b['iso_1'] < 0.15)
                        & (df_b['mt_1'] < 40)
                        & (df_b['mva_olddm_tight_2'] > 0.5)
                        & (df_b['antiele_2'] == True)
                        & (df_b['antimu_2'] == True)
                        & (df_b['leptonveto'] == False)
                        & (df_b['pt_2'] > 20)
                        & ((df_b['trg_singlemuon']*df_b['pt_1'] > 23)
                            | (df_b['trg_mutaucross']*df_b['pt_1'] < 23))
                        & (df_b['os'] == False)
                        ]

            elif channel == 'et':
                df_b = df_b[
                        (df_b['iso_1'] < 0.1)
                        & (df_b['mt_1'] < 40)
                        & (df_b['mva_olddm_tight_2'] > 0.5)
                        & (df_b['antiele_2'] == True)
                        & (df_b['antimu_2'] == True)
                        & (df_b['leptonveto'] == False)
                        & (df_b['pt_2'] > 20)
                        & (df_b['trg_singleelectron'] == True)
                        & (df_b['os'] == False)
                        ]

            elif channel == 'em':
                df_b = df_b[
                        (df_b['iso_1'] < 0.15)
                        & (df_b['iso_2'] < 0.2)
                        & (df_b['pzeta'] > -50)
                        & (df_b['leptonveto'] == False)
                        & (df_b['trg_muonelectron'] == True)
                        & (df_b['os'] == False)
                        ]
            else:
                assert ValueError('Channel not in ["tt", "mt", "et", "em"]')

            ## TO SELECT THE SIGNAL SAMPLE ACCORDING TO
            ## CUTS APPLIED RELATING TO n_jets AND mjj
            if not split_by_sample:
                df_b = df_b
            else:
                if sig == 'powheg':
                        df_b = df_b[
                                (df_b['n_jets'] < 2)
                                | ((df_b['n_jets'] >= 2)
                                    & (df_b['mjj'] < 300))
                                ]
                elif sig == 'JHU':
                        df_b = df_b[
                                ((df_b['n_jets'] >= 2)
                                & (df_b['mjj'] > 300))
                                ]
                else:
                    assert ValueError('Signal sample not in ["powheg", "JHU"]')

            df_b = df_b[(df_b['m_sv'] > 0)] ## SOME m_sv ARE MISSING

        df_b = df_b.drop(cut_feats, axis=1)
        df.append(df_b)


    df = pd.concat(df, ignore_index=True)

    return df

def load_files(filelist):

    with open(filelist) as f:
        files = f.read().splitlines()
        file_names = [os.path.splitext(os.path.basename(file))[0] for file in files]

    return file_names

