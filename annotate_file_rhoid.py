#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import logging
logger = logging.getLogger("annotate_file_rhoid.py")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

import numpy as np
import yaml
import os
import pickle
from array import array
import argparse
from sklearn.preprocessing import StandardScaler


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Apply XGB model on ROOT file")
    parser.add_argument(
        "--config-training",
        default="mt_xgb_training_config.yaml",
        help="Path to training config file")
    parser.add_argument(
        "--dir-prefix",
        type=str,
        default="ntuple",
        help="Prefix of directories in ROOT file to be annotated.")
    parser.add_argument(
        "input", help="Path to input file, where response will be added.")
    parser.add_argument(
        "tag", help="Tag to be used as prefix of the annotation.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["multi_fold1_cpsm_mt_JHU_xgb.pkl", "multi_fold0_cpsm_mt_JHU_xgb.pkl"],
        help=
        "Keras models to be used for the annotation. Note that these have to be booked in the reversed order [fold1*, fold0*], so that the training is independent from the application."
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        nargs="+",
        default=[
            "tt_training_9May_tests/tt_fold1_scaler.pkl",
            "tt_training_9May_tests/tt_fold0_scaler.pkl"
        ],
        help=
        "Data preprocessing to be used. Note that these have to be booked in the reversed order [fold1*, fold0*], so that the preprocessing is independent for the folds."
    )
    parser.add_argument(
        "--tree", default="ntuple", help="Name of trees in the directories.")
    parser.add_argument(
        "--training", default="JHU", help="Name of training to use.")
    parser.add_argument(
        "--mjj", default="high", help="mjj training to use.")
    parser.add_argument(
        "--channel", default="mt", help="Name of channel to annotate.")
    parser.add_argument(
        "--model_folder", default="mt_training_10May_mjj_jdeta_dijetpt/", help="Folder name where trained model is.")
    parser.add_argument(
        "--era", default="", help="Year to use.")

    return parser.parse_args()


def parse_config(filename):
    return yaml.load(open(filename, "r"))


def load_files(filelist):

    with open(filelist) as f:
        file_names = f.read().splitlines()

    return file_names


def main(args, config, file_names):

    path = "/vols/cms/akd116/Offline/output/SM/2019/Feb26_2016/"

    # Sanity checks
    for sample in file_names:
        print sample
        # if not os.path.exists("{}/{}_{}_{}.root".format(path, sample, args.channel, args.era)):
        if not os.path.exists("{}/{}".format(path, sample)):
            logger.fatal("Input file %s does not exist.", sample)
            raise Exception

        logger.debug("Following mapping of classes to class numbers is used.")
        for i, class_ in enumerate(config["classes"]):
            logger.debug("%s : %s", i, class_)

        # Load models and preprocessing

        with open('{}/multi_fold1_cpsm_{}_{}_{}_xgb.pkl'
                .format(args.model_folder, args.channel, args.training, args.era), 'r') as f:
            xgb_clf_fold1 = pickle.load(f)
        with open('{}/multi_fold0_cpsm_{}_{}_{}_xgb.pkl'
                .format(args.model_folder, args.channel, args.training, args.era), 'r') as f:
            xgb_clf_fold0 = pickle.load(f)

        # with open('{}_{}_scaler.pkl'
        #         .format(args.channel,args.mjj), 'r') as f:
        #     preprocessing = pickle.load(f)
        # preprocessing = StandardScaler()
        # preprocessing = [pickle.load(open(x, "rb")) for x in args.preprocessing]

        # Open input file
        file_ = ROOT.TFile("{}/{}".format(path, sample), "UPDATE")
        if file_ == None:
            logger.fatal("File %s is not existent.", sample)
            raise Exception

        tree = file_.Get(args.tree)

        # Book branches for annotation
        values = []
        for variable in config["variables"]:
            if variable in ["dijetpt","eta_h","IC_binary_test_4_score","IC_binary_test_4_index"]:
                values.append(array("f", [-9999]))
            if variable in ["eta_1","eta_2","jdeta","jpt_1","jpt_2","m_sv","m_vis","met","jeta_1","jeta_2","mt_tot","mt_sv",
                    "met_dphi_1","met_dphi_2","mjj","mt_1","mt_2","mt_lep","pt_1","pt_2","pt_h","pt_tt","pt_vis","pzeta","dR"]:
                values.append(array("d", [-9999]))
            if variable in ["n_jets","n_bjets","opp_sides"]:
                values.append(array("I", [0]))
            if variable not in ["zfeld","centrality","mjj_jdeta","dijetpt_pth","dijetpt_jpt1","dijetpt_pth_over_pt1",
                    "msv_mvis","msvsq_mvis","msv_sq","log_metsq_jeta2","met_jeta2","oppsides_centrality","pthsq_ptvis","msv_rec","dR_custom","rms_pt","rms_jpt","rec_sqrt_msv"]:
                tree.SetBranchAddress(variable, values[-1])
            # else:
            #     tree.SetBranchAddress("eta_h", values[-1])

        response_max_score = array("f", [-9999])
        branch_max_score = tree.Branch("{}_max_score".format(
            args.tag), response_max_score, "{}_max_score/F".format(
                args.tag))

        response_max_index = array("f", [-9999])
        branch_max_index = tree.Branch("{}_max_index".format(
            args.tag), response_max_index, "{}_max_index/F".format(
                args.tag))

        # response_ggh_score = array("f", [-9999])
        # branch_ggh_score = tree.Branch("{}_ggh_score".format(
        #     args.tag), response_ggh_score, "{}_ggh_score/F".format(
        #         args.tag))

        # response_qqh_score = array("f", [-9999])
        # branch_qqh_score = tree.Branch("{}_qqh_score".format(
        #     args.tag), response_qqh_score, "{}_qqh_score/F".format(
        #         args.tag))

        # Run the event loop
        for i_event in range(tree.GetEntries()):
            tree.GetEntry(i_event)

            # Get event number and compute response
            event = int(getattr(tree, "event"))
            m_sv = float(getattr(tree, "m_sv"))
            # m_vis = float(getattr(tree, "m_vis"))

            if m_sv > 0:
                zfeld = float(getattr(tree,"eta_h")) - (float(getattr(tree,"jeta_1"))+float(getattr(tree,"jeta_2")))/2
                centrality = np.exp(-4*(zfeld/np.fabs(float(getattr(tree,"jdeta"))))**2)
                dphi_custom = np.arccos(1-float(getattr(tree,"mt_lep"))**2/(2.*float(getattr(tree,"pt_1"))*float(getattr(tree,"pt_2"))))
                dR_custom = np.sqrt((float(getattr(tree,"eta_1"))-float(getattr(tree,"eta_2")))**2 + dphi_custom**2)

                # additional_vars = [
                #         zfeld,centrality,mjj_jdeta,dijetpt_pth,dijetpt_jpt1,dijetpt_pth_over_pt1,
                #         msv_mvis,msvsq_mvis,msv_sq,log_metsq_jeta2,met_jeta2,oppsides_centrality,pthsq_ptvis,msv_rec
                #         ]
                additional_vars = [
                        centrality,#mjj_jdeta,
                        #msvsq_mvis,pthsq_ptvis,
                        dR_custom,
                        ]
                if len(values) < len(config["variables"]):
                    values.extend(additional_vars) ## for the first loop
                    # values.append(centrality) ## for the first loop
                else:
                    for index,val in enumerate(additional_vars):
                        ind = len(additional_vars)-index
                        values[-ind] = val
                    # values[-1] = centrality # then replace the last value
                values_stacked = np.hstack(values).reshape(1, len(values))
                # values_preprocessed = preprocessing.fit_transform(
                #     values_stacked)
                # response = classifier[event % 2].predict_proba(values_preprocessed)
                response = classifier[event % 2].predict_proba(values_stacked,
                        ntree_limit=classifier[event % 2].best_iteration+1)
                response = np.squeeze(response)

                # Find max score and index
                response_max_score[0] = -9999.0
                for i, r in enumerate(response):
                    if r > response_max_score[0]:
                        response_max_score[0] = r
                        response_max_index[0] = i

                # rms = np.sqrt(1./3. * (response[1]**2+response[2]**2+response[3]**2))
                # if response_max_index[0] == 0 and (response[1]>0.3 or response[2]>0.3 or response[3]>0.3):
                #     response_max_score[0] += rms
                # print response
                # print response_max_score[0]
                # exit()

                # # Take ggH score as well
                # response_ggh_score[0] = -9999.0
                # # if args.channel in ['tt','em']:
                # response_ggh_score[0] = response[0]
                # # elif args.channel in ['mt','et']:
                # #     response_ggh_score[0] = response[1]

                # # Take qqH score 
                # response_qqh_score[0] = -9999.0
                # # if args.channel in ['tt','em']:
                # response_qqh_score[0] = response[2]
                # # elif args.channel in ['mt','et']:
                # #     response_qqh_score[0] = response[1]


                # Fill branches
                branch_max_score.Fill()
                branch_max_index.Fill()
                # branch_ggh_score.Fill()
                # branch_qqh_score.Fill()

            else:
                response_max_score[0] = -9999.0
                response_max_index[0] = -9999.0
                # response_ggh_score[0] = -9999.0
                # response_qqh_score[0] = -9999.0

                # Fill branches
                branch_max_score.Fill()
                branch_max_index.Fill()
                # branch_ggh_score.Fill()
                # branch_qqh_score.Fill()

        # Write everything to file
        file_.Write("ntuple",ROOT.TObject.kWriteDelete)
        file_.Close()


if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config_training)
    file_names = load_files(args.input)
    main(args, config, file_names)
