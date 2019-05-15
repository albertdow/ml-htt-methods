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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-training",
        default="mt_xgb_training_config.yaml",
        help="Path to training config file")
    parser.add_argument(
        "input", help="Path to input file, where response will be added.")
    parser.add_argument(
        "tag", help="Tag to be used as prefix of the annotation.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default="multi_fold0_cpsm_mt_JHU_xgb.pkl",
        help=
        "Keras models to be used for the annotation. Note that these have to be booked in the reversed order [fold1*, fold0*], so that the training is independent from the application."
    )
    parser.add_argument(
        "--tree", default="ntuple", help="Name of trees in the directories.")
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

    path = "/vols/cms/mhh18/Offline/output/SM/2016_trees/"

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

        # Load model
        with open('{}/multiLargeSample_xgb_clf.pkl'
                .format(args.model_folder), 'r') as f:
            classifier = pickle.load(f)

        # Open input file
        file_ = ROOT.TFile("{}/{}".format(path, sample), "UPDATE")
        if file_ == None:
            logger.fatal("File %s is not existent.", sample)
            raise Exception

        tree = file_.Get(args.tree)

        # Book branches for annotation
        values = []
        for variable in config["variables"]:
            if variable in ["dijetpt","eta_h","IC_binary_test_4_score","IC_binary_test_4_index",]:
                values.append(array("f", [-9999]))
            if variable in ["eta_1","eta_2","jdeta","jpt_1","jpt_2","m_sv","m_vis","met","jeta_1","jeta_2","mt_tot","mt_sv",
                    "met_dphi_1","met_dphi_2","mjj","mt_1","mt_2","mt_lep","pt_1","pt_2","pt_h","pt_tt","pt_vis","pzeta","dR",]:
                values.append(array("d", [-9999]))
            if variable in ["n_jets","n_bjets","opp_sides",]:
                values.append(array("I", [0]))
            if variable not in ["dR_custom",]:
                tree.SetBranchAddress(variable, values[-1])

        # Prepare branches with the predictions of the classifier
        response_other_score = array("f", [-9999])
        branch_other_score = tree.Branch("{}_other_score".format(
            args.tag), response_other_score, "{}_other_score/F".format(
                args.tag))

        response_rho_score = array("f", [-9999])
        branch_rho_score = tree.Branch("{}_rho_score".format(
            args.tag), response_rho_score, "{}_rho_score/F".format(
                args.tag))

        response_pion_score = array("f", [-9999])
        branch_pion_score = tree.Branch("{}_pion_score".format(
            args.tag), response_pion_score, "{}_pion_score/F".format(
                args.tag))

        response_a1_score = array("f", [-9999])
        branch_a1_score = tree.Branch("{}_a1_score".format(
            args.tag), response_a1_score, "{}_a1_score/F".format(
                args.tag))

        # Run the event loop
        for i_event in range(tree.GetEntries()):
            tree.GetEntry(i_event)

            # add here any variables that are to be calculated on the fly
            # eg. dR between two leading leptons --> need dphi_custom then
            # make dR_custom
            # dphi_custom = np.arccos(1-float(getattr(tree,"mt_lep"))**2 \
            #        /(2.*float(getattr(tree,"pt_1"))*float(getattr(tree,"pt_2"))))
            # dR_custom = np.sqrt((float(getattr(tree,"eta_1")) \
            #         -float(getattr(tree,"eta_2")))**2 + dphi_custom**2)

            # then define in additional_vars
            additional_vars = [
                    # dR_custom,
                    ]
            if len(values) < len(config["variables"]):
                values.extend(additional_vars)
            else:
                for index,val in enumerate(additional_vars):
                    ind = len(additional_vars)-index
                    values[-ind] = val
            values_stacked = np.hstack(values).reshape(1, len(values))
            response = classifier.predict_proba(values_stacked,
                    ntree_limit=classifier.best_iteration+1)
            response = np.squeeze(response)

            # Take scores in order as well
            response_other_score[0] = -9999.0
            response_other_score[0] = response[0]

            response_rho_score[0] = -9999.0
            response_rho_score[0] = response[1]

            response_pion_score[0] = -9999.0
            response_pion_score[0] = response[2]

            response_a1_score[0] = -9999.0
            response_a1_score[0] = response[3]

            exit()

            # Fill branches
            branch_other_score.Fill()
            branch_rho_score.Fill()
            branch_pion_score.Fill()
            branch_a1_score.Fill()


        # Write everything to file
        file_.Write("ntuple",ROOT.TObject.kWriteDelete)
        file_.Close()


if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config_training)
    file_names = load_files(args.input)
    main(args, config, file_names)
