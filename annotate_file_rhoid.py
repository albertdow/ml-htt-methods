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
        "--alt_tree", default="train_ntuple", help="Name of trees in the directories.")
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

def setupVarsForRhoID(numTaus):
    pass

def useBranchNames(channel):
    if channel in ["mt","et"]:
        return ["other_2","rho_2","pi_2","a1_2"]
    elif channel == "tt":
        return ["other_1","rho_1","pi_1","a1_1","other_2","rho_2","pi_2","a1_2"]
    else: return None

def main(args, config, file_names):

    # path = "/vols/cms/mhh18/Offline/output/SM/2016_trees/"
    path = "/vols/cms/akd116/Offline/output/SM/2019/test_trees/"

    # Load model
    with open('{}/multiLargeSample_xgb_clf_NewRhoMass_maxdepth5_lr0p05_fNames.pkl'
            .format(args.model_folder), 'r') as f:
        classifier = pickle.load(f)

    for sample in file_names:
        print(sample)
        # if not os.path.exists("{}/{}_{}_{}.root".format(path, sample, args.channel, args.era)):
        if not os.path.exists("{}/{}".format(path, sample)):
            logger.fatal("Input file %s does not exist.", sample)
            raise Exception

        logger.debug("Following mapping of classes to class numbers is used.")
        for i, class_ in enumerate(config["classes"]):
            logger.debug("%s : %s", i, class_)


        # Open input file
        file_ = ROOT.TFile("{}/{}".format(path, sample), "UPDATE")
        if file_ == None:
            logger.fatal("File %s is not existent.", sample)
            raise Exception

        tree = file_.Get(args.tree)
        alt_tree = file_.Get(args.alt_tree)

        # Book branches for annotation
        values = []

        # don't actually need this bit since all features are custom
        # for variable in config["variables"]:
        #     if variable not in ["dR_custom",]:
        #         tree.SetBranchAddress(variable[:-2], values[-1])

        # Prepare branches with the predictions of the classifier
        response_other = array("f", [-9999])
        branch_other = tree.Branch("{}_other".format(
            args.tag), response_other, "{}_other/F".format(
                args.tag))

        response_rho = array("f", [-9999])
        branch_rho = tree.Branch("{}_rho".format(
            args.tag), response_rho, "{}_rho/F".format(
                args.tag))

        response_pi = array("f", [-9999])
        branch_pi = tree.Branch("{}_pi".format(
            args.tag), response_pi, "{}_pi/F".format(
                args.tag))

        response_a1 = array("f", [-9999])
        branch_a1 = tree.Branch("{}_a1".format(
            args.tag), response_a1, "{}_a1/F".format(
                args.tag))

        # Run the event loop
        for i_event in range(alt_tree.GetEntries()):
            alt_tree.GetEntry(i_event)

            if args.channel in ["mt","et"]:
                # only one tau to apply rho ID to (second ie. _2)
                Egamma1_tau = float(getattr(alt_tree,"Egamma1_2")) \
                        / float(getattr(alt_tree,"E_2"))
                Egamma2_tau = float(getattr(alt_tree,"Egamma2_2")) \
                        / float(getattr(alt_tree,"E_2"))
                Egamma3_tau = float(getattr(alt_tree,"Egamma3_2")) \
                        / float(getattr(alt_tree,"E_2"))
                Epi_tau = float(getattr(alt_tree,"Epi_2")) \
                        / float(getattr(alt_tree,"E_2"))
                rho_dEta_tau = float(getattr(alt_tree,"rho_dEta_2")) \
                        * float(getattr(alt_tree,"E_2"))
                rho_dphi_tau = float(getattr(alt_tree,"rho_dphi_2")) \
                        * float(getattr(alt_tree,"E_2"))
                gammas_dEta_tau = float(getattr(alt_tree,"gammas_dEta_2")) \
                        * float(getattr(alt_tree,"E_2"))
                gammas_dR_tau = np.sqrt(float(getattr(alt_tree,"gammas_dEta_2"))**2 \
                        + float(getattr(alt_tree,"gammas_dphi_2"))**2)
                DeltaR2WRTtau_tau = float(getattr(alt_tree,"DeltaR2WRTtau_2")) \
                        * float(getattr(alt_tree,"E_2"))**2
                eta = float(getattr(alt_tree,"eta_2"))
                pt = float(getattr(alt_tree,"pt_2"))
                Epi0 = float(getattr(alt_tree,"Epi0_2"))
                Epi = float(getattr(alt_tree,"Epi_2"))
                # are the next three features actually needed? 
                # we already have them above basically
                rho_dEta = float(getattr(alt_tree,"rho_dEta_2"))
                rho_dphi = float(getattr(alt_tree,"rho_dphi_2"))
                gammas_dEta = float(getattr(alt_tree,"gammas_dEta_2"))
                #
                tau_decay_mode = float(getattr(alt_tree,"tau_decay_mode_2"))
                Mrho = float(getattr(alt_tree,"Mrho_2"))
                Mpi0 = float(getattr(alt_tree,"Mpi0_2"))
                DeltaR2WRTtau = float(getattr(alt_tree,"DeltaR2WRTtau_2"))
                Mpi0_TwoHighGammas = float(getattr(alt_tree,"Mpi0_TwoHighGammas_2"))
                Mpi0_ThreeHighGammas = float(getattr(alt_tree,"Mpi0_ThreeHighGammas_2"))
                Mpi0_FourHighGammas = float(getattr(alt_tree,"Mpi0_FourHighGammas_2"))
                Mrho_OneHighGammas = float(getattr(alt_tree,"Mrho_OneHighGammas_2"))
                Mrho_TwoHighGammas = float(getattr(alt_tree,"Mrho_TwoHighGammas_2"))
                Mrho_ThreeHighGammas = float(getattr(alt_tree,"Mrho_ThreeHighGammas_2"))
                Mrho_subleadingGamma = float(getattr(alt_tree,"Mrho_subleadingGamma_2"))

            # then define in additional_vars
            additional_vars = [
                    Egamma1_tau,
                    Egamma2_tau,
                    Egamma3_tau,
                    Epi_tau,
                    rho_dEta_tau,
                    rho_dphi_tau,
                    gammas_dEta_tau,
                    gammas_dR_tau,
                    DeltaR2WRTtau_tau,
                    eta,
                    pt,
                    Epi0,
                    Epi,
                    rho_dEta,
                    rho_dphi,
                    gammas_dEta,
                    tau_decay_mode,
                    Mrho,
                    Mpi0,
                    DeltaR2WRTtau,
                    Mpi0_TwoHighGammas,
                    Mpi0_ThreeHighGammas,
                    Mpi0_FourHighGammas,
                    Mrho_OneHighGammas,
                    Mrho_TwoHighGammas,
                    Mrho_ThreeHighGammas,
                    Mrho_subleadingGamma,
                    ]

            for index,val in enumerate(additional_vars):
                values.append(val)
            values_stacked = np.hstack(values).reshape(1, len(values))
            response = classifier.predict_proba(values_stacked,
                    ntree_limit=classifier.best_iteration+1)
            response = np.squeeze(response)

            # Take scores in order as well
            response_other[0] = -9999.0
            response_other[0] = response[0]

            response_rho[0]   = -9999.0
            response_rho[0]   = response[1]

            response_pi[0]    = -9999.0
            response_pi[0]    = response[2]

            response_a1[0]    = -9999.0
            response_a1[0]    = response[3]

            # Fill branches
            branch_other.Fill()
            branch_rho.Fill()
            branch_pi.Fill()
            branch_a1.Fill()


        # Write everything to file
        file_.Write("ntuple",ROOT.TObject.kWriteDelete)
        file_.Close()


if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config_training)
    file_names = load_files(args.input)
    main(args, config, file_names)
