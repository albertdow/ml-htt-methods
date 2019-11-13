#!/bin/bash

# cd /vols/build/cms/akd116/MLStudies/CMSSW_9_4_0/src/
# cd /vols/build/cms/akd116/newest/CMSSW_10_2_9/src/
cd /vols/build/cms/akd116/newest/CMSSW_10_2_14/src/
export SCRAM_ARCH=slc6_amd64_gcc481
eval `scramv1 runtime -sh`
source /vols/build/cms/akd116/MLStudies/latest/CMSSW_8_0_25/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTau/scripts/setup_libs.sh
# for keras others use
# conda activate mlFramework
# source ~/.profile

cd /vols/build/cms/akd116/MLStudies/local
ulimit -c 0
inputNumber=$(printf "%04d\n" $((SGE_TASK_ID-1)))

# python annotate_file_inc.py filelist/tmp/2017/tt/x${inputNumber} IC_Dec10 --model_folder ./Dec10_training_tt_inc --training powheg --era 2017 --channel tt --config-training tt_xgb_training_2017_config_Dec10.yaml &> filelist/tmp/2017/tt/x${inputNumber}.log

## tt ones
# python annotate_file_inc.py filelist/tmp/Aug14/tt/x${inputNumber} IC_Feb13_fix1 --model_folder ./Feb13_training_tt_inc_2016 --training powheg --era 2016 --channel tt --config-training tt_xgb_training_2016_config_Feb13_inc.yaml &> filelist/tmp/Aug14/tt/${inputNumber}.log 
# python annotate_file_inc.py filelist/tmp_systs/tt/x${inputNumber} IC_Feb13_fix1 --model_folder ./Feb13_training_tt_inc_2016 --training powheg --era 2016 --channel tt --config-training tt_xgb_training_2016_config_Feb13_inc.yaml &> filelist/tmp/Aug14/tt/${inputNumber}.log 
# python annotate_file_inc.py filelist/tmp_full/tt/x${inputNumber} IC_Feb13_fix1 --model_folder ./Feb13_training_tt_inc_2016 --training powheg --era 2016 --channel tt --config-training tt_xgb_training_2016_config_Feb13_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log 
# python annotate_file_inc.py filelist/tmp_met/tt/x${inputNumber} IC_Feb13_fix1 --model_folder ./Feb13_training_tt_inc_2016 --training powheg --era 2016 --channel tt --config-training tt_xgb_training_2016_config_Feb13_inc.yaml &> filelist/tmp_met/tt/${inputNumber}.log 
# python annotate_file_inc.py filelist/tmp_full/tt/x${inputNumber} IC_Jun13_dR_tauspinner --model_folder ./Jun13_training_tt_2016_bdt_inc_dR_tauspinner --training tauspinner --era 2016 --channel tt --config-training Jun13_training_tt_2016_bdt_inc_dR_tauspinner/tt_xgb_training_2016_config_Jun13_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log 
# python annotate_file_inc.py filelist/tmp_full/tt/x${inputNumber} IC_Jun13_dR_tauspinner_split --model_folder ./Jun13_training_tt_2016_bdt_inc_dR_tauspinner_split --training tauspinner --era 2016 --channel tt --config-training Jun13_training_tt_2016_bdt_inc_dR_tauspinner_split/tt_xgb_training_2016_config_Jun13_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log 


# python annotate_file_inc.py filelist/tmp_full/tt/x${inputNumber} IC_Oct07_tauspinnerPS --model_folder ./data_CPdecay2016_trainOnPS/ --training tauspinner --era 2016 --channel tt --config-training data_CPdecay2016_trainOnPS/tt_xgb_training_2016_config_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log 
# python annotate_file_inc.py filelist/tmp_full/tt/x${inputNumber} IC_Oct07_tauspinnerSM_split --model_folder ./data_CPdecay2016_trainOnSM_splitHiggs/ --training tauspinner --era 2016 --channel tt --config-training data_CPdecay2016_trainOnSM_splitHiggs/tt_xgb_training_2016_config_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log 
# python annotate_file_inc.py filelist/tmp_full/tt/x${inputNumber} IC_Oct07_tauspinnerSM_individualSigWts --model_folder ./data_CPdecay2016_trainOnSM_individualSigWeights/ --training tauspinner --era 2016 --channel tt --config-training data_CPdecay2016_trainOnSM_individualSigWeights/tt_xgb_training_2016_config_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log 
# python annotate_file_inc.py filelist/tmp_full/tt/x${inputNumber} IC_Oct07_tauspinnerSM_DMA1Rho --model_folder ./data_CPdecay2016_SM_splitByDMA1Rho/ --training tauspinner --era 2016 --channel tt --config-training data_CPdecay2016_SM_splitByDMRho/tt_xgb_training_2016_config_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log 
python annotate_file_inc.py filelist/tmp_full/tt/x${inputNumber} IC_Oct22_tauspinnerSM_classic --model_folder ./Oct22_tt_2016_bdt_inc_classicMC/ --training tauspinner --era 2016 --channel tt --config-training Oct22_tt_2016_bdt_inc_classicMC/tt_xgb_training_2016_config_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log 

# keras
# python annotate_file_inc_keras.py filelist/tmp_full/tt/x${inputNumber} IC_keras_testsm --model_folder IC_tt_keras_inc/ --training powheg --era 2016 --channel tt --config-training IC_tt_keras_inc/tt_keras_training_2016_config_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log

# keras others
# python annotate_file_inc_keras_others.py filelist/tmp_full/tt/x${inputNumber} IC_Vienna_fix --model_folder vienna_tt_keras_inc/ --training powheg --era 2016 --channel tt --config-training vienna_tt_keras_inc/tt_keras_training_2016_config_vienna_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log

## mt 
# python annotate_file_inc.py filelist/tmp_full/mt/x${inputNumber} IC_Mar26_fix2 --model_folder ./Mar26_training_mt_inc_2016 --training powheg --era 2016 --channel mt --config-training mt_xgb_training_2016_config_Mar26_inc.yaml &> filelist/tmp_full/mt/${inputNumber}.log 

## et
# python annotate_file_inc.py filelist/tmp_full/et/x${inputNumber} IC_Apr02 --model_folder ./Apr02_training_et_inc_2016 --training powheg --era 2016 --channel et --config-training mt_xgb_training_2016_config_Mar26_inc.yaml &> filelist/tmp_full/et/${inputNumber}.log 

