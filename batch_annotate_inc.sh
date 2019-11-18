#!/bin/bash

# cd /vols/build/cms/akd116/MLStudies/CMSSW_9_4_0/src/
# cd /vols/build/cms/akd116/newest/CMSSW_10_2_9/src/
cd /vols/build/cms/mhh18/CMSSW_10_2_16/src/
export SCRAM_ARCH=slc6_amd64_gcc481
eval `scramv1 runtime -sh`
source /vols/build/cms/mhh18/CMSSW_8_0_25/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTau/scripts/setup_libs.sh
# for keras others use
# conda activate mlFramework
# source ~/.profile

cd /vols/build/cms/mhh18/train_HiggsVsBkgs/ml-htt-methods/
ulimit -c 0
inputNumber=$(printf "%04d\n" $((SGE_TASK_ID-1)))

# python annotate_file_inc.py filelist/tmp/2017/tt/x${inputNumber} IC_Dec10 --model_folder ./Dec10_training_tt_inc --training powheg --era 2017 --channel tt --config-training tt_xgb_training_2017_config_Dec10.yaml &> filelist/tmp/2017/tt/x${inputNumber}.log

## tt ones
# 2017
# python annotate_file_inc.py filelist/tmp_2017/tt/x${inputNumber} IC_Nov13_tauspinner --model_folder ./data_tauspinner_2017/ --training tauspinner --era 2017 --channel tt --config-training data_tauspinner_2017/tt_2017_config_inc.yaml &> filelist/tmp_2017/tt/${inputNumber}.log 

# 2018
python annotate_file_inc.py filelist/tmp_2018/tt/x${inputNumber} IC_Nov13_tauspinner --model_folder ./data_tauspinner_2018/ --training tauspinner --era 2018 --channel tt --config-training data_tauspinner_2018/tt_2018_config_inc.yaml &> filelist/tmp_2018/tt/${inputNumber}.log 

# keras
# python annotate_file_inc_keras.py filelist/tmp_full/tt/x${inputNumber} IC_keras_testsm --model_folder IC_tt_keras_inc/ --training powheg --era 2016 --channel tt --config-training IC_tt_keras_inc/tt_keras_training_2016_config_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log

# keras others
# python annotate_file_inc_keras_others.py filelist/tmp_full/tt/x${inputNumber} IC_Vienna_fix --model_folder vienna_tt_keras_inc/ --training powheg --era 2016 --channel tt --config-training vienna_tt_keras_inc/tt_keras_training_2016_config_vienna_inc.yaml &> filelist/tmp_full/tt/${inputNumber}.log

## mt 
# python annotate_file_inc.py filelist/tmp_full/mt/x${inputNumber} IC_Mar26_fix2 --model_folder ./Mar26_training_mt_inc_2016 --training powheg --era 2016 --channel mt --config-training mt_xgb_training_2016_config_Mar26_inc.yaml &> filelist/tmp_full/mt/${inputNumber}.log 

## et
# python annotate_file_inc.py filelist/tmp_full/et/x${inputNumber} IC_Apr02 --model_folder ./Apr02_training_et_inc_2016 --training powheg --era 2016 --channel et --config-training mt_xgb_training_2016_config_Mar26_inc.yaml &> filelist/tmp_full/et/${inputNumber}.log 

