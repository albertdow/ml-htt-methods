#!/bin/bash

dir=/vols/build/cms/dw515/test_crash/CMSSW_8_0_25/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTau/

cd /vols/build/cms/dw515/JER/CMSSW_10_2_18/src/
export SCRAM_ARCH=slc6_amd64_gcc481
eval `scramv1 runtime -sh`
source $dir/scripts/setup_libs.sh
# for keras others use
# conda activate mlFramework
# source ~/.profile

cd $dir/ml-htt-methods
ulimit -c 0
inputNumber=$SGE_TASK_ID


# tt 
python annotate_file_split_16.py filelist/tmp_2016_split/tt/x${inputNumber} IC_15Mar2020 --model_folder ./data_tauspinner_12Mar2020_2016_NopT2dijetpT/ --training tauspinner --era 2016 --channel tt --config-training data_tauspinner_12Mar2020_2016_NopT2dijetpT/tt_2016_config_inc.yaml &> filelist/tmp_2016_split/tt/${inputNumber}.log 

