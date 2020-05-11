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
inputNumber=$(printf "%04d\n" $((SGE_TASK_ID-1)))


# tt 
python annotate_file_inc_17.py filelist/tmp_2017/tt/x${inputNumber} IC_12Mar2020 --model_folder ./data_tauspinner_12Mar2020_2017_NopT2dijetpT/ --training tauspinner --era 2017 --channel tt --config-training data_tauspinner_12Mar2020_2017_NopT2dijetpT/tt_2017_config_inc.yaml &> filelist/tmp_2017/tt/${inputNumber}.log 

# mt 
# python annotate_file_inc_17.py filelist/tmp_2017/mt/x${inputNumber} IC_Nov25_tauspinner --model_folder ./data_tauspinner_2017 --training tauspinner --era 2017 --channel mt --config-training data_tauspinner_2017/mt_config_2017.yaml &> filelist/tmp_2017/mt/${inputNumber}.log 

