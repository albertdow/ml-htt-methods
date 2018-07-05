#!/bin/bash

cd /vols/build/cms/akd116/MLStudies/CMSSW_9_4_0/src/
export SCRAM_ARCH=slc6_amd64_gcc481
eval `scramv1 runtime -sh`
source /vols/build/cms/akd116/MLStudies/latest/CMSSW_8_0_25/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTau/scripts/setup_libs.sh

cd /vols/build/cms/akd116/MLStudies/local
ulimit -c 0
inputNumber=$(printf "%04d\n" $((SGE_TASK_ID-1)))

# python annotate_file.py filelist/tmp/full_tt_cpsm_files_with_systematics_${inputNumber} IC_highMjj_21May_3 --model_folder ./tt_training_highMjj_21May_full --training JHU --channel tt --config-training tt_xgb_training_config_May21.yaml &> filelist/tmp/tt/full_tt_cpsm_files_with_systematics_${inputNumber}.log
# python annotate_file.py filelist/tmp/full_tt_cpsm_files_with_systematics_${inputNumber} IC_highMjj_Jun28 --model_folder ./Jun28_training_tt_highMjj --training JHU --channel tt --config-training tt_xgb_training_config_Jun28.yaml &> filelist/tmp/tt/full_tt_cpsm_files_with_systematics_${inputNumber}.log
# python annotate_file.py filelist/tmp/Jun22/tt/${inputNumber} IC_highMjj_July05_1 --model_folder ./July5_training_tt_highMjj --training JHU --channel tt --config-training tt_xgb_training_config_July4.yaml &> filelist/tmp/Jun22/tt/${inputNumber}.log

# python annotate_file.py filelist/tmp/mt/full_mt_cpsm_files_with_systematics_${inputNumber} IC_highMjj_15May_2_logloss --model_folder ./mt_training_15May_fulljetvars --training JHU --channel mt --config-training mt_xgb_training_config_new_fulljetvars.yaml &> filelist/tmp/mt/full_mt_cpsm_files_with_systematics_${inputNumber}.log
# python annotate_file.py filelist/tmp/mt/full_mt_cpsm_files_with_systematics_${inputNumber} IC_highMjj_Jun4 --model_folder ./Jun4_training_mt_highMjj --training JHU --channel mt --config-training mt_xgb_training_config_Jun4.yaml &> filelist/tmp/mt/full_mt_cpsm_files_with_systematics_${inputNumber}.log
# python annotate_file.py filelist/tmp/resubmit_${inputNumber} IC_highMjj_21May --model_folder ./mt_training_highMjj_21May_full --training JHU --channel mt --config-training mt_xgb_training_config_May21.yaml &> filelist/tmp/mt/resubmit_${inputNumber}.log
# python annotate_file.py filelist/tmp/Jun22/mt/${inputNumber} IC_highMjj_July05_1 --model_folder ./July5_training_mt_highMjj --training JHU --channel mt --config-training mt_xgb_training_config_July4.yaml &> filelist/tmp/Jun22/mt/${inputNumber}.log

# python annotate_file.py filelist/tmp/et/full_et_cpsm_files_with_systematics_${inputNumber} IC_highMjj_15May_2_logloss --model_folder ./et_training_15May_fulljetvars --training JHU --channel et --config-training mt_xgb_training_config_new_fulljetvars.yaml &> filelist/tmp/et/full_et_cpsm_files_with_systematics_${inputNumber}.log
# python annotate_file.py filelist/tmp/et/full_et_cpsm_files_with_systematics_${inputNumber} IC_highMjj_21May --model_folder ./et_training_highMjj_21May_full --training JHU --channel et --config-training mt_xgb_training_config_May21.yaml &> filelist/tmp/et/full_et_cpsm_files_with_systematics_${inputNumber}.log
# python annotate_file.py filelist/tmp/resubmit_${inputNumber} IC_highMjj_21May --model_folder ./et_training_highMjj_21May_full --training JHU --channel et --config-training mt_xgb_training_config_May21.yaml &> filelist/tmp/et_high/resubmit_${inputNumber}.log
# python annotate_file.py filelist/tmp/et/full_et_cpsm_files_with_systematics_${inputNumber} IC_highMjj_Jun4 --model_folder ./Jun4_training_et_highMjj --training JHU --channel et --config-training mt_xgb_training_config_Jun4.yaml &> filelist/tmp/et/full_et_cpsm_files_with_systematics_${inputNumber}.log
# python annotate_file.py filelist/tmp/Jun22/et/${inputNumber} IC_highMjj_July05_1 --model_folder ./July5_training_et_highMjj --training JHU --channel et --config-training mt_xgb_training_config_July4.yaml &> filelist/tmp/Jun22/et/${inputNumber}.log

# python annotate_file.py filelist/tmp/em/full_em_cpsm_files_with_systematics_${inputNumber} IC_highMjj_14May_logloss --model_folder ./em_training_15May_mjj_jdeta --training JHU --channel em --config-training em_xgb_training_config_new_jetvars.yaml &> filelist/tmp/em/full_em_cpsm_files_with_systematics_${inputNumber}.log
# python annotate_file.py filelist/tmp/em/full_em_cpsm_files_with_systematics_${inputNumber} IC_highMjj_21May --model_folder ./em_training_highMjj_21May_full --training JHU --channel em --config-training em_xgb_training_config_May21.yaml &> filelist/tmp/em/full_em_cpsm_files_with_systematics_${inputNumber}.log
# python annotate_file.py filelist/tmp/resubmit_${inputNumber} IC_highMjj_21May --model_folder ./em_training_highMjj_21May_full --training JHU --channel em --config-training em_xgb_training_config_May21.yaml &> filelist/tmp/em_high/resubmit_${inputNumber}.log
# python annotate_file.py filelist/tmp/em/full_em_cpsm_files_with_systematics_${inputNumber} IC_highMjj_Jun4 --model_folder ./Jun4_training_em_highMjj --training JHU --channel em --config-training em_xgb_training_config_Jun4.yaml &> filelist/tmp/em/full_em_cpsm_files_with_systematics_${inputNumber}.log
python annotate_file.py filelist/tmp/Jun22/em/${inputNumber} IC_highMjj_July05_1 --model_folder ./July5_training_em_highMjj --training JHU --channel em --config-training em_xgb_training_config_July4.yaml &> filelist/tmp/Jun22/em/${inputNumber}.log

