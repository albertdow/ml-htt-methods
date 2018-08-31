#!/bin/bash

cd /vols/build/cms/akd116/MLStudies/CMSSW_9_4_0/src/
export SCRAM_ARCH=slc6_amd64_gcc481
eval `scramv1 runtime -sh`
source /vols/build/cms/akd116/MLStudies/latest/CMSSW_8_0_25/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTau/scripts/setup_libs.sh

cd /vols/build/cms/akd116/MLStudies/local
ulimit -c 0
inputNumber=$(printf "%04d\n" $((SGE_TASK_ID-1)))

# python annotate_file_lowMjj.py filelist/tmp/full_tt_cpsm_files_with_systematics_${inputNumber} IC_lowMjj_Jun4 --model_folder ./Jun4_training_tt_lowMjj --training powheg --channel tt --config-training tt_xgb_training_config_Jun4.yaml &> filelist/tmp/tt/full_tt_cpsm_files_with_systematics_${inputNumber}_lowMjj.log
# python annotate_file_lowMjj.py filelist/tmp/Jun22/tt/${inputNumber} IC_lowMjj_July05_3 --model_folder ./July5_training_tt_lowMjj --training powheg --channel tt --config-training tt_xgb_training_config_July4.yaml &> filelist/tmp/Jun22/tt/${inputNumber}.log
# python annotate_file_lowMjj.py filelist/tmp/Aug14/tt/x${inputNumber} IC_lowMjj_July05_3 --model_folder ./July5_training_tt_lowMjj --training powheg --channel tt --config-training tt_xgb_training_config_July4.yaml &> filelist/tmp/Aug14/tt/${inputNumber}.log

# python annotate_file_lowMjj.py filelist/tmp/mt/full_mt_cpsm_files_with_systematics_${inputNumber} IC_lowMjj_Jun4 --model_folder ./Jun4_training_mt_lowMjj --training powheg --channel mt --config-training mt_xgb_training_lowMjj_config_Jun4.yaml &> filelist/tmp/mt/full_mt_cpsm_files_with_systematics_${inputNumber}_lowMjj.log
# python annotate_file_lowMjj.py filelist/tmp/resubmit_${inputNumber} IC_lowMjj_21May --model_folder ./mt_training_lowMjj_21May_full --training powheg --channel mt --config-training mt_xgb_training_lowMjj_config_May21.yaml &> filelist/tmp/mt/resubmit_${inputNumber}.log
# python annotate_file_lowMjj.py filelist/tmp/Jun22/mt/${inputNumber} IC_lowMjj_July05_3 --model_folder ./July5_training_mt_lowMjj --training powheg --channel mt --config-training mt_xgb_training_lowMjj_config_July4.yaml &> filelist/tmp/Jun22/mt/${inputNumber}.log
# python annotate_file_lowMjj.py filelist/tmp/Aug14/mt/x${inputNumber} IC_lowMjj_July05_3 --model_folder ./July5_training_mt_lowMjj --training powheg --channel mt --config-training mt_xgb_training_lowMjj_config_July4.yaml &> filelist/tmp/Jun22/mt/${inputNumber}.log

# python annotate_file_lowMjj.py filelist/tmp/et/full_et_cpsm_files_with_systematics_${inputNumber} IC_lowMjj_21May --model_folder ./et_training_lowMjj_21May_full --training powheg --channel et --config-training mt_xgb_training_lowMjj_config_May21.yaml &> filelist/tmp/et/full_et_cpsm_files_with_systematics_${inputNumber}.log
# python annotate_file_lowMjj.py filelist/tmp/resubmit_${inputNumber} IC_lowMjj_21May --model_folder ./et_training_lowMjj_21May_full --training powheg --channel et --config-training mt_xgb_training_lowMjj_config_May21.yaml &> filelist/tmp/et/resubmit_${inputNumber}.log
# python annotate_file_lowMjj.py filelist/tmp/et/full_et_cpsm_files_with_systematics_${inputNumber} IC_lowMjj_Jun4 --model_folder ./Jun4_training_et_lowMjj --training powheg --channel et --config-training mt_xgb_training_lowMjj_config_Jun4.yaml &> filelist/tmp/et/full_et_cpsm_files_with_systematics_${inputNumber}_lowMjj.log
# python annotate_file_lowMjj.py filelist/tmp/Jun22/et/${inputNumber} IC_lowMjj_July05_3 --model_folder ./July5_training_et_lowMjj --training powheg --channel et --config-training mt_xgb_training_lowMjj_config_July4.yaml &> filelist/tmp/Jun22/et/${inputNumber}.log
python annotate_file_lowMjj.py filelist/tmp/Aug14/et/x${inputNumber} IC_lowMjj_July05_3 --model_folder ./July5_training_et_lowMjj --training powheg --channel et --config-training mt_xgb_training_lowMjj_config_July4.yaml &> filelist/tmp/Aug14/et/${inputNumber}.log

# python annotate_file_lowMjj.py filelist/tmp/em/full_em_cpsm_files_with_systematics_${inputNumber} IC_lowMjj_21May_3 --model_folder ./em_training_lowMjj_21May_full_2 --training powheg --channel em --config-training em_xgb_training_lowMjj_config_May21.yaml &> filelist/tmp/em/full_em_cpsm_files_with_systematics_${inputNumber}.log
# python annotate_file_lowMjj.py filelist/tmp/resubmit_${inputNumber} IC_lowMjj_21May_3 --model_folder ./em_training_lowMjj_21May_full_2 --training powheg --channel em --config-training em_xgb_training_lowMjj_config_May21.yaml &> filelist/tmp/em/resubmit_${inputNumber}.log
# python annotate_file_lowMjj.py filelist/tmp/em/full_em_cpsm_files_with_systematics_${inputNumber} IC_lowMjj_Jun15 --model_folder ./Jun15_training_em_lowMjj --training powheg --channel em --config-training em_xgb_training_lowMjj_config_Jun4.yaml &> filelist/tmp/em/full_em_cpsm_files_with_systematics_${inputNumber}_lowMjj.log
# python annotate_file_lowMjj.py filelist/tmp/Jun22/em/${inputNumber} IC_lowMjj_July05_4 --model_folder ./July5_training_em_lowMjj --training powheg --channel em --config-training em_xgb_training_lowMjj_config_July4.yaml &> filelist/tmp/Jun22/em/${inputNumber}.log
# python annotate_file_lowMjj.py filelist/tmp/Aug14/em/x${inputNumber} IC_lowMjj_July05_4 --model_folder ./July5_training_em_lowMjj --training powheg --channel em --config-training em_xgb_training_lowMjj_config_July4.yaml &> filelist/tmp/Aug14/em/${inputNumber}.log

