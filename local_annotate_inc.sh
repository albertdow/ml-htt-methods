# keras for local application
for loop in {0..64}
do 
    inputNumber=$(printf "%04d\n" ${loop})
    nohup python annotate_file_inc_keras.py filelist/tmp_full_grouped/tt/x${inputNumber} IC_keras_sm3 --model_folder IC_tt_keras_inc3/ --training powheg --era 2016 --channel tt --config-training IC_tt_keras_inc3/tt_keras_training_2016_config_inc.yaml &> filelist/tmp_full_grouped/tt/${inputNumber}.log &
done
