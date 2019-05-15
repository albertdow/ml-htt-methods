# keras for local application
for loop in {0..70}
do 
    inputNumber=$(printf "%04d\n" ${loop})
    nohup python annotate_file_rhoid.py filelist/tmp_nominal/mt/x${inputNumber} IC_test --model_folder rhoid_test --config-training rhoid_test/rhoid_config.yaml &> logs/${inputNumber}.log &
done
