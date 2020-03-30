## HiggsTauTau CP ML methods
Repository for creating training datasets from ROOT files 
and then train on with an algorithm of choice (XGBoost, keras, sklearn etc) / or implement others.

### Git instructions

`git clone -b fullrun2 git@github.com:albertdow/ml-htt-methods.git`

### Train
TO DO

### Annotate

To annotate files ROOT files with trained (XBG) model follow these steps:

- Open annotate_file_inc_16.py, annotate_file_inc_17.py and annotate_file_inc_18.py, 
search for path and replace with your directory where the ROOT files are stored (ie. /vols/cms/)

- Open batch_annotate_inc_16.sh, batch_annotate_inc_17.sh and batch_annotate_inc_18.sh
and change `cd` and `source` lines to your own CMSSW repo. Also check model_folder 
is the one you want to use, and config is the corresponding one. 
    `--model_folder <path_to_model_folder>`
    `--config-training <path_to_model_folder>/tt_config_<era>.yaml`
For example if want to use training with all variables (2016 training):
    `--model_folder data_tauspinner_12Mar2020_2016/`
    `--config-training data_tauspinner_12Mar2020_2016/tt_config_2016.yaml`
If want to use training without pT2,dijetpt variables (2016):
    `--model_folder data_tauspinner_12Mar2020_2016_NopT2dijetpT/`
    `--config-training data_tauspinner_12Mar2020_2016_NopT2dijetpT//tt_config_2016.yaml`
    

- Run following commands
    
    `for era in 2016 2017 2018 ; do mkdir filelist/tmp_${era}/ && mkdir filelist/tmp_${era}/tt/; done`

    `for era in 2016 2017 2018 ; do cd filelist/tmp_${era}/tt/ && split -l 1 -a 4 --numeric-suffixes ../../full_tt_${era}.txt && cd ../../../; done`

    `mkdir err && mkdir out`

At this stage the user ready to run:

- If you want to test on one job on the batch do:

    `qsub -e err/ -o out/ -cwd -V -q hep.q -t 210-210:1 batch_annotate_inc_16.sh`

- If you want to submit all years for all systematics -- data + embedding first (long queue), 
then MC + systematics (3 hour queue):

    2016:

    `qsub -e err/ -o out/ -cwd -V -l h_rt=10:0:0 -q hep.q -t 1-209:1 batch_annotate_inc_16.sh; done`

    `qsub -e err/ -o out/ -cwd -V -l h_rt=3:0:0 -q hep.q -t 210-2665:1 batch_annotate_inc_16.sh; done`

    2017:

    `qsub -e err/ -o out/ -cwd -V -l h_rt=10:0:0 -q hep.q -t 1-195:1 batch_annotate_inc_17.sh; done`

    `qsub -e err/ -o out/ -cwd -V -l h_rt=3:0:0 -q hep.q -t 196-2121:1 batch_annotate_inc_17.sh; done`

    2018:

    `qsub -e err/ -o out/ -cwd -V -l h_rt=10:0:0 -q hep.q -t 1-526:1 batch_annotate_inc_18.sh; done`

    `qsub -e err/ -o out/ -cwd -V -l h_rt=3:0:0 -q hep.q -t 527-3592:1 batch_annotate_inc_18.sh; done`

to split large jobs up more

- Run following commands (not this example is only for 2017 running only tauF and the filtered VBF signal samples)

    `for era in 2016 2017 2018 ; do mkdir filelist/tmp_${era}_split/ && mkdir filelist/tmp_${era}_split/tt/; done`

    `for era in 2016 2017 2018 ; do cd filelist/tmp_${era}_split/tt/ && python ../../../geteratejobs.py --filelist=../../full_tt_${era}.txt --dir=/vols/cms/dw515/Offline/output/SM/Jan24_${era}_ttonly/ && cd ../../../; done`

    2016:

    `qsub -e err/ -o out/ -cwd -V -l h_rt=3:0:0 -q hep.q -t 1-3001:1 batch_annotate_split_16.sh`

    2017:


    `qsub -e err/ -o out/ -cwd -V -l h_rt=3:0:0 -q hep.q -t 1-2456:1 batch_annotate_split_17.sh`

    2018:


    `qsub -e err/ -o out/ -cwd -V -l h_rt=3:0:0 -q hep.q -t 1-2289:1 batch_annotate_split_18.sh`


