## HiggsTauTau CP ML methods
Repository for creating training datasets from ROOT files 
and then train on with an algorithm of choice (XGBoost, keras, sklearn etc) / or implement others.

### Train
TO DO

### Annotate

To annotate files ROOT files with trained (XBG) model follow these steps:

- Open annotate_file_inc_16.py, annotate_file_inc_17.py and annotate_file_inc_18.py, 
search for path and replace with your directory where the ROOT files are stored (ie. /vols/cms/)

- Open batch_annotate_inc_16.sh, batch_annotate_inc_17.sh and batch_annotate_inc_18.sh
and change `cd` and `source` lines to your own CMSSW repo. Also check model_folder 
is the one you want to use, and config is the corresponding one.

- Run following commands
    
    `for era in 2016 2017 2018 ; do mkdir filelist/tmp_${era}/ && mkdir filelist/tmp_${era}/tt/; done`

    `for era in 2016 2017 2018 ; do cd filelist/tmp_${era}/tt/ && split -l 1 -a 4 --numeric-suffixes ../../full_tt_2016.txt && cd ../../../; done`

    `mkdir err && mkdir out`

At this stage the user ready to run:

If you want to test on one job on the batch do:
    `qsub -e err/ -o out/ -cwd -V -q hep.q -t 1-1:1 batch_annotate_inc_16.sh; done`

If you want to submit all years for all systematics:

    `for era in 16 17 18 ; do for jid in $(ls -l filelist/tmp_20${era}/tt/x* | tail -n 1 | awk '{print $9}' | tr -d -c 0-9); do qsub -e err/ -o out/ -cwd -V -q hep.q -t 1-${jid}:1 batch_annotate_inc_${era}.sh; done`

