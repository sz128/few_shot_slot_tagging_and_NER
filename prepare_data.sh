#!/bin/bash

python3 data_utils/download.py

raw_data_path=data/ACL2020data

mkdir -p data/xval_ner_shot_1_out_{1,2,3,4}
mkdir -p data/xval_ner_shot_5_out_{1,2,3,4}
mkdir -p data/xval_snips_shot_1_out_{1,2,3,4,5,6,7}
mkdir -p data/xval_snips_shot_5_out_{1,2,3,4,5,6,7}

for n in 1 2 3 4
do
  for tag in train valid test
  do
    cp ${raw_data_path}/xval_ner/ner_${tag}_${n}.json data/xval_ner_shot_1_out_${n}/${tag}.json
    cp ${raw_data_path}/xval_ner_shot_5/ner-${tag}-${n}-shot-5.json data/xval_ner_shot_5_out_${n}/${tag}.json
  done
done

for n in 1 2 3 4 5 6 7
do
  for tag in train valid test
  do
    cp ${raw_data_path}/xval_snips/snips_${tag}_${n}.json data/xval_snips_shot_1_out_${n}/${tag}.json
    cp ${raw_data_path}/xval_snips_shot_5/snips-${tag}-${n}-shot-5.json data/xval_snips_shot_5_out_${n}/${tag}.json
  done
done

### get output vocab
for n in 1 2 3 4
do
  for k in 1 5
  do
    python3 data_utils/get_vocab.py --bio_files data/xval_ner_shot_${k}_out_${n}/{train,valid,test}.json --slot_vocab data/xval_ner_shot_${k}_out_${n}/vocab.slot --intent_vocab data/xval_ner_shot_${k}_out_${n}/vocab.intent --multi_task_vocab_sharing_type full
  done
done
for n in 1 2 3 4 5 6 7
do
  for k in 1 5
  do
    python3 data_utils/get_vocab.py --bio_files data/xval_snips_shot_${k}_out_${n}/{train,valid,test}.json --slot_vocab data/xval_snips_shot_${k}_out_${n}/vocab.slot --intent_vocab data/xval_snips_shot_${k}_out_${n}/vocab.intent --multi_task_vocab_sharing_type full
  done
done

### slot descriptions
for n in 1 2 3 4
do
  for k in 1 5
  do
    ln -s ../ner_slot_description.txt data/xval_ner_shot_${k}_out_${n}/slot_description
  done
done

for n in 1 2 3 4 5 6 7
do
  for k in 1 5
  do
    ln -s ../snips_slot_description.txt data/xval_snips_shot_${k}_out_${n}/slot_description
  done
done

