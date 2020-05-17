## Few-Shot Slot Tagging and NER
Few-shot slot tagging and NER.

## Data
Download the SNIPS and NER [dataset](https://atmahou.github.io/attachments/ACL2020data.zip) formatted as episodes.
```console
❱❱❱ bash prepare_data.sh
```

Data statistic:
```console 
❱❱❱ python3 data_utils/data_statistic.py --data_path data/xval_ner_shot_1_out_1
```
