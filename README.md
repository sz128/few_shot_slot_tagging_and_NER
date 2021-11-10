# Few-Shot Slot Tagging and NER

This is the PyTorch implementation of the paper:
**Vector Projection Network for Few-shot Slot Tagging in Natural Language Understanding**. **Su Zhu**, Ruisheng Cao, Lu Chen and Kai Yu. 
[[PDF]](https://arxiv.org/pdf/2009.09568.pdf) 

We achieve a new state-of-the-art of few-shot slot tagging on SNIPS and NER datasets.

## Requirements

* pytorch >= 1.4.0
* transformers >= 3.3.1

## Data
Download the SNIPS and NER [dataset](https://atmahou.github.io/attachments/ACL2020data.zip) formatted as episodes.
```console
❱❱❱ bash prepare_data.sh
```

Data statistic:
```console 
❱❱❱ python3 data_utils/data_statistic.py --data_path data/xval_ner_shot_1_out_1
```

## Training & Validation & Evaluation

 - We try ten different random seeds (999, 189, 114, 929, 290, 848, 538, 874, 295, 266) and report average F1 scores in the [paper](https://arxiv.org/pdf/2009.09568.pdf) . 
 - For other data splits, please change the dataset path as "xval_snips_shot_{1,5}\_out_{1,2,3,4,5,6,7}" and "xval_ner_shot_{1,5}\_out_{1,2,3,4}".

* ProtoNet+CDT+VP:
```shell
bash run_few_shot_slot_tagger_protoNet_with_pure_bert.sh \
    --matching_similarity_y ctx \
    --matching_similarity_type xy1 \
    --matching_similarity_function dot \
    --test_finetune false \
    --dataset_name HIT_ner_shot_5_out_1 \
    --dataset_path ./data/xval_ner_shot_5_out_1 \
    --random_seed 999 \
    --model_removed no
```

* L-ProtoNet+CDT+VP:
```shell 
bash run_few_shot_slot_tagger_protoNet_with_pure_bert.sh \
    --matching_similarity_y ctx_desc \
    --matching_similarity_type xy1 \
    --matching_similarity_function dot \
    --test_finetune false \
    --dataset_name HIT_ner_shot_5_out_1 \
    --dataset_path ./data/xval_ner_shot_5_out_1 \
    --random_seed 999 \
    --model_removed no
```

* ProtoNet+CDT+VPB: 
```shell 
bash run_few_shot_slot_tagger_protoNet_with_pure_bert.sh \
    --matching_similarity_y ctx \
    --matching_similarity_type xy \
    --matching_similarity_function euclidean2 \
    --test_finetune false \
    --dataset_name HIT_ner_shot_5_out_1 \
    --dataset_path ./data/xval_ner_shot_5_out_1 \
    --random_seed 999 \
    --model_removed no
```
You can refer to an example of saved [log file](./example_log_file.txt) which is produced by the above script.

* L-ProtoNet+CDT+VPB:
```shell 
bash run_few_shot_slot_tagger_protoNet_with_pure_bert.sh \
    --matching_similarity_y ctx_desc \
    --matching_similarity_type xy \
    --matching_similarity_function euclidean2 \
    --test_finetune false \
    --dataset_name HIT_ner_shot_5_out_1 \
    --dataset_path ./data/xval_ner_shot_5_out_1 \
    --random_seed 999 \
    --model_removed no
```

If you want to keep fine-tuning the model with the support set of the target domain after pre-training on source domains, please set "--test_finetune true".

## Citation
This code has been written using PyTorch >= 1.4.0. If you use any source codes included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@article{zhu2020vector,
  title={Vector Projection Network for Few-shot Slot Tagging in Natural Language Understanding},
  author={Zhu, Su and Cao, Ruisheng and Chen, Lu and Yu, Kai},
  journal={arXiv preprint arXiv:2009.09568},
  year={2020}
}
@InProceedings{zhu2021few,
author="Zhu, Su and Chen, Lu and Cao, Ruisheng and Chen, Zhi and Miao, Qingliang and Yu, Kai",
title="Few-Shot NLU with Vector Projection Distance and Abstract Triangular CRF",
booktitle="Natural Language Processing and Chinese Computing",
year="2021",
pages="505--516",
}
</pre>
