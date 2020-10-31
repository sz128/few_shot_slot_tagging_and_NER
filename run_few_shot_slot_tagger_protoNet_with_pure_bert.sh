#!/bin/bash

. ./path.sh

task_st=slot_tagger_with_abstract_crf # slot_tagger, slot_tagger_with_adaptive_crf, slot_tagger_with_abstract_crf
task_sc=hiddenAttention # hiddenAttention, hiddenCNN, maxPooling, 2tails
task_sc_type=single_cls_CE # single_cls_CE, multi_cls_BCE
slot_embedding_type=with_BIO # with_BIO, with_BI, without_BIO
matching_similarity_type=xy # xy, x1y, xy1, x1y1
slot_embedding_dependent_query_encoding=none # none, ctx, desc
matching_similarity_y=ctx # ctx, desc, ctx_desc
matching_similarity_function=euclidean2 # dot, euclidean, euclidean2

dataset_name=HIT_ner_shot_5_out_1
dataset_path=./data/xval_ner_shot_5_out_1

input_embedding_type=tf_emb #word_emb, char_emb, word_char_emb, tf_emb, word_tf_emb, char_tf_emb, word_char_tf_emb
input_pretrained_wordEmb= #../data/few_shot_NLU/ACL2020_few_shot_slot_tagging_data/data/snips_shot_1_word2vec_Glove-KazumaChar_400d_uncased.txt
input_pretrained_transformer_model_type=bert # bert, xlnet
input_pretrained_transformer_model_name=bert-base-uncased # bert-base-cased, xlnet-base-cased
input_pretrained_wordEmb_fixed=false
input_word_lowercase=false

arch_dropout=0.1 # For tf_emb, dropout_rate of BERT\XLNET\etc. is still 0.1

tr_st_weight=1 #0.5
test_finetune=false # true false
test_finetune_accumulate_grad_batchSize=0
tr_optim=adam
tr_lr=1e-5
tf_layerwise_lr_decay=0.9
tr_lr_np=1e-3
tr_batchSize=20 # it will be reset to 1 in the training script for HIT data format
tr_max_norm=1
tr_max_epoch=5 #50
tr_exp_path=exp
random_seed=999
model_removed=no # yes, no

deviceId=0

train_or_test=train
saved_path=exp/ori_slot_tagger/model_slot_tagger/data_bcd_map_weather/bidir_True__emb_dim_100__hid_dim_200_x_1__bs_20__dropout_0.5__optimizer_adam__lr_0.001__mn_5.0__me_50__tes_100__lowercase  # used if $train_or_test == test;

source ./utils/parse_options.sh

if [[ $input_word_lowercase != true && $input_word_lowercase != True ]]; then
  unset input_word_lowercase
fi
if [[ $input_pretrained_wordEmb_fixed != true && $input_pretrained_wordEmb_fixed != True ]]; then
  unset input_pretrained_wordEmb_fixed
fi
if [[ $test_finetune != true && $test_finetune != True ]]; then
  unset test_finetune
fi

task_args="--task_st $task_st --task_sc $task_sc --task_sc_type $task_sc_type --slot_embedding_type ${slot_embedding_type} --matching_similarity_type ${matching_similarity_type} --slot_embedding_dependent_query_encoding ${slot_embedding_dependent_query_encoding} --matching_similarity_y ${matching_similarity_y} --matching_similarity_function ${matching_similarity_function}"
dataset_args="--dataset_name $dataset_name --dataset_path $dataset_path"
arch_args="--arch_dropout $arch_dropout"
tr_args="--tr_optim $tr_optim --tr_lr $tr_lr --tf_layerwise_lr_decay ${tf_layerwise_lr_decay} --tr_lr_np $tr_lr_np --tr_batchSize $tr_batchSize --tr_max_norm $tr_max_norm --tr_max_epoch $tr_max_epoch --tr_exp_path $tr_exp_path --tr_st_weight $tr_st_weight ${test_finetune:+--test_finetune} --test_finetune_accumulate_grad_batchSize ${test_finetune_accumulate_grad_batchSize} --random_seed ${random_seed} --model_removed ${model_removed}"

if [[ $train_or_test == "train" ]]; then
  python scripts/slot_tagging_with_prototypical_network_with_pure_bert.py $task_args $dataset_args $arch_args $tr_args \
    --deviceId $deviceId \
    --input_embedding_type $input_embedding_type \
    ${input_word_lowercase:+--input_word_lowercase} \
    ${input_pretrained_wordEmb:+--input_pretrained_wordEmb ${input_pretrained_wordEmb}} \
    ${input_pretrained_transformer_model_type:+--input_pretrained_transformer_model_type ${input_pretrained_transformer_model_type}} \
    ${input_pretrained_transformer_model_name:+--input_pretrained_transformer_model_name ${input_pretrained_transformer_model_name}} \
    ${input_pretrained_wordEmb_fixed:+--input_pretrained_wordEmb_fixed}
else
  python scripts/slot_tagging_with_prototypical_network_with_pure_bert.py $task_args $dataset_args $arch_args $tr_args \
    --deviceId $deviceId \
    --input_embedding_type $input_embedding_type \
    ${input_word_lowercase:+--input_word_lowercase} \
    ${input_pretrained_wordEmb:+--input_pretrained_wordEmb ${input_pretrained_wordEmb}} \
    ${input_pretrained_transformer_model_type:+--input_pretrained_transformer_model_type ${input_pretrained_transformer_model_type}} \
    ${input_pretrained_transformer_model_name:+--input_pretrained_transformer_model_name ${input_pretrained_transformer_model_name}} \
    --testing --testing_output_path ${saved_path} --testing_model_read_path ${saved_path}/model --testing_input_read_vocab ${saved_path}/vocab
fi
