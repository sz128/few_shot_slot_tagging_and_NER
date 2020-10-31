""" slot tagging with multi-task learning for zero-shot cases """

import os, sys, time
import logging
import argparse
import gc
import random
import numpy as np
import copy
import torch
import re
#import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from models.utils.optim_bert import BertAdam

from models.utils.config import ModelConfig
from models.slot_intent_with_prototypical_network_and_pure_bert import FewShotIntentSlot_ProtoNet

from utils.pretrained_transformer import load_pretrained_transformer
from utils.vocab_reader import SLUWordTokenizer, SLUCharTokenizer, read_input_vocab_from_data_file, read_input_vocab_from_data_file_in_HIT_form
from utils.vocab_reader import FewShotSlotVocab
from utils.basic_data_reader import SlotIntentDataset, convert_examples_to_features, collate_fn_do_nothing, prepare_inputs_of_word_sequences_for_bert_xlnet
from utils.data_reader_HIT import FewShotSlotIntentDataset_in_HIT_form, read_label_indicator_of_support_set
import utils.metric as metric
from utils.util import set_logger, setup_device

def get_hyperparam_string(args):
    """Hyerparam string."""
    task_path = 'model_%s' % (args.task)
    dataset_path = 'data_%s' % (args.dataset_name)
    
    exp_name = 'preTF-%s' % (args.input_pretrained_transformer_model_name)
    exp_name += '_bs-%s' % (args.tr_batchSize)
    exp_name += '_dp-%s' % (args.arch_dropout)
    exp_name += '_optim-%s' % (args.tr_optim)
    exp_name += '_lr-%s' % (args.tr_lr)
    exp_name += '_layer_decay-%s' % (args.tf_layerwise_lr_decay)
    exp_name += '__lr_np_%s' % (args.tr_lr_np)
    exp_name += '_mn-%s' % (args.tr_max_norm)
    exp_name += '_me-%s' % (args.tr_max_epoch)
    exp_name += '_seed-%s' % (args.random_seed)

    if args.input_word_lowercase:
        exp_name += '_uncased'
    else:
        exp_name += '_cased'
    exp_name += '_%s' % (args.input_embedding_type)

    return task_path, dataset_path, exp_name

def set_exp_path(args, exp_path_tag):
    args.task = args.task_st + '__and__' + args.task_sc + '__' + args.task_sc_type
    task_path, dataset_path, exp_name = get_hyperparam_string(args)
    if args.task_sc:
        exp_name += '__alpha_%s' % (args.tr_st_weight)
    exp_name += '__slot_emb_%s' % (args.slot_embedding_type)
    exp_name += '__match_sim_%s' % (args.matching_similarity_type)
    exp_name += '_y_%s_f_%s' % (args.matching_similarity_y, args.matching_similarity_function)
    exp_name += '__QEnc_on_LE_%s' % (args.slot_embedding_dependent_query_encoding)
    if args.test_finetune:
        exp_name += '__finetune'
    exp_path = os.path.join(args.tr_exp_path, exp_path_tag, task_path, dataset_path, exp_name)
    return exp_path

def set_seed(args):
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        if args.device.type != 'cuda':
            print("WARNING: You have a CUDA device, so you should probably run with --deviceId [1|2|3]")
        else:
            torch.cuda.manual_seed(args.random_seed)
            #if args.n_gpu > 0:
            #    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

def decode_simBERT(args, model_ori, eval_dataset, output_path):
    word_tokenizer = args.word_tokenizer
    char_tokenizer = args.char_tokenizer
    tf_tokenizer = args.tf_tokenizer
    tf_input_args = args.tf_input_args
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.test_batchSize, collate_fn=collate_fn_do_nothing)
    
    model_ori.eval()
    losses = []
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
    marco_f1_scores = []
    with open(output_path, 'w') as writer, torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if args.dataset_name.startswith("HIT_"):
                support_batch, query_batch = batch[0][0], batch[0][1]
                slot_tags_vocab, intents_vocab = batch[0][2], batch[0][3]
                support_batch_inputs = convert_examples_to_features(support_batch, word_tokenizer=word_tokenizer, char_tokenizer=char_tokenizer, tf_tokenizer=tf_tokenizer, tf_input_args=tf_input_args, slot_tags_vocab=slot_tags_vocab, intents_vocab=intents_vocab, bos_eos=args.input_bos_eos, intent_multi_class=args.intent_multi_class, slot_tag_enc_dec=args.enc_dec, device=args.device)
            query_batch_inputs = convert_examples_to_features(query_batch, word_tokenizer=word_tokenizer, char_tokenizer=char_tokenizer, tf_tokenizer=tf_tokenizer, tf_input_args=tf_input_args, slot_tags_vocab=slot_tags_vocab, intents_vocab=intents_vocab, bos_eos=args.input_bos_eos, intent_multi_class=args.intent_multi_class, slot_tag_enc_dec=args.enc_dec, device=args.device)
            top_pred_tags = model_ori.decode_by_similarity_of_BERT(support_batch_inputs["inputs"], support_batch_inputs["input_mask"], support_batch_inputs["tag_ids"], support_batch_inputs["intent_ids"], query_batch_inputs["inputs"])
            
            losses.append([0, 0])
            
            line_nums = query_batch_inputs["line_nums"]
            lengths = query_batch_inputs["lengths"]
            inner_TP, inner_FP, inner_FN = 0, 0, 0
            for idx, length in enumerate(lengths):
                pred_tag_seq = [slot_tags_vocab._convert_id_to_token(index) for index in top_pred_tags[idx]][:length]
                lab_tag_seq = query_batch_inputs["tags"][idx]
                assert len(lab_tag_seq) == length
                pred_slot_chunks = metric.get_chunks(['O'] + pred_tag_seq + ['O'])
                label_slot_chunks = metric.get_chunks(['O'] + lab_tag_seq + ['O'])
                TP_1, FP_1, FN_1 = metric.analysis_fscore(pred_slot_chunks, label_slot_chunks)
                TP += TP_1
                FP += FP_1
                FN += FN_1

                inner_TP += TP_1
                inner_FP += FP_1
                inner_FN += FN_1

                tokens = query_batch_inputs["tokens"][idx]
                assert len(tokens) == length
                token_tag_list = [':'.join((tokens[_idx], pred_tag_seq[_idx], lab_tag_seq[_idx])) for _idx in range(length)]
                
                if args.testing:
                    writer.write(str(line_nums[idx])+' : '+' '.join(token_tag_list)+'\n')
                else:
                    writer.write(' '.join(token_tag_list)+'\n')
            marco_f1_scores.append(2 * inner_TP / (2 * inner_TP + inner_FP + inner_FN) if 2 * inner_TP + inner_FP + inner_FN != 0 else 0)

    mean_losses = np.mean(losses, axis=0)
    slot_metrics = metric.compute_fscore(TP, FP, FN)
    args.logger.info('mirco f1 = %.2f;\tepisode number = %d;\taveraged mirco f1 = %.2f' % (slot_metrics['f'], len(marco_f1_scores), 100 * sum(marco_f1_scores)/ len(marco_f1_scores)))

def decode(args, model_ori, eval_dataset, output_path, test_finetune_step=0, model_ori_saved_path=None, lr=0.001, optim_name='adam', lr_np=1e-3):
    word_tokenizer = args.word_tokenizer
    char_tokenizer = args.char_tokenizer
    tf_tokenizer = args.tf_tokenizer
    tf_input_args = args.tf_input_args
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.test_batchSize, collate_fn=collate_fn_do_nothing)
    
    model_ori.eval()
    losses = []
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
    TP2, FP2, FN2, TN2 = 0.0, 0.0, 0.0, 0.0
    marco_f1_scores = []
    with open(output_path, 'w') as writer:
        for step, batch in enumerate(eval_dataloader):
            if args.dataset_name.startswith("HIT_"):
                support_batch, query_batch = batch[0][0], batch[0][1]
                slot_tags_vocab, intents_vocab = batch[0][2], batch[0][3]
                slot_desc_in_words = batch[0][4]
                slot_head_labels_token_ids, slot_other_labels_seg_ids, slot_other_labels_selected_slot_ids = slot_tags_vocab.get_elements_used_in_label_embeddings(device=args.device)
                support_batch_inputs = convert_examples_to_features(support_batch, word_tokenizer=word_tokenizer, char_tokenizer=char_tokenizer, tf_tokenizer=tf_tokenizer, tf_input_args=tf_input_args, slot_tags_vocab=slot_tags_vocab, intents_vocab=intents_vocab, bos_eos=args.input_bos_eos, intent_multi_class=args.intent_multi_class, slot_tag_enc_dec=args.enc_dec, device=args.device)
                slot_label_indicator, intent_label_indicator = read_label_indicator_of_support_set(slot_tags_vocab, intents_vocab, support_batch_inputs["tags"], support_batch_inputs["intents"], indicator_type='PN', slot_embedding_type=args.slot_embedding_type, device=args.device)
                label_desc_inputs = prepare_inputs_of_word_sequences_for_bert_xlnet(slot_desc_in_words, tf_tokenizer, bos_eos=args.input_bos_eos, device=args.device, **tf_input_args)
            if test_finetune_step > 0:
                model_ori.load_model(model_ori_saved_path) # load original model parameters for different episodes
                #for param in model_ori.input_embeddings.parameters():
                #    param.requires_grad = False
                #for param in model_ori.sentence_encoder.parameters():
                #    param.requires_grad = False
                if args.input_embedding_type in {'tf_emb', 'word_tf_emb', 'char_tf_emb', 'word_char_tf_emb'}:
                    named_params = list(model_ori.named_parameters())
                    named_params = list(filter(lambda p: p[1].requires_grad, named_params))
                    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                    large_lr = ['crf']
                    if args.tf_layerwise_lr_decay >= 1.0:
                        optimizer_grouped_parameters = [
                            {'params': [p for n, p in named_params if (not any(nd in n for nd in no_decay)) and (not any(ll in n for ll in large_lr))], 'weight_decay': 0.01}, # 0.01
                            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay) and (not any(ll in n for ll in large_lr))], 'weight_decay': 0.0},
                            {'params': [p for n, p in named_params if any(ll in n for ll in large_lr)], 'weight_decay': 0.0, 'lr': lr_np} #1e-3
                            ]
                    else:
                        optimizer_grouped_parameters = []
                        for n, p in named_params:
                            params_group = {}
                            params_group['params'] = p
                            params_group['weight_decay'] = 0.01 if (not any(nd in n for nd in no_decay)) and (not any(ll in n for ll in large_lr)) else 0.0
                            if any(ll in n for ll in large_lr):
                                params_group['lr'] = lr_np
                            else:
                                if 'tf_model.embeddings' in n:
                                    depth = 0
                                elif 'tf_model.encoder.layer' in n:
                                    depth = int(re.search(r'tf_model.encoder.layer.(\d+)', n).group(1)) + 1
                                else:
                                    depth = args.tf_model.config.num_hidden_layers
                                params_group['lr'] = lr * \
                                        (args.tf_layerwise_lr_decay ** (args.tf_model.config.num_hidden_layers - depth))
                            optimizer_grouped_parameters.append(params_group)
                else:
                    optimizer_grouped_parameters = [{'params': [param for param in model_ori.parameters() if param.requires_grad]}]
                if optim_name == 'adam':
                    optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8) # (beta1, beta2)
                elif optim_name == 'bertadam':
                    num_train_optimization_steps = 1 * test_finetune_step
                    optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1, t_total=num_train_optimization_steps, max_grad_norm=1)
                model_ori.train()
                for i in range(test_finetune_step):
                    optimizer.zero_grad()
                    support_batch_size = len(support_batch)
                    if args.test_finetune_accumulate_grad_batchSize == 0:
                        accum_batch_size = support_batch_size
                    else:
                        accum_batch_size = args.test_finetune_accumulate_grad_batchSize # 15, 20
                    for j in range(0, support_batch_size, accum_batch_size):
                        slot_tag_embeds, intent_embeds = model_ori.get_label_embeddings_from_support_set(support_batch_inputs["inputs"], support_batch_inputs["lengths"], support_batch_inputs["input_mask"], slot_label_indicator, intent_label_indicator, slot_head_labels_token_ids, slot_other_labels_seg_ids=slot_other_labels_seg_ids, slot_other_labels_selected_slot_ids=slot_other_labels_selected_slot_ids, label_desc_inputs=label_desc_inputs)
                        accum_batch = support_batch[j:j + accum_batch_size]
                        inputs = convert_examples_to_features(accum_batch, word_tokenizer=word_tokenizer, char_tokenizer=char_tokenizer, tf_tokenizer=tf_tokenizer, tf_input_args=tf_input_args, slot_tags_vocab=slot_tags_vocab, intents_vocab=intents_vocab, bos_eos=args.input_bos_eos, intent_multi_class=args.intent_multi_class, slot_tag_enc_dec=args.enc_dec, device=args.device)
                        outputs = model_ori(slot_tag_embeds, intent_embeds, inputs["inputs"], inputs["lengths"], inputs["input_mask"], slot_tags=inputs["tag_ids"], intents=inputs["intent_ids"], slot_tag_masked_output=None, intent_masked_output=None, slot_tag_to_id=slot_tags_vocab.label_to_id)
                        tag_loss, intent_loss = outputs[:2]
                        total_loss = args.tr_st_weight * tag_loss + (1 - args.tr_st_weight) * intent_loss
                        total_loss.backward()
                    optimizer.step()
                gc.collect()
                model_ori.eval()
                model = model_ori
            else:
                model = model_ori
            with torch.no_grad():
                slot_tag_embeds, intent_embeds = model.get_label_embeddings_from_support_set(support_batch_inputs["inputs"], support_batch_inputs["lengths"], support_batch_inputs["input_mask"], slot_label_indicator, intent_label_indicator, slot_head_labels_token_ids, slot_other_labels_seg_ids=slot_other_labels_seg_ids, slot_other_labels_selected_slot_ids=slot_other_labels_selected_slot_ids, label_desc_inputs=label_desc_inputs)
                inputs = convert_examples_to_features(query_batch, word_tokenizer=word_tokenizer, char_tokenizer=char_tokenizer, tf_tokenizer=tf_tokenizer, tf_input_args=tf_input_args, slot_tags_vocab=slot_tags_vocab, intents_vocab=intents_vocab, bos_eos=args.input_bos_eos, intent_multi_class=args.intent_multi_class, slot_tag_enc_dec=args.enc_dec, device=args.device)
                outputs = model(slot_tag_embeds, intent_embeds, inputs["inputs"], inputs["lengths"], inputs["input_mask"], slot_tags=inputs["tag_ids"], intents=inputs["intent_ids"], slot_tag_masked_output=None, intent_masked_output=None, slot_tag_to_id=slot_tags_vocab.label_to_id)
            tag_loss, intent_loss, tag_logits, intent_logits = outputs[:4]
            if args.enc_dec:
                rnn_out, reversed_h_t_c_t = outputs[4:]
                top_pred_tags, top_pred_intents = model.decode_top_hyp_for_enc_dec(slot_tag_embeds, intent_logits, rnn_out, inputs["input_mask"], reversed_h_t_c_t, inputs["tag_ids"][:, 0:1], inputs["lengths"], slot_tags_vocab)
            else:
                top_pred_tags, top_pred_intents = model.decode_top_hyp(slot_tag_embeds, tag_logits, intent_logits, inputs["input_mask"], slot_tag_to_id=slot_tags_vocab.label_to_id)
            
            losses.append([tag_loss.item()/sum(inputs["lengths"]), intent_loss.item()/len(inputs["lengths"])])
            
            line_nums = inputs["line_nums"]
            lengths = inputs["lengths"]
            inner_TP, inner_FP, inner_FN = 0, 0, 0
            for idx, length in enumerate(lengths):
                pred_tag_seq = [slot_tags_vocab._convert_id_to_token(index) for index in top_pred_tags[idx]][:length]
                lab_tag_seq = inputs["tags"][idx]
                assert len(lab_tag_seq) == length
                pred_slot_chunks = metric.get_chunks(['O'] + pred_tag_seq + ['O'])
                label_slot_chunks = metric.get_chunks(['O'] + lab_tag_seq + ['O'])
                TP_1, FP_1, FN_1 = metric.analysis_fscore(pred_slot_chunks, label_slot_chunks)
                TP += TP_1
                FP += FP_1
                FN += FN_1

                inner_TP += TP_1
                inner_FP += FP_1
                inner_FN += FN_1
                
                tokens = inputs["tokens"][idx]
                assert len(tokens) == length
                token_tag_list = [':'.join((tokens[_idx], pred_tag_seq[_idx], lab_tag_seq[_idx])) for _idx in range(length)]

                label_intents = inputs["intents"][idx]
                if args.intent_multi_class:
                    assert isinstance(label_intents, list)
                    pred_intents = [intents_vocab._convert_id_to_token(i) for i,p in enumerate(top_pred_intents[idx]) if p > 0.5] 
                    TP_FP_FN = metric.analysis_fscore(set(pred_intents), set(label_intents))
                    TP2 += TP_FP_FN[0]
                    FP2 += TP_FP_FN[1]
                    FN2 += TP_FP_FN[2]
                    pred_intents_str = ';'.join(pred_intents)
                else:
                    pred_intent = intents_vocab._convert_id_to_token(top_pred_intents[idx])
                    if isinstance(label_intents, str):
                        label_intents = [label_intents]
                    if pred_intent in label_intents:
                        TP2 += 1
                    else:
                        FP2 += 1
                        FN2 += 1
                    pred_intents_str = pred_intent
                label_intents_str = ';'.join(label_intents)

                if args.testing:
                    writer.write(str(line_nums[idx])+' : '+' '.join(token_tag_list)+' <=> '+label_intents_str+' <=> '+pred_intents_str+'\n')
                else:
                    writer.write(' '.join(token_tag_list)+' <=> '+label_intents_str+' <=> '+pred_intents_str+'\n')
            
            marco_f1_scores.append(2 * inner_TP / (2 * inner_TP + inner_FP + inner_FN) if 2 * inner_TP + inner_FP + inner_FN != 0 else 0)

    mean_losses = np.mean(losses, axis=0)
    slot_metrics = metric.compute_fscore(TP, FP, FN)
    intent_metrics = metric.compute_fscore(TP2, FN2, FP2)
    args.logger.info('episode number = %d;\taveraged mirco f1 = %.2f' % (len(marco_f1_scores), 100 * sum(marco_f1_scores)/ len(marco_f1_scores)))
    
    #return mean_losses, slot_metrics, intent_metrics
    return mean_losses, {'f': 100 * sum(marco_f1_scores)/ len(marco_f1_scores)}, intent_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_st', required=True, help='slot_tagger | slot_tagger_with_focus | slot_tagger_with_crf | slot_tagger_with_adaptive_crf | slot_tagger_with_abstract_crf | slot_tagger_with_attention_decoder | slot_tagger_with_focus_and_attention_decoder')
    parser.add_argument('--task_sc', required=True, help='2tails | maxPooling | hiddenCNN | hiddenAttention')
    parser.add_argument('--task_sc_type', required=True, help='single_cls_CE | multi_cls_BCE')
    
    parser.add_argument('--slot_embedding_type', required=True, help='with_BIO | with_BI | without_BIO')
    parser.add_argument('--matching_similarity_type', required=True, help='xy | xy1 | x1y | x1y1 | rxy')
    parser.add_argument('--matching_similarity_y', required=True, help='ctx | desc | ctx_desc')
    parser.add_argument('--matching_similarity_function', required=True, help='dot | euclidean')
    parser.add_argument('--slot_embedding_dependent_query_encoding', required=True, help='none | ctx | desc')

    parser.add_argument('--dataset_name', required=True, help='HIT_NER_shot_1_xval_1 | SJTU_shot_1_xval_1')
    parser.add_argument('--dataset_path', required=True, help='path to dataset')

    parser.add_argument('--input_mini_word_freq', type=int, default=2, help='mini_word_freq in the training data used in building input vocabulary when pretrained transformer is not used')
    parser.add_argument('--input_word_lowercase', action='store_true', help='word lowercase')
    parser.add_argument('--input_bos_eos', action='store_true', help='Whether to add <s> and </s> to the input sentence (default is not)')
    parser.add_argument('--input_save_vocab', default='vocab', help='save vocab to this file')
    parser.add_argument('--input_embedding_type', default="word_emb", help='word_emb, char_emb, word_char_emb, tf_emb, word_tf_emb, char_tf_emb, word_char_tf_emb')
    parser.add_argument('--input_pretrained_wordEmb', required=False, help='read word embedding from word2vec file')
    parser.add_argument('--input_pretrained_wordEmb_fixed', action='store_true', help='pre-trained word embeddings are fixed')
    parser.add_argument('--input_pretrained_transformer_model_type', required=False, help='bert, xlnet')
    parser.add_argument('--input_pretrained_transformer_model_name', required=False, help='bert-base-uncased, bert-base-cased, bert-large-uncased, bert-large-cased, bert-base-multilingual-cased, bert-base-chinese; xlnet-base-cased, xlnet-large-cased')
    #parser.add_argument('--input_pretrained_transformer_model_fixed', action='store_true', help='fix pretrained (bert/xlnet) model')

    parser.add_argument('--model_save_path', default='model', help='save model to this file')
    parser.add_argument('--model_removed', default='no', help='yes | no; (not to save model finally)')
    #parser.add_argument('--arch_word_emb_size', type=int, default=100, help='word embedding dimension')
    #parser.add_argument('--arch_char_emb_size', type=int, default=25, help='char embedding dimension')
    #parser.add_argument('--arch_char_hidden_size', type=int, default=25, help='char-level RNN hidden dimension')
    #parser.add_argument('--arch_tag_emb_size', type=int, default=100, help='tag embedding dimension')
    #parser.add_argument('--arch_hidden_size', type=int, default=100, help='hidden layer dimension')
    #parser.add_argument('--arch_num_layers', type=int, default=0, help='number of hidden layers')
    #parser.add_argument('--arch_rnn_cell', default='lstm', help='type of RNN cell: rnn, gru, lstm')
    #parser.add_argument('--arch_bidirectional', action='store_true', help='Whether to use bidirectional RNN (default is unidirectional)')
    #parser.add_argument('--arch_decoder_tied', action='store_true', help='To tie the output layer and input embedding in decoder')
    parser.add_argument('--arch_dropout', type=float, default=0., help='dropout rate at each non-recurrent layer')
    parser.add_argument('--arch_init_weight', type=float, default=0.2, help='all weights will be set to [-init_weight, init_weight] during initialization')

    parser.add_argument('--tr_lr', type=float, default=1e-5, help='learning rate for finetuning BERT')
    parser.add_argument('--tf_layerwise_lr_decay', type=float, default=1.0, help='layerwise lr decay for finetuning BERT')
    parser.add_argument('--tr_lr_np', type=float, default=1e-3, help='learning rate for newly initialized paramters')
    parser.add_argument('--tr_lr_warmup_proportion', type=float, default=0.1, help='Linear warmup over warmup_proportion of total steps.')
    parser.add_argument('--tr_batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--tr_max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
    parser.add_argument('--tr_max_epoch', type=int, default=50, help='max number of epochs to train for')
    parser.add_argument('--tr_exp_path', default='exp', help='Where to store samples and models')
    parser.add_argument('--tr_optim', default='adam', help='choose an optimizer: sgd, adam, adamW')
    parser.add_argument('--tr_st_weight', type=float, default=0.5, help='loss weight for slot tagging task, ranging from 0 to 1.')
    parser.add_argument('--test_batchSize', type=int, default=0, help='input batch size in decoding')
    parser.add_argument('--test_finetune', action='store_true', help='finetune with support set on the test set')
    parser.add_argument('--test_finetune_accumulate_grad_batchSize', type=int, default=0, help='test_finetune_accumulate_grad_batchSize: 0 means not accumulating gradients')

    parser.add_argument('--deviceId', type=int, default=-1, help='train model on ith gpu. -1:cpu, 0:auto_select')
    parser.add_argument('--random_seed', type=int, default=999, help='set initial random seed')
    parser.add_argument('--noStdout', action='store_true', help='Only log to a file; no stdout')
    parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
    parser.add_argument('--testing_output_path', required=False, help='Online test: output path')
    parser.add_argument('--testing_model_read_path', required=False, help='Online test: read model from this file')
    parser.add_argument('--testing_input_read_vocab', required=False, help='Online test: read input vocab from this file')

    args = parser.parse_args()

    config = ModelConfig.from_argparse(args)
    print(config)

    assert args.task_st in {'slot_tagger', 'slot_tagger_with_adaptive_crf', 'slot_tagger_with_abstract_crf', 'slot_tagger_with_focus'} #, 'slot_tagger_with_crf', 'slot_tagger_with_attention_decoder', 'slot_tagger_with_focus_and_attention_decoder'}
    assert args.task_sc in {'2tails', 'maxPooling', 'hiddenCNN', 'hiddenAttention'}
    assert args.task_sc_type in {'single_cls_CE', 'multi_cls_BCE'}
    args.intent_multi_class = (args.task_sc_type == "multi_cls_BCE")
    if args.task_st in {'slot_tagger_with_focus', 'slot_tagger_with_attention_decoder', 'slot_tagger_with_focus_and_attention_decoder'}:
        args.enc_dec = True
    else:
        args.enc_dec = False
    assert 0 < args.tr_st_weight <= 1
    assert args.slot_embedding_type in {'with_BIO', 'with_BI', 'without_BIO'}
    assert args.matching_similarity_type in {'xy', 'x1y', 'xy1', 'x1y1', 'rxy'}
    assert args.matching_similarity_y in {'ctx', 'ctx_desc', 'desc'}
    assert args.matching_similarity_function in {'dot', 'euclidean', 'euclidean2', 'euclidean3'}
    assert args.slot_embedding_dependent_query_encoding in {'none', 'ctx', 'desc'}
    #assert args.input_embedding_type in {'word_emb', 'char_emb', 'word_char_emb', 'tf_emb', 'word_tf_emb', 'char_tf_emb', 'word_char_tf_emb'}
    assert args.input_embedding_type == 'tf_emb'
    assert args.testing == bool(args.testing_output_path) == bool(args.testing_model_read_path) ==  bool(args.testing_input_read_vocab)
    if args.test_batchSize == 0:
        args.test_batchSize = args.tr_batchSize
    if args.dataset_name.startswith("HIT_"):
        args.tr_batchSize = 1
        args.test_batchSize = 1
    assert 0 < args.tf_layerwise_lr_decay <= 1.0
    assert args.model_removed in {'yes', 'no'}

    if not args.testing:
        exp_path = set_exp_path(args, 'few_shot_slot_intent/prototypical_network_with_pure_bert')
    else:
        exp_path = args.testing_output_path
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    logger = set_logger(args, exp_path)
    args.logger = logger
    args.device = setup_device(args.deviceId, logger)
    set_seed(args)

    ## read vocab && dataset
    dataroot = args.dataset_path
    if args.dataset_name.startswith("HIT_"):
        train_data_path = os.path.join(dataroot, 'train.json')
        valid_data_path = os.path.join(dataroot, 'valid.json')
        test_data_path = os.path.join(dataroot, 'test.json')
        slot_desc_data_path = os.path.join(dataroot, 'slot_description')
    else:
        # not ready
        train_data_path = os.path.join(dataroot, 'train')
        valid_data_path = os.path.join(dataroot, 'valid')
        test_data_path = os.path.join(dataroot, 'test')
    
    ### build vocab
    #### complex input embeddings
    if args.input_embedding_type in {'word_emb', 'word_char_emb', 'word_tf_emb', 'word_char_tf_emb'}:
        word_tokenizer = SLUWordTokenizer(bos_eos=args.input_bos_eos, lowercase=args.input_word_lowercase)
    else:
        word_tokenizer = None
    if args.input_embedding_type in {'char_emb', 'word_char_emb', 'char_tf_emb', 'word_char_tf_emb'}:
        char_tokenizer = SLUCharTokenizer(bos_eos=args.input_bos_eos, lowercase=args.input_word_lowercase)
    else:
        char_tokenizer = None
    if args.input_embedding_type in {'tf_emb', 'word_tf_emb', 'char_tf_emb', 'word_char_tf_emb'}:
        logger.info("Loading pretrained %s model with %s" % (args.input_pretrained_transformer_model_type, args.input_pretrained_transformer_model_name))
        tf_tokenizer, tf_model = load_pretrained_transformer(args.input_pretrained_transformer_model_type, args.input_pretrained_transformer_model_name)
        tf_input_args = {
                'cls_token_at_end': bool(args.input_pretrained_transformer_model_type in ['xlnet']),  # xlnet has a cls token at the end
                'cls_token_segment_id': 2 if args.input_pretrained_transformer_model_type in ['xlnet'] else 0,
                'pad_on_left': bool(args.input_pretrained_transformer_model_type in ['xlnet']), # pad on the left for xlnet
                'pad_token_segment_id': 4 if args.input_pretrained_transformer_model_type in ['xlnet'] else 0,
                }
    else:
        tf_tokenizer, tf_model = None, None
        tf_input_args = {}
    assert word_tokenizer is not None or char_tokenizer is not None or tf_tokenizer is not None
    if not args.testing:
        if word_tokenizer is not None or char_tokenizer is not None:
            if args.dataset_name.startswith("HIT_"):
                read_input_vocab_from_data_file_in_HIT_form(train_data_path, word_tokenizer=word_tokenizer, char_tokenizer=char_tokenizer, lowercase=args.input_word_lowercase, mini_word_freq=args.input_mini_word_freq)
            else:
                read_input_vocab_from_data_file(train_data_path, word_tokenizer=word_tokenizer, char_tokenizer=char_tokenizer, lowercase=args.input_word_lowercase, mini_word_freq=args.input_mini_word_freq)
        if word_tokenizer is not None and args.input_pretrained_wordEmb:
            # pretrained-embedding initialization for training
            special_token_embeddings, pretrained_normal_token_num, pretrained_normal_token_embeddings, token_out_of_pretrained_emb_num = word_tokenizer.read_word2vec_inText(args.input_pretrained_wordEmb, device=args.device)
            logger.info("%s token(s) is/are out of pretrained word embeddings!" %(token_out_of_pretrained_emb_num))
        if word_tokenizer is not None:
            word_tokenizer.save_vocab(os.path.join(exp_path, args.input_save_vocab+'.input_word'))
        if char_tokenizer is not None:
            char_tokenizer.save_vocab(os.path.join(exp_path, args.input_save_vocab+'.input_char'))
        if tf_tokenizer is not None:
            xxxx = 0
            #tf_tokenizer.save_vocab(os.path.join(exp_path, args.input_save_vocab+'.input_tf'))
    else:
        if word_tokenizer is not None:
            word_tokenizer.read_vocab(args.testing_input_read_vocab+'.input_word')
        if char_tokenizer is not None:
            char_tokenizer.read_vocab(args.testing_input_read_vocab+'.input_char')
        if tf_tokenizer is not None:
            xxxx = 0
            #tf_tokenizer.read_vocab(args.testing_input_read_vocab+'.input_tf')
    config.input_word_vocab_size = word_tokenizer.get_vocab_size() if word_tokenizer is not None else None
    config.input_char_vocab_size = char_tokenizer.get_vocab_size() if char_tokenizer is not None else None
    config.input_tf_vocab_size = tf_tokenizer.vocab_size if tf_tokenizer is not None else None
    args.word_tokenizer = word_tokenizer
    args.char_tokenizer = char_tokenizer
    args.tf_tokenizer = tf_tokenizer
    args.tf_input_args = tf_input_args
    args.tf_model = tf_model
    #### output vocab
    basic_slot_tags_vocab = FewShotSlotVocab([], slot_embedding_type=args.slot_embedding_type, bos_eos=args.input_bos_eos)
    config.output_tag_pad_id = basic_slot_tags_vocab._convert_token_to_id(basic_slot_tags_vocab.pad_token)
    config.input_additional_feature_dim = None
    logger.info("Vocab size: input tokens (word-%s, char-%s, tf-%s)" % (config.input_word_vocab_size, config.input_char_vocab_size, config.input_tf_vocab_size))

    ### read data
    if args.dataset_name.startswith("HIT_"):
        if not args.testing:
            train_dataset = FewShotSlotIntentDataset_in_HIT_form(train_data_path, slot_desc_data_path, slot_embedding_type=args.slot_embedding_type, bert_tokenized=False, lowercase=args.input_word_lowercase, input_bos_eos=args.input_bos_eos)
        valid_dataset = FewShotSlotIntentDataset_in_HIT_form(valid_data_path, slot_desc_data_path, slot_embedding_type=args.slot_embedding_type, bert_tokenized=False, lowercase=args.input_word_lowercase, input_bos_eos=args.input_bos_eos)
        test_dataset = FewShotSlotIntentDataset_in_HIT_form(test_data_path, slot_desc_data_path, slot_embedding_type=args.slot_embedding_type, bert_tokenized=False, lowercase=args.input_word_lowercase, input_bos_eos=args.input_bos_eos)
    else:
        if not args.testing:
            train_dataset = FewShotSlotIntentDataset(train_data_path, lowercase=args.input_word_lowercase, input_bos_eos=args.input_bos_eos)
        valid_dataset = FewShotSlotIntentDataset(valid_data_path, lowercase=args.input_word_lowercase, input_bos_eos=args.input_bos_eos)
        test_dataset = FewShotSlotIntentDataset(test_data_path, lowercase=args.input_word_lowercase, input_bos_eos=args.input_bos_eos)
    
    ## init model architecture
    model = FewShotIntentSlot_ProtoNet(config, len(basic_slot_tags_vocab.init_token_to_id), slot_embedding_type=args.slot_embedding_type, matching_similarity_type=args.matching_similarity_type, slot_embedding_dependent_query_encoding=args.slot_embedding_dependent_query_encoding, matching_similarity_y=args.matching_similarity_y, matching_similarity_function=args.matching_similarity_function, device=args.device, pretrained_tf_model=tf_model)
    print(model)
    model.to(args.device)

    if not args.testing:
        # pretrained-embedding initialization for training
        if word_tokenizer is not None and args.input_pretrained_wordEmb:
            model.input_embeddings.load_pretrained_word_embeddings(special_token_embeddings, pretrained_normal_token_num, pretrained_normal_token_embeddings)
            if args.input_pretrained_wordEmb_fixed:
                model.input_embeddings.fix_word_embeddings()
    else:
        # read pretrained model for evaluation
        model.load_model(args.testing_model_read_path)

    ## training or testing
    if not args.testing:
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.tr_batchSize, collate_fn=collate_fn_do_nothing)
        batch_number = len(train_dataloader)
        #print(len(train_dataset), args.tr_batchSize, batch_number, len(train_dataloader))
        
        logger.info("***** Training starts at %s *****" % (time.asctime(time.localtime(time.time()))))
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num batches = %d", batch_number)
        logger.info("  Num Epochs = %d", args.tr_max_epoch)
        
        # optimizer
        if args.input_embedding_type in {'tf_emb', 'word_tf_emb', 'char_tf_emb', 'word_char_tf_emb'}:
            named_params = list(model.named_parameters())
            named_params = list(filter(lambda p: p[1].requires_grad, named_params))
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            large_lr = ['crf']
            if args.tf_layerwise_lr_decay >= 1.0:
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in named_params if (not any(nd in n for nd in no_decay)) and (not any(ll in n for ll in large_lr))], 'weight_decay': 0.01}, # 0.01
                    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay) and (not any(ll in n for ll in large_lr))], 'weight_decay': 0.0},
                    {'params': [p for n, p in named_params if any(ll in n for ll in large_lr)], 'weight_decay': 0.0, 'lr': args.tr_lr_np} #1e-3
                    ]
            else:
                optimizer_grouped_parameters = []
                for n, p in named_params:
                    params_group = {}
                    params_group['params'] = p
                    params_group['weight_decay'] = 0.01 if (not any(nd in n for nd in no_decay)) and (not any(ll in n for ll in large_lr)) else 0.0
                    if any(ll in n for ll in large_lr):
                        params_group['lr'] = args.tr_lr_np
                    else:
                        if 'tf_model.embeddings' in n:
                            depth = 0
                        elif 'tf_model.encoder.layer' in n:
                            depth = int(re.search(r'tf_model.encoder.layer.(\d+)', n).group(1)) + 1
                        else:
                            depth = args.tf_model.config.num_hidden_layers
                        params_group['lr'] = args.tr_lr * \
                                (args.tf_layerwise_lr_decay ** (args.tf_model.config.num_hidden_layers - depth))
                    optimizer_grouped_parameters.append(params_group)
        else:
            optimizer_grouped_parameters = [{'params': [param for param in model.parameters() if param.requires_grad]}]
        if args.tr_optim.lower() == 'sgd':
            optimizer = optim.SGD(optimizer_grouped_parameters, lr=args.tr_lr)
        elif args.tr_optim.lower() == 'adam':
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.tr_lr, betas=(0.9, 0.999), eps=1e-8) # (beta1, beta2)
        elif args.tr_optim.lower() == 'adamw':
            num_train_optimization_steps = batch_number * args.tr_max_epoch
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.tr_lr, eps=1e-8, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.tr_lr_warmup_proportion * num_train_optimization_steps), num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
        elif args.tr_optim.lower() == 'bertadam':
            num_train_optimization_steps = batch_number * args.tr_max_epoch
            optimizer = BertAdam(optimizer_grouped_parameters, lr=args.tr_lr, warmup=args.tr_lr_warmup_proportion, t_total=num_train_optimization_steps, max_grad_norm=args.tr_max_norm)
        else:
            exit()
        
        decode_simBERT(args, model, valid_dataset, os.path.join(exp_path, 'valid.iter'+str(-1)))
        decode_simBERT(args, model, test_dataset, os.path.join(exp_path, 'test.iter'+str(-1)))
        best_f1, best_result = -1, {}
        for i in range(args.tr_max_epoch):
            start_time = time.time()
            losses = []
            model.train()

            piece_steps = 1 if int(batch_number * 0.1) == 0 else int(batch_number * 0.1)
            for step, batch in enumerate(train_dataloader):
                if args.dataset_name.startswith("HIT_"):
                    support_batch, query_batch = batch[0][0], batch[0][1]
                    slot_tags_vocab, intents_vocab = batch[0][2], batch[0][3]
                    slot_desc_in_words = batch[0][4]
                    slot_head_labels_token_ids, slot_other_labels_seg_ids, slot_other_labels_selected_slot_ids = slot_tags_vocab.get_elements_used_in_label_embeddings(device=args.device)
                    support_batch_inputs = convert_examples_to_features(support_batch, word_tokenizer=word_tokenizer, char_tokenizer=char_tokenizer, tf_tokenizer=tf_tokenizer, tf_input_args=tf_input_args, slot_tags_vocab=slot_tags_vocab, intents_vocab=intents_vocab, bos_eos=args.input_bos_eos, intent_multi_class=args.intent_multi_class, slot_tag_enc_dec=args.enc_dec, device=args.device)
                    slot_label_indicator, intent_label_indicator = read_label_indicator_of_support_set(slot_tags_vocab, intents_vocab, support_batch_inputs["tags"], support_batch_inputs["intents"], indicator_type='PN', slot_embedding_type=args.slot_embedding_type, device=args.device)
                    label_desc_inputs = prepare_inputs_of_word_sequences_for_bert_xlnet(slot_desc_in_words, tf_tokenizer, bos_eos=args.input_bos_eos, device=args.device, **tf_input_args)
                    
                    optimizer.zero_grad()
                    slot_tag_embeds, intent_embeds = model.get_label_embeddings_from_support_set(support_batch_inputs["inputs"], support_batch_inputs["lengths"], support_batch_inputs["input_mask"], slot_label_indicator, intent_label_indicator, slot_head_labels_token_ids, slot_other_labels_seg_ids=slot_other_labels_seg_ids, slot_other_labels_selected_slot_ids=slot_other_labels_selected_slot_ids, label_desc_inputs=label_desc_inputs)
                    inputs = convert_examples_to_features(query_batch, word_tokenizer=word_tokenizer, char_tokenizer=char_tokenizer, tf_tokenizer=tf_tokenizer, tf_input_args=tf_input_args, slot_tags_vocab=slot_tags_vocab, intents_vocab=intents_vocab, bos_eos=args.input_bos_eos, intent_multi_class=args.intent_multi_class, slot_tag_enc_dec=args.enc_dec, device=args.device)
                    outputs = model(slot_tag_embeds, intent_embeds, inputs["inputs"], inputs["lengths"], inputs["input_mask"], slot_tags=inputs["tag_ids"], intents=inputs["intent_ids"], slot_tag_masked_output=None, intent_masked_output=None, slot_tag_to_id=slot_tags_vocab.label_to_id)
                    tag_loss, intent_loss = outputs[:2]
                    losses.append([tag_loss.item()/sum(inputs["lengths"]), intent_loss.item()/len(inputs["lengths"])])
                    total_loss = args.tr_st_weight * tag_loss + (1 - args.tr_st_weight) * intent_loss
                    
                    total_loss.backward()
                    # Clips gradient norm of an iterable of parameters.
                    if args.tr_optim.lower() != 'bertadam' and args.tr_max_norm > 0:
                        for group in optimizer_grouped_parameters:
                            torch.nn.utils.clip_grad_norm_(group['params'], args.tr_max_norm)
                    if args.tr_optim.lower() == 'adamw':
                        scheduler.step()
                    optimizer.step()
                
                if step % piece_steps == 0:
                    print('[learning] epoch %i >> %2.2f%%' % (i, (step+1)*100./batch_number), 'completed in %.2f (sec) <<\r' % (time.time()-start_time), end='')
                    sys.stdout.flush()
            print('')
            mean_loss = np.mean(losses, axis=0)
            logger.info('Training:\tEpoch : %d\tTime : %.4fs\tLoss of tag : %.4f\tLoss of class : %.4f' % (i, time.time() - start_time, mean_loss[0], mean_loss[1]))
            gc.collect()

            if hasattr(model, 'crf_transition_layer'):
                logger.info(model.crf.scaling_feats)
                logger.info(model.crf_transition_layer.crf_transitions_model)
            # Validation & Evaluation
            start_time = time.time()
            loss_val, slot_metrics_val, intent_metrics_val = decode(args, model, valid_dataset, os.path.join(exp_path, 'valid.iter'+str(i)))
            logger.info('Validation:\tEpoch : %d\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f' % (i, time.time() - start_time, loss_val[0], loss_val[1], slot_metrics_val['f'], intent_metrics_val['f']))
            start_time = time.time()
            loss_te, slot_metrics_te, intent_metrics_te = decode(args, model, test_dataset, os.path.join(exp_path, 'test.iter'+str(i)))
            logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f' % (i, time.time() - start_time, loss_te[0], loss_te[1], slot_metrics_te['f'], intent_metrics_te['f']))

            if args.task_sc:
                val_f1_score = (args.tr_st_weight * slot_metrics_val['f'] + (1 - args.tr_st_weight) * intent_metrics_val['f'])
            else:
                val_f1_score = slot_metrics_val['f']
            if best_f1 < val_f1_score:
                model.save_model(os.path.join(exp_path, args.model_save_path))
                best_f1 = val_f1_score
                logger.info('NEW BEST:\tEpoch : %d\tbest valid F1 : %.2f, cls-F1 : %.2f;\ttest F1 : %.2f, cls-F1 : %.2f' % (i, slot_metrics_val['f'], intent_metrics_val['f'], slot_metrics_te['f'], intent_metrics_te['f']))
                best_result['epoch'] = i
                best_result['vf1'], best_result['vcf1'], best_result['vce'] = slot_metrics_val['f'], intent_metrics_val['f'], loss_val
                best_result['tf1'], best_result['tcf1'], best_result['tce'] = slot_metrics_te['f'], intent_metrics_te['f'], loss_te
        logger.info('BEST RESULT: \tEpoch : %d\tbest valid (Loss: (%.2f, %.2f) F1 : %.2f cls-F1 : %.2f)\tbest test (Loss: (%.2f, %.2f) F1 : %.2f cls-F1 : %.2f)' % (best_result['epoch'], best_result['vce'][0], best_result['vce'][1], best_result['vf1'], best_result['vcf1'], best_result['tce'][0], best_result['tce'][1], best_result['tf1'], best_result['tcf1']))
        # finetune  # Validation & Evaluation
        del optimizer
        try:
            if args.test_finetune:
                model_saved_path = os.path.join(exp_path, args.model_save_path)
                #optim_name, lr, finetune_steps ='bertadam', 4e-5, (5, 10) ## it performs worse than the adam for finetuning
                optim_name, lr, lr_np, finetune_steps ='adam', 1e-5, 1e-3, (1, 3, 5) ## (5, 10)
                if args.matching_similarity_y == 'desc':
                    finetune_steps = (5, 10)
                for test_finetune_step in finetune_steps:
                    logger.info('----------optim_name=%s, step=%d, lr=%f, lr_np=%f' % (optim_name, test_finetune_step, lr, lr_np))
                    start_time = time.time()
                    loss_val, slot_metrics_val, intent_metrics_val = decode(args, model, valid_dataset, os.path.join(exp_path, 'valid.final.iter'+str(test_finetune_step)), test_finetune_step=test_finetune_step, model_ori_saved_path=model_saved_path, lr=lr, optim_name=optim_name, lr_np=lr_np)
                    logger.info('Validation:\tEpoch : %d\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f' % (i, time.time() - start_time, loss_val[0], loss_val[1], slot_metrics_val['f'], intent_metrics_val['f']))
                    start_time = time.time()
                    loss_te, slot_metrics_te, intent_metrics_te = decode(args, model, test_dataset, os.path.join(exp_path, 'test.final.iter'+str(test_finetune_step)), test_finetune_step=test_finetune_step, model_ori_saved_path=model_saved_path, lr=lr, optim_name=optim_name, lr_np=lr_np)
                    logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f' % (i, time.time() - start_time, loss_te[0], loss_te[1], slot_metrics_te['f'], intent_metrics_te['f']))
        except Exception as error:
            logger.info(str(error))
        finally:
            if args.model_removed == 'yes':
                os.remove(os.path.join(exp_path, args.model_save_path))
                print("Model is removed!")
    else:    
        logger.info("Testing starts at %s" % (time.asctime(time.localtime(time.time()))))
        start_time = time.time()
        loss_val, slot_metrics_val, intent_metrics_val = decode(args, model, valid_dataset, os.path.join(exp_path, 'valid.eval'))
        logger.info('Validation:\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f' % (time.time() - start_time, loss_val[0], loss_val[1], slot_metrics_val['f'], intent_metrics_val['f']))
        start_time = time.time()
        loss_te, slot_metrics_te, intent_metrics_te = decode(args, model, test_dataset, os.path.join(exp_path, 'test.eval'))
        logger.info('Evaluation:\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f' % (time.time() - start_time, loss_te[0], loss_te[1], slot_metrics_te['f'], intent_metrics_te['f']))
        if hasattr(model, 'crf_transition_layer'):
            logger.info(model.crf_transition_layer.crf_transitions_model)
        # finetune  # Validation & Evaluation
        if args.test_finetune:
            start_time = time.time()
            #optim_name, lr, finetune_steps ='bertadam', 4e-5, (5, 10) ## it performs worse than the adam for finetuning
            optim_name, lr, lr_np, finetune_steps ='adam', 1e-5, 1e-3, (1, 3, 5, 10)
            for test_finetune_step in finetune_steps:
                logger.info('----------optim_name=%s, step=%d, lr=%f, lr_np=%f' % (optim_name, test_finetune_step, lr, lr_np))
                #start_time = time.time()
                #loss_val, slot_metrics_val, intent_metrics_val = decode(args, model, valid_dataset, os.path.join(exp_path, 'valid.eval.iter'+str(test_finetune_step)), test_finetune_step=test_finetune_step, model_ori_saved_path=args.testing_model_read_path, lr=lr, optim_name=optim_name, lr_np=lr_np)
                #logger.info('Validation:\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f' % (time.time() - start_time, loss_val[0], loss_val[1], slot_metrics_val['f'], intent_metrics_val['f']))
                start_time = time.time()
                loss_te, slot_metrics_te, intent_metrics_te = decode(args, model, test_dataset, os.path.join(exp_path, 'test.eval.iter'+str(test_finetune_step)), test_finetune_step=test_finetune_step, model_ori_saved_path=args.testing_model_read_path, lr=lr, optim_name=optim_name, lr_np=lr_np)
                logger.info('Evaluation:\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f' % (time.time() - start_time, loss_te[0], loss_te[1], slot_metrics_te['f'], intent_metrics_te['f']))

if __name__ == "__main__":
    main()
