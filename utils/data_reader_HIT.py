#!/usr/bin/env python3

'''
@Time   : 2020-04-20 23:19:44
@Author : su.zhu
@Desc   : 
'''

import json

import torch
from torch.utils.data import Dataset

from utils.basic_vocab_reader import SLUOutputVocab
from utils.vocab_reader import FewShotSlotVocab

from utils.basic_data_reader import _get_intent_vector
from utils.pretrained_transformer import prepare_inputs_for_bert_xlnet

from utils.metric import get_chunks

class SlotIntentDataset_in_HIT_form(Dataset):
    ''' for multi-task learning '''
    def __init__(self, data_file, slot_tags_vocab, intents_vocab, MTL_vocab_share_type, lowercase=False, separator=':'):

        self.all_data_samples = self.read_seqtag_data_with_class(data_file, slot_tags_vocab, intents_vocab, MTL_vocab_share_type, lowercase=lowercase, separator=separator)

    @classmethod
    def read_seqtag_data_with_class(self, data_file, slot_tags_vocab, intents_vocab, MTL_vocab_share_type, lowercase=False, separator=':'):
        all_data_samples = []
        with open(data_file, 'r') as f:
            data = json.load(f)
            ind = 0
            for domain_name in data:
                for batch in data[domain_name]:
                    #print(list(batch['support'])) # ['seq_ins', 'labels', 'seq_outs', 'word_piece_marks', 'tokenized_texts', 'word_piece_labels']
                    #print(list(batch['batch'])) # ['seq_ins', 'labels', 'seq_outs', 'word_piece_marks', 'tokenized_texts', 'word_piece_labels']
                    supported_slot_tags = set()
                    for slots in batch['support']['seq_outs']:
                        supported_slot_tags |= set(slots)
                    supported_intents = set(batch['support']['labels'])
                    if MTL_vocab_share_type == 'just_O':
                        supported_slot_tags = {slot + '__' + domain_name if slot != 'O' else 'O' for slot in supported_slot_tags}
                    elif MTL_vocab_share_type == 'no':
                        supported_slot_tags = {slot + '__' + domain_name for slot in supported_slot_tags}
                    if MTL_vocab_share_type in {'just_O', 'no'}:
                        supported_intents = {intent + '__' + domain_name for intent in supported_intents}
                    masked_slot_tags = []
                    for idx in slot_tags_vocab.id_to_label:
                        label = slot_tags_vocab.id_to_label[idx]
                        if label in slot_tags_vocab.special_labels or label in supported_slot_tags:
                            pass
                        else:
                            masked_slot_tags.append(idx)
                    masked_slot_tags = torch.tensor(masked_slot_tags, dtype=torch.long) 
                    masked_intents = []
                    for idx in intents_vocab.id_to_label:
                        label = intents_vocab.id_to_label[idx]
                        if label in intents_vocab.special_labels or label in supported_intents:
                            pass
                        else:
                            masked_intents.append(idx)
                    masked_intents = torch.tensor(masked_intents, dtype=torch.long) 
                    data_batch_support_query = []
                    for data_type in ['support', 'batch']:
                        data_batch = []
                        batch_size = len(batch[data_type]['seq_ins'])
                        for i in range(batch_size):
                            in_seq = []
                            for word in batch[data_type]['seq_ins'][i]:
                                if lowercase:
                                    word = word.lower()
                                in_seq.append(word)
                            tag_seq = []
                            for slot in batch[data_type]['seq_outs'][i]:
                                if MTL_vocab_share_type == 'just_O':
                                    slot = slot + '__' + domain_name if slot != 'O' else 'O'
                                elif MTL_vocab_share_type == 'no':
                                    slot = slot + '__' + domain_name
                                tag_seq.append(slot)
                            class_name = batch[data_type]['labels'][i]
                            if MTL_vocab_share_type in {'just_O', 'no'}:
                                class_name = class_name + '__' + domain_name
                            data_batch.append([ind, in_seq, tag_seq, class_name])
                            ind += 1
                        data_batch_support_query.append(data_batch)
                    all_data_samples.append(data_batch_support_query + [masked_slot_tags, masked_intents])

        return all_data_samples

    def __len__(self):
        return len(self.all_data_samples)

    def __getitem__(self, idx):
        return self.all_data_samples[idx]

class FewShotSlotIntentDataset_in_HIT_form(Dataset):
    ''' for few-shot learning '''
    def __init__(self, data_file, slot_desc_file, slot_embedding_type='with_BIO', bert_tokenized=False, lowercase=False, separator=':', input_bos_eos=False):

        self.all_data_samples = self.read_seqtag_data_with_class(data_file, slot_desc_file, slot_embedding_type=slot_embedding_type, bert_tokenized=bert_tokenized, lowercase=lowercase, separator=separator, input_bos_eos=input_bos_eos)

    @classmethod
    def read_seqtag_data_with_class(self, data_file, slot_desc_file, slot_embedding_type='with_BIO', bert_tokenized=False, lowercase=False, separator=':', input_bos_eos=False):
        slot_to_desc = {}
        with open(slot_desc_file, 'r') as f:
            for line in f:
                slot, desc = line.strip('\n\r\t ').split(' : ')[:2]
                slot_to_desc[slot] = desc.split(' ')
        all_data_samples = []
        with open(data_file, 'r') as f:
            data = json.load(f)
            ind = 0
            for domain_name in data:
                for batch in data[domain_name]:
                    #print(list(batch['support'])) # ['seq_ins', 'labels', 'seq_outs', 'word_piece_marks', 'tokenized_texts', 'word_piece_labels']
                    #print(list(batch['batch'])) # ['seq_ins', 'labels', 'seq_outs', 'word_piece_marks', 'tokenized_texts', 'word_piece_labels']
                    if not bert_tokenized:
                        seq_ins_tag = 'seq_ins'
                        seq_out_tag = 'seq_outs'
                    else:
                        seq_ins_tag = 'tokenized_texts'
                        seq_out_tag = 'word_piece_labels'
                    ## get vocabulary
                    supported_slot_tags = set()
                    for slots in batch['support'][seq_out_tag]:
                        supported_slot_tags |= set(slots)
                    supported_intents = set(batch['support']['labels'])
                    supported_slot_tags = sorted(list(supported_slot_tags))
                    supported_intents = sorted(list(supported_intents))
                    slot_tags_vocab = FewShotSlotVocab(supported_slot_tags, slot_embedding_type=slot_embedding_type, bos_eos=input_bos_eos)
                    intents_vocab = SLUOutputVocab(supported_intents, no_special_labels=True)
                    slot_desc_in_words = slot_tags_vocab.get_label_description_for_HIT_data(slot_to_desc)
                    ## get data
                    data_batch_support_query = []
                    for data_type in ['support', 'batch']:
                        data_batch = []
                        batch_size = len(batch[data_type][seq_ins_tag])
                        for i in range(batch_size):
                            in_seq = []
                            for word in batch[data_type][seq_ins_tag][i]:
                                if lowercase:
                                    word = word.lower()
                                in_seq.append(word)
                            tag_seq = []
                            for slot in batch[data_type][seq_out_tag][i]:
                                tag_seq.append(slot)
                            class_name = batch[data_type]['labels'][i]
                            data_batch.append([ind, in_seq, tag_seq, class_name])
                            ind += 1
                        data_batch_support_query.append(data_batch)
                    all_data_samples.append(data_batch_support_query + [slot_tags_vocab, intents_vocab, slot_desc_in_words])

        return all_data_samples

    def __len__(self):
        return len(self.all_data_samples)

    def __getitem__(self, idx):
        return self.all_data_samples[idx]

def read_label_indicator_of_support_set(slot_tags_vocab, intents_vocab, support_tags, support_intents, indicator_type='PN', slot_embedding_type='with_BIO', device=None):
    '''
    indicator_type: (different ways to compute similarities between a hidden vector and each label embedding)
      1. PN (prototypical network);
      2. MN (matching network)
      3. NMN (normalized matching network)
    slot_embedding_type:
      1. with_BIO : all slot-tag embeddings are calculated from support set;
      2. with_BI : except that 'O' is ranomly initilized as an embedding vector with the same dimension;
      3. without_BIO : all slot embeddings are calculated from support set except for 'O', and 'B/I' are ranomly initilized as embedding vectors which will be added into slot embeddings to represent slot-tags.
    '''
    batch_size = len(support_tags)
    lengths = [len(seq) for seq in support_tags]
    max_length = max(lengths)
    if indicator_type == 'PN':
        if slot_embedding_type == 'with_BIO' or slot_embedding_type == 'with_BI':
            slot_label_indicator = []
            for tag in slot_tags_vocab.other_labels:
                weights = [0.0] * (batch_size * max_length)
                for i in range(batch_size):
                    for j, tag_2 in enumerate(support_tags[i]):
                        if tag_2 == tag:
                            weights[max_length * i + j] = 1.0
                slot_label_indicator.append(weights)
            slot_label_indicator = torch.tensor(slot_label_indicator, dtype=torch.float, device=device)
            slot_label_indicator = slot_label_indicator / slot_label_indicator.sum(dim=1, keepdim=True)
        else:
            slot_label_indicator = []
            for slot in slot_tags_vocab.other_labels_slot_names:
                weights = [0.0] * (batch_size * max_length)
                for i in range(batch_size):
                    chunks = get_chunks(['O'] + support_tags[i] + ['O'])
                    for (start, end, slot_2) in chunks:
                        start -= 1
                        if slot_2 == slot:
                            for j in range(start, end):
                                weights[max_length * i + j] = 1.0 / (end - start)
                slot_label_indicator.append(weights)
            slot_label_indicator = torch.tensor(slot_label_indicator, dtype=torch.float, device=device)
            slot_label_indicator = slot_label_indicator / slot_label_indicator.sum(dim=1, keepdim=True)
        
        ## for intent
        intent_label_indicator = []
        for intent in intents_vocab.label_to_id:
            weights = [0.0] * batch_size
            for i in range(batch_size):
                if type(support_intents[i]) == str:
                    if support_intents[i] == intent:
                        weights[i] = 1.0
                elif intent in support_intents[i]:
                    weights[i] = 1.0
            intent_label_indicator.append(weights)
        intent_label_indicator = torch.tensor(intent_label_indicator, dtype=torch.float, device=device)
        intent_label_indicator = intent_label_indicator / intent_label_indicator.sum(dim=1, keepdim=True)
        return slot_label_indicator, intent_label_indicator
    else:
        pass

def convert_examples_to_features(examples, tf_tokenizer, tf_input_args={}, slot_tags_vocab=None, intents_vocab=None, bos_eos=False, intent_multi_class=False, intent_separator=';', slot_tag_enc_dec=False, mask_padding_with_zero=True, device=None):
    # sort the batch in increasing order of sentence
    examples = sorted(examples, key=lambda x: len(x[1]), reverse=True)
    line_nums = [example[0] for example in examples]
    lengths = [len(example[1]) for example in examples]
    max_len = max(lengths)
    padding_lengths = [max_len - l for l in lengths]

    if bos_eos:
        bos_tag = slot_tags_vocab.bos_token
        eos_tag = slot_tags_vocab.eos_token
    if slot_tag_enc_dec:
        bos_tag = slot_tags_vocab.bos_token
    pad_tag_id = slot_tags_vocab.convert_tokens_to_ids(slot_tags_vocab.pad_token)
    
    input_word_ids = []
    tag_ids = []
    intent_ids = []
    input_mask = []
    batch_tokens, batch_tags, batch_intents = [], [], [] # used for evaluation
    for example_index, example in enumerate(examples):
        tokens = example[1]
        tags = example[2]
        intent = example[3]
        if bos_eos:
            tokens = ['<s>'] + tokens + ['</s>']
            tags = [bos_tag] + tags + [eos_tag]
            lengths[example_index] += 2
        mask_vector = [1 if mask_padding_with_zero else 0] * lengths[example_index]
        mask_vector += [0 if mask_padding_with_zero else 1] * padding_lengths[example_index]
        input_mask.append(mask_vector)
        
        batch_tokens.append(tokens)
        batch_tags.append(tags)
        if slot_tag_enc_dec:
            # used for training
            tags = [bos_tag] + tags
        tag_ids.append(slot_tags_vocab.convert_tokens_to_ids(tags) + [pad_tag_id] * padding_lengths[example_index])
        
        intents, intent_vector_or_id = _get_intent_vector(intent, intents_vocab, intent_multi_class=intent_multi_class, intent_separator=intent_separator)
        batch_intents.append(intents)
        intent_ids.append(intent_vector_or_id)
    
    sentences = [example[1] for example in examples]
    input_tf = prepare_inputs_for_bert_xlnet(sentences, tf_tokenizer, bos_eos=bos_eos, device=device, **tf_input_args)
    input_mask = torch.tensor(input_mask, dtype=torch.float, device=device)
    tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
    if intent_multi_class:
        intent_ids = torch.tensor(intent_ids, dtype=torch.float, device=device)
    else:
        intent_ids = torch.tensor(intent_ids, dtype=torch.long, device=device)

    return {
            "line_nums": line_nums,
            "tokens": batch_tokens,
            "tags": batch_tags,
            "intents": batch_intents,
            "inputs": {
                "input_tf": input_tf
            },
            "input_mask": input_mask,
            "tag_ids": tag_ids,
            "intent_ids": intent_ids,
            "lengths": lengths
            }

def concat_query_and_support_samples_for_pari_wise_embeddings(query_inputs, support_inputs, device=None):
    query_input_ids = query_inputs["inputs"]["input_tf"]["input_ids"]
    support_input_ids = support_inputs["inputs"]["input_tf"]["input_ids"][:, 1:] # no CLS
    query_attention_mask = query_inputs["inputs"]["input_tf"]["attention_mask"]
    support_attention_mask = support_inputs["inputs"]["input_tf"]["attention_mask"][:, 1:] # no CLS
    query_lengths_of_tokens = query_inputs["inputs"]["input_tf"]["lengths"]
    #for x in (query_input_ids, support_input_ids, query_attention_mask, support_attention_mask):
    #    print(x, x.size())
    #print(query_lengths_of_tokens)
    B1, L1 = query_input_ids.size()
    B2, L2 = support_input_ids.size()
    concat_input_ids = torch.cat((query_input_ids[:, None, :].expand(-1, B2, -1).reshape(B1 * B2, L1), support_input_ids[None, :, :].expand(B1, -1, -1).reshape(B1 * B2, L2)), dim=1)
    concat_segment_ids = torch.tensor([[0] * L1 + [1] * L2] * (B1 * B2), dtype=torch.long, device=device)
    query_pos_ids = torch.arange(L1, dtype=torch.long, device=device)[None, :].expand(B1 * B2, -1)
    support_pos_ids = torch.arange(L2, dtype=torch.long, device=device)[None, None, :].expand(B1, B2, -1)
    support_start_pos = torch.tensor(query_lengths_of_tokens, dtype=torch.long, device=device)
    support_pos_ids = support_start_pos[:, None, None] + support_pos_ids
    support_pos_ids = support_pos_ids.reshape(B1 * B2, -1)
    concat_position_ids = torch.cat((query_pos_ids, support_pos_ids), dim=1)
    concat_attention_mask = torch.cat((query_attention_mask[:, None, :].repeat(1, B2, 1).view(B1 * B2, L1), support_attention_mask[None, :, :].repeat(B1, 1, 1).view(B1 * B2, L2)), dim=1)
    ret =  {
            "B1": B1,
            "B2": B2,
            "L1": L1,
            "L2": L2,
            "input_ids": concat_input_ids,
            "segment_ids": concat_segment_ids,
            "position_ids": concat_position_ids,
            "attention_mask": concat_attention_mask,
            "query": {
                "attention_mask": query_inputs["inputs"]["input_tf"]["attention_mask"],
                "gather_index": query_inputs["inputs"]["input_tf"]["gather_index"]
                },
            "support": {
                "attention_mask": support_inputs["inputs"]["input_tf"]["attention_mask"],
                "gather_index": support_inputs["inputs"]["input_tf"]["gather_index"]
                }
            }
    return {
            "inputs": {
                "input_tf": ret
                }
            }
