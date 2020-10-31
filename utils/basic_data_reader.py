#!/usr/bin/env python3

'''
@Time   : 2019-08-11 17:57:13
@Author : su.zhu
@Desc   : 
'''

import torch
from torch.utils.data import Dataset
import itertools

class SlotIntentDataset(Dataset):

    def __init__(self, data_file, lowercase=False, separator=':', lattice_used=None):

        self.all_data_samples = self.read_seqtag_data_with_class(data_file, lowercase=lowercase, separator=separator, lattice_used=lattice_used)

    @classmethod
    def read_seqtag_data_with_class(self, data_file, lowercase=False, separator=':', lattice_used=None):
        all_data_samples = []
        with open(data_file, 'r') as f:
            for ind, line in enumerate(f):
                slot_tag_line, class_name = line.strip('\n\r').split(' <=> ')
                if slot_tag_line == "":
                    continue
                in_seq, tag_seq = [], []
                for item in slot_tag_line.split(' '):
                    tmp = item.split(separator)
                    assert len(tmp) >= 2
                    word, tag = separator.join(tmp[:-1]), tmp[-1]
                    if lowercase:
                        word = word.lower()
                    in_seq.append(word)
                    tag_seq.append(tag)

                all_data_samples.append([
                    ind,
                    in_seq,
                    tag_seq,
                    class_name,
                    lattice_used.get_lattice_inputs(in_seq) if lattice_used else None
                    ])
        return all_data_samples

    def __len__(self):
        return len(self.all_data_samples)

    def __getitem__(self, idx):
        return self.all_data_samples[idx]

def collate_fn_do_nothing(batch):
    return batch

def convert_examples_to_features_for_word_input(examples, tokenizer, slot_tags_vocab, intents_vocab, bos_eos=False, intent_multi_class=False, intent_separator=';', slot_tag_enc_dec=False, mask_padding_with_zero=True, device=None):
    if bos_eos:
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        bos_tag = slot_tags_vocab.bos_token
        eos_tag = slot_tags_vocab.eos_token
    if slot_tag_enc_dec:
        bos_tag = slot_tags_vocab.bos_token
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    pad_tag_id = slot_tags_vocab.convert_tokens_to_ids(slot_tags_vocab.pad_token)
    
    # sort the batch in increasing order of sentence
    examples = sorted(examples, key=lambda x: len(x[1]), reverse=True)

    line_nums = [example[0] for example in examples]
    
    lengths = [len(example[1]) for example in examples]
    max_len = max(lengths)
    padding_lengths = [max_len - l for l in lengths]
    
    input_ids = []
    tag_ids = []
    intent_ids = []
    input_mask = []
    batch_tokens, batch_tags, batch_intents = [], [], []
    for example_index, example in enumerate(examples):
        tokens = example[1]
        tags = example[2]
        intent = example[3]
        if bos_eos:
            tokens = [bos_token] + tokens + [eos_token]
            tags = [bos_tag] + tags + [eos_tag]
            lengths[example_index] += 2
        mask_vector = [1 if mask_padding_with_zero else 0] * len(tokens)
        batch_tokens.append(tokens)
        batch_tags.append(tags)
        
        input_ids.append(tokenizer.convert_tokens_to_ids(tokens) + [pad_token_id] * padding_lengths[example_index])
        if slot_tag_enc_dec:
            # used for training
            tags = [bos_tag] + tags
        tag_ids.append(slot_tags_vocab.convert_tokens_to_ids(tags) + [pad_tag_id] * padding_lengths[example_index])
        
        if intent_multi_class:
            intents = intent.split(intent_separator)
            batch_intents.append(intents)
            intents_vector = [0] * intents_vocab.get_vocab_size()
            for intent_id in intents_vocab.convert_tokens_to_ids(intents):
                intents_vector[intent_id] = 1
            intent_ids.append(intents_vector)
        else:
            intents = intent.split(intent_separator)
            batch_intents.append(intents)
            intent_ids.append(intents_vocab.convert_tokens_to_ids(intents[0])) # we use the first intent to train a softmax classifier

        mask_vector += [0 if mask_padding_with_zero else 1] * padding_lengths[example_index]
        input_mask.append(mask_vector)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
    if intent_multi_class:
        intent_ids = torch.tensor(intent_ids, dtype=torch.float, device=device)
    else:
        intent_ids = torch.tensor(intent_ids, dtype=torch.long, device=device)
    input_mask = torch.tensor(input_mask, dtype=torch.float, device=device)

    return {
            "line_nums": line_nums,
            "tokens": batch_tokens,
            "tags": batch_tags,
            "intents": batch_intents,
            "input_ids": input_ids,
            "tag_ids": tag_ids,
            "intent_ids": intent_ids,
            "input_mask": input_mask,
            "lengths": lengths
            }

def _get_intent_vector(intent, intents_vocab, intent_multi_class=False, intent_separator=';'):
    if intent_multi_class:
        intents = intent.split(intent_separator)
        intents_vector = [0] * intents_vocab.get_vocab_size()
        for intent_id in intents_vocab.convert_tokens_to_ids(intents):
            intents_vector[intent_id] = 1
        return intents, intents_vector
    else:
        intents = intent.split(intent_separator)
        return intents, intents_vocab.convert_tokens_to_ids(intents[0]) # we use the first intent to train a softmax classifier

def get_char_features(sorted_examples, char_tokenizer, padding_lengths, bos_eos=False, device=None, feed_transformer=False):
    if bos_eos:
        bos_char_id = char_tokenizer.convert_tokens_to_ids(char_tokenizer.bos_token)
        eos_char_id = char_tokenizer.convert_tokens_to_ids(char_tokenizer.eos_token)
    if feed_transformer:
        cls_char_id = char_tokenizer.convert_tokens_to_ids(char_tokenizer.cls_token)
    pad_char_id = char_tokenizer.convert_tokens_to_ids(char_tokenizer.pad_token)
    input_char_ids = []
    for example_index, example in enumerate(sorted_examples):
        tokens = example[1]
        char_ids = []
        for token in tokens:
            chars = tuple(token)
            char_ids.append(char_tokenizer.convert_tokens_to_ids(chars))
        if bos_eos:
            char_ids_of_this_sentence = [[bos_char_id]] + char_ids + [[eos_char_id]] + [[pad_char_id]] * padding_lengths[example_index]
        else:
            char_ids_of_this_sentence = char_ids + [[pad_char_id]] * padding_lengths[example_index]
        if feed_transformer:
            char_ids_of_this_sentence = [[cls_char_id]] + char_ids_of_this_sentence
        input_char_ids += char_ids_of_this_sentence
    tmp = sorted(enumerate(input_char_ids), key=lambda x: len(x[1]), reverse=True)
    reverse_index = [0] * len(tmp)
    for idx,x in enumerate(tmp):
        reverse_index[x[0]] = idx
    reverse_index = torch.tensor(reverse_index, dtype=torch.long, device=device)
    len_word_chars = [len(char_ids) for _,char_ids in tmp]
    max_len_word_chars = max(len_word_chars)
    input_char_ids = [char_ids + [pad_char_id] * (max_len_word_chars - len(char_ids)) for _,char_ids in tmp]
    input_char_ids = torch.tensor(input_char_ids, dtype=torch.long, device=device)
    return input_char_ids, reverse_index, len_word_chars

def prepare_inputs_for_bert_xlnet(sorted_examples, tokenizer, bos_eos=False, cls_token_at_end=False, pad_on_left=False, pad_token=0, sequence_a_segment_id=0, cls_token_segment_id=1, pad_token_segment_id=0, device=None, feed_transformer=False):
    """
    TODO: if feed_transformer == True, select CLS output as the first embedding of cls_token
    """
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    """ output: {
        'tokens': tokens_tensor,        # input_ids
        'segments': segments_tensor,    # token_type_ids
        'mask': input_mask,             # attention_mask
        'selects': selects_tensor,      # original_word_to_token_position
        'copies': copies_tensor         # original_word_position
        }
    """
    ## sentences are sorted by sentence length
    cls_token = tokenizer.cls_token # [CLS]
    sep_token = tokenizer.sep_token # [SEP]
    if bos_eos:
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
    
    word_lengths = []
    tokens = []
    segment_ids = []
    selected_indexes = []
    start_pos = 0
    for example_index, example in enumerate(sorted_examples):
        words = example[1]
        if bos_eos:
            words = [bos_token] + words + [eos_token]
        word_lengths.append(len(words))
        selected_index = []
        ts = []
        for w in words:
            if cls_token_at_end:
                selected_index.append(len(ts))
            else:
                selected_index.append(len(ts) + 1)
            ts += tokenizer.tokenize(w)
        ts += [sep_token]
        si = [sequence_a_segment_id] * len(ts)
        if cls_token_at_end:
            ts = ts + [cls_token]
            si = si + [cls_token_segment_id]
        else:
            ts = [cls_token] + ts
            si = [cls_token_segment_id] + si
        tokens.append(ts)
        segment_ids.append(si)
        selected_indexes.append(selected_index)
    lengths_of_tokens = [len(tokenized_text) for tokenized_text in tokens]
    max_length_of_tokens = max(lengths_of_tokens)
    #if not cls_token_at_end: # bert
    #    assert max_length_of_tokens <= model_bert.config.max_position_embeddings
    padding_lengths = [max_length_of_tokens - len(tokenized_text) for tokenized_text in tokens]
    if pad_on_left:
        input_mask = [[0] * padding_lengths[idx] + [1] * len(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [[pad_token] * padding_lengths[idx] + tokenizer.convert_tokens_to_ids(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        segments_ids = [[pad_token_segment_id] * padding_lengths[idx] + si for idx,si in enumerate(segment_ids)]
        selected_indexes = [[padding_lengths[idx] + i + idx * max_length_of_tokens for i in selected_index] for idx,selected_index in enumerate(selected_indexes)]
    else:
        input_mask = [[1] * len(tokenized_text) + [0] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_text) + [pad_token] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        segments_ids = [si + [pad_token_segment_id] * padding_lengths[idx] for idx,si in enumerate(segment_ids)]
        selected_indexes = [[0 + i + idx * max_length_of_tokens for i in selected_index] for idx,selected_index in enumerate(selected_indexes)]
    max_length_of_sentences = max(word_lengths) # the length is already +2 when bos_eos is True.
    copied_indexes = [[i + idx * max_length_of_sentences for i in range(length)] for idx,length in enumerate(word_lengths)]

    input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long, device=device)
    segments_tensor = torch.tensor(segments_ids, dtype=torch.long, device=device)
    selects_tensor = torch.tensor(list(itertools.chain.from_iterable(selected_indexes)), dtype=torch.long, device=device)
    copies_tensor = torch.tensor(list(itertools.chain.from_iterable(copied_indexes)), dtype=torch.long, device=device)
    #return {'tokens': tokens_tensor, 'segments': segments_tensor, 'selects': selects_tensor, 'copies': copies_tensor, 'mask': input_mask}
    return tokens_tensor, segments_tensor, input_mask, selects_tensor, copies_tensor, lengths_of_tokens

def convert_examples_to_features(examples, word_tokenizer=None, char_tokenizer=None, tf_tokenizer=None, tf_input_args={}, slot_tags_vocab=None, intents_vocab=None, bos_eos=False, intent_multi_class=False, intent_separator=';', slot_tag_enc_dec=False, mask_padding_with_zero=True, device=None, feed_transformer=False, lattice_used=None):
    # sort the batch in increasing order of sentence
    examples = sorted(examples, key=lambda x: len(x[1]), reverse=True)
    line_nums = [example[0] for example in examples]
    lengths = [len(example[1]) for example in examples]
    max_len = max(lengths)
    padding_lengths = [max_len - l for l in lengths]

    if bos_eos:
        if word_tokenizer is not None:
            bos_token = word_tokenizer.bos_token
            eos_token = word_tokenizer.eos_token
        bos_tag = slot_tags_vocab.bos_token
        eos_tag = slot_tags_vocab.eos_token
    if slot_tag_enc_dec:
        bos_tag = slot_tags_vocab.bos_token
    if word_tokenizer is not None:
        pad_token_id = word_tokenizer.convert_tokens_to_ids(word_tokenizer.pad_token)
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
            if word_tokenizer is not None:
                tokens = [bos_token] + tokens + [eos_token]
            else:
                tokens = ['<s>'] + tokens + ['</s>']
            tags = [bos_tag] + tags + [eos_tag]
            lengths[example_index] += 2
        if feed_transformer == True:
            tokens = ['<cls>'] + tokens
            lengths[example_index] += 1
        mask_vector = [1 if mask_padding_with_zero else 0] * lengths[example_index]
        mask_vector += [0 if mask_padding_with_zero else 1] * padding_lengths[example_index]
        input_mask.append(mask_vector)
        
        if word_tokenizer is not None:
            input_word_ids.append(word_tokenizer.convert_tokens_to_ids(tokens) + [pad_token_id] * padding_lengths[example_index])
        batch_tokens.append(tokens)

        batch_tags.append(tags)
        if slot_tag_enc_dec:
            # used for training
            tags = [bos_tag] + tags
        tag_ids.append(slot_tags_vocab.convert_tokens_to_ids(tags) + [pad_tag_id] * padding_lengths[example_index])
        
        intents, intent_vector_or_id = _get_intent_vector(intent, intents_vocab, intent_multi_class=intent_multi_class, intent_separator=intent_separator)
        batch_intents.append(intents)
        intent_ids.append(intent_vector_or_id)

    input_word_ids = torch.tensor(input_word_ids, dtype=torch.long, device=device)
    if char_tokenizer is not None:
        input_char_ids, reverse_index, len_word_chars = get_char_features(examples, char_tokenizer, padding_lengths, bos_eos=bos_eos, device=device, feed_transformer=feed_transformer)
    else:
        input_char_ids, len_word_chars, reverse_index = None, None, None
    if tf_tokenizer is not None:
        input_tf_ids, tf_segment_ids, tf_attention_mask, tf_output_selects, tf_output_copies, lengths_of_tokens = prepare_inputs_for_bert_xlnet(examples, tf_tokenizer, bos_eos=bos_eos, device=device, feed_transformer=feed_transformer, **tf_input_args)
    else:
        input_tf_ids, tf_segment_ids, tf_attention_mask, tf_output_selects, tf_output_copies, lengths_of_tokens = None, None, None, None, None, None
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
                "input_word": input_word_ids,
                "input_char": {
                    "input_char_ids": input_char_ids, 
                    "lengths": len_word_chars,
                    "inverse_mapping_index": reverse_index,
                    "batch_size": len(lengths)
                    },
                "input_tf": {
                    "input_ids": input_tf_ids,
                    "segment_ids": tf_segment_ids,
                    "attention_mask": tf_attention_mask,
                    "selects": tf_output_selects,
                    "copies": tf_output_copies,
                    "batch_size": len(lengths),
                    "lengths": lengths_of_tokens,
                    "max_word_length": max(lengths)
                    }
            },
            "input_mask": input_mask,
            "tag_ids": tag_ids,
            "intent_ids": intent_ids,
            "lengths": lengths,
            "lattice": lattice_used.get_lattice_batch([ex[-1] for ex in examples], device=device) if lattice_used else None
            }

def prepare_inputs_of_word_sequences_for_bert_xlnet(word_sequences, tokenizer, bos_eos=False, cls_token_at_end=False, pad_on_left=False, pad_token=0, sequence_a_segment_id=0, cls_token_segment_id=1, pad_token_segment_id=0, device=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    """ output: {
        'tokens': tokens_tensor,        # input_ids
        'segments': segments_tensor,    # token_type_ids
        'mask': input_mask,             # attention_mask
        }
    """
    ## sentences are sorted by sentence length
    cls_token = tokenizer.cls_token # [CLS]
    sep_token = tokenizer.sep_token # [SEP]
    if bos_eos:
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
    
    tokens = []
    segment_ids = []
    for example_index, words in enumerate(word_sequences):
        if bos_eos:
            words = [bos_token] + words + [eos_token]
        ts = []
        for w in words:
            ts += tokenizer.tokenize(w)
        ts += [sep_token]
        si = [sequence_a_segment_id] * len(ts)
        if cls_token_at_end:
            ts = ts + [cls_token]
            si = si + [cls_token_segment_id]
        else:
            ts = [cls_token] + ts
            si = [cls_token_segment_id] + si
        tokens.append(ts)
        segment_ids.append(si)
    max_length_of_tokens = max([len(tokenized_text) for tokenized_text in tokens])
    #if not cls_token_at_end: # bert
    #    assert max_length_of_tokens <= model_bert.config.max_position_embeddings
    padding_lengths = [max_length_of_tokens - len(tokenized_text) for tokenized_text in tokens]
    if pad_on_left:
        input_mask = [[0] * padding_lengths[idx] + [1] * len(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [[pad_token] * padding_lengths[idx] + tokenizer.convert_tokens_to_ids(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        segments_ids = [[pad_token_segment_id] * padding_lengths[idx] + si for idx,si in enumerate(segment_ids)]
    else:
        input_mask = [[1] * len(tokenized_text) + [0] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_text) + [pad_token] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        segments_ids = [si + [pad_token_segment_id] * padding_lengths[idx] for idx,si in enumerate(segment_ids)]

    input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long, device=device)
    segments_tensor = torch.tensor(segments_ids, dtype=torch.long, device=device)
    return {'input_ids': tokens_tensor, 'token_type_ids': segments_tensor, 'attention_mask': input_mask}

def prepare_inputs_of_word_sequences_for_rnn(sentences, word_tokenizer=None, char_tokenizer=None, tf_tokenizer=None, tf_input_args={}, slot_tags_vocab=None, intents_vocab=None, bos_eos=False, intent_multi_class=False, intent_separator=';', slot_tag_enc_dec=False, mask_padding_with_zero=True, device=None, feed_transformer=False):
    # sort the batch in increasing order of sentence
    sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
    lengths = [len(snt) for snt in sentences]
    max_len = max(lengths)
    padding_lengths = [max_len - l for l in lengths]

    if bos_eos:
        if word_tokenizer is not None:
            bos_token = word_tokenizer.bos_token
            eos_token = word_tokenizer.eos_token
    if word_tokenizer is not None:
        pad_token_id = word_tokenizer.convert_tokens_to_ids(word_tokenizer.pad_token)
    
    input_word_ids = []
    input_mask = []
    batch_tokens = [] # used for evaluation
    for example_index, example in enumerate(sentences):
        tokens = example
        if bos_eos:
            if word_tokenizer is not None:
                tokens = [bos_token] + tokens + [eos_token]
            else:
                tokens = ['<s>'] + tokens + ['</s>']
            lengths[example_index] += 2
        if feed_transformer == True:
            tokens = ['<cls>'] + tokens
            lengths[example_index] += 1
        mask_vector = [1 if mask_padding_with_zero else 0] * lengths[example_index]
        mask_vector += [0 if mask_padding_with_zero else 1] * padding_lengths[example_index]
        input_mask.append(mask_vector)
        
        if word_tokenizer is not None:
            input_word_ids.append(word_tokenizer.convert_tokens_to_ids(tokens) + [pad_token_id] * padding_lengths[example_index])
        batch_tokens.append(tokens)
        
    input_word_ids = torch.tensor(input_word_ids, dtype=torch.long, device=device)
    '''
    if char_tokenizer is not None:
        input_char_ids, reverse_index, len_word_chars = get_char_features(examples, char_tokenizer, padding_lengths, bos_eos=bos_eos, device=device, feed_transformer=feed_transformer)
    else:
        input_char_ids, len_word_chars, reverse_index = None, None, None
    if tf_tokenizer is not None:
        input_tf_ids, tf_segment_ids, tf_attention_mask, tf_output_selects, tf_output_copies, lengths_of_tokens = prepare_inputs_for_bert_xlnet(examples, tf_tokenizer, bos_eos=bos_eos, device=device, feed_transformer=feed_transformer, **tf_input_args)
    else:
        input_tf_ids, tf_segment_ids, tf_attention_mask, tf_output_selects, tf_output_copies, lengths_of_tokens = None, None, None, None, None, None
    '''
    input_mask = torch.tensor(input_mask, dtype=torch.float, device=device)

    return {
            #"tokens": batch_tokens,
            "inputs": {
                "input_word": input_word_ids,
            },
            "input_mask": input_mask,
            "lengths": lengths
            }
