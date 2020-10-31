#!/usr/bin/env python3

'''
@Time   : 2019-08-20 14:11:56
@Author : su.zhu
@Desc   : 
'''

import torch
from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel, ElectraTokenizer, ElectraModel

MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer),
        'xlnet': (XLNetModel, XLNetTokenizer),
        'electra': (ElectraModel, ElectraTokenizer),
        'roberta_chinese': (BertModel, BertTokenizer)
        }

def load_pretrained_transformer(model_type, model_name):
    pretrained_model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_name)
    print('tokenizer.basic_tokenizer.do_lower_case:', tokenizer.basic_tokenizer.do_lower_case)
    #pretrained_model = pretrained_model_class.from_pretrained(model_name, output_hidden_states = True)
    pretrained_model = pretrained_model_class.from_pretrained(model_name)
    def add_no_init_flag(module):
        module.sz128__no_init_flag = True
    pretrained_model.apply(add_no_init_flag)
    print('pretrained_model.config:', pretrained_model.config)
    return tokenizer, pretrained_model

def load_pretrained_transformer_full(model_type, model_name):
    pretrained_model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_name)
    print('tokenizer.basic_tokenizer.do_lower_case:', tokenizer.basic_tokenizer.do_lower_case)
    #pretrained_model = pretrained_model_class.from_pretrained(model_name, output_hidden_states = True)
    pretrained_model = pretrained_model_class.from_pretrained(model_name)
    def add_no_init_flag(module):
        module.sz128__no_init_flag = True
    pretrained_model.apply(add_no_init_flag)
    print('pretrained_model.config:', pretrained_model.config)
    tf_input_args = {
            'cls_token_at_end': bool(model_type in ['xlnet']),  # xlnet has a cls token at the end
            'cls_token_segment_id': 2 if model_type in ['xlnet'] else 0,
            'pad_on_left': bool(model_type in ['xlnet']), # pad on the left for xlnet
            'pad_token_segment_id': 4 if model_type in ['xlnet'] else 0,
            }
    return tokenizer, pretrained_model, tf_input_args

def transformer_forward_by_ignoring_suffix(transformer, batch_size, max_word_length, input_ids, segment_ids, selects, copies, attention_mask, device=None, get_pooled_output=False):
    '''
    Ignore hidden states of all suffixes: [CLS] from ... to de ##n ##ver [SEP] => from ... to de
    '''
    outputs = transformer(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
    pretrained_top_hiddens = outputs[0]
    if get_pooled_output:
        pooled_output = outputs[1]
    batch_size, pretrained_seq_length, hidden_size = pretrained_top_hiddens.size(0), pretrained_top_hiddens.size(1), pretrained_top_hiddens.size(2)
    chosen_encoder_hiddens = pretrained_top_hiddens.view(-1, hidden_size).index_select(0, selects)
    embeds = torch.zeros(batch_size * max_word_length, hidden_size, device=device)
    embeds = embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(batch_size, max_word_length, -1)
    if get_pooled_output:
        return embeds, pooled_output
    else:
        return embeds

def prepare_inputs_for_bert_xlnet(input_sentences, tokenizer, bos_eos=False, cls_token_at_end=False, pad_on_left=False, pad_token=0, sequence_a_segment_id=0, cls_token_segment_id=1, pad_token_segment_id=0, device=None, feed_transformer=False, alignment='first'):
    """
    NOTE: If you want to feed bert/xlnet embeddings into RNN/GRU/LSTM by using pack_padded_sequence, you'd better sort input sentences in advance.
    """
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
        'gather_index': gather_index,      # gather_index
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
    if alignment == 'first':
        selected_indexes = []
    elif alignment == 'avg':
        remove_cls_seq_gather_indexes = [] # for aggregation; first, remove cls and seq
        aggregated_indexes = [] # for aggregation; after removing cls and seq
        aggregated_counts = [] # for aggregation; after removing cls and seq
    elif alignment == 'ori':
        remove_cls_seq_gather_indexes = [] # remove cls and seq
    output_tokens = []
    start_pos = 0
    for snt_idx, words in enumerate(input_sentences):
        if bos_eos:
            words = [bos_token] + words + [eos_token]
        word_lengths.append(len(words))
        if alignment == 'first':
            selected_index = []
        elif alignment == 'avg':
            aggregated_index = []
            aggregated_count = []
        ts_1 = []
        for w_idx, w in enumerate(words):
            _toks = tokenizer.tokenize(w)
            if alignment == 'first':
                if cls_token_at_end:
                    selected_index.append(len(ts_1))
                else:
                    selected_index.append(len(ts_1) + 1)
            elif alignment == 'avg':
                aggregated_index += [w_idx] * len(_toks)
                aggregated_count.append(len(_toks))
            ts_1 += _toks
        if alignment in {'first', 'avg'}:
            output_tokens.append(words)
        elif alignment == 'ori':
            output_tokens.append(ts_1)
        si = [sequence_a_segment_id] * len(ts_1)
        if cls_token_at_end:
            ts = ts_1 + [sep_token, cls_token]
            si = si + [sequence_a_segment_id, cls_token_segment_id]
        else:
            ts = [cls_token] + ts_1 + [sep_token]
            si = [cls_token_segment_id] + si + [sequence_a_segment_id]
        tokens.append(ts)
        #print(ts)
        segment_ids.append(si)
        if alignment == 'first':
            selected_indexes.append(selected_index)
        elif alignment == 'avg':
            if cls_token_at_end:
                remove_cls_seq_gather_indexes.append(list(range(len(ts) - 2))) # ..... [SEP] [CLS]
            else:
                remove_cls_seq_gather_indexes.append([i + 1 for i in range(len(ts) - 2)]) # [CLS] ..... [SEP]
            aggregated_indexes.append(aggregated_index)
            aggregated_counts.append(aggregated_count)
        elif alignment == 'ori':
            if cls_token_at_end:
                remove_cls_seq_gather_indexes.append(list(range(len(ts) - 2))) # ..... [SEP] [CLS]
            else:
                remove_cls_seq_gather_indexes.append([i + 1 for i in range(len(ts) - 2)]) # [CLS] ..... [SEP]
    
    max_length_of_sentences = max(word_lengths) # the length is already +2 when bos_eos is True.
    token_lengths = [len(tokenized_text) for tokenized_text in tokens]
    max_length_of_tokens = max(token_lengths)
    #if not cls_token_at_end: # bert
    #    assert max_length_of_tokens <= model_bert.config.max_position_embeddings
    padding_lengths = [max_length_of_tokens - len(tokenized_text) for tokenized_text in tokens]
    if pad_on_left:
        input_mask = [[0] * padding_lengths[idx] + [1] * len(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [[pad_token] * padding_lengths[idx] + tokenizer.convert_tokens_to_ids(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        segments_ids = [[pad_token_segment_id] * padding_lengths[idx] + si for idx,si in enumerate(segment_ids)]
        ## word embeddings will always pad on the right size!
        if alignment == 'first':
            #gather_indexes = [[0] * (max_length_of_sentences - word_lengths[idx]) + [padding_lengths[idx] + i for i in selected_index] for idx,selected_index in enumerate(selected_indexes)]
            gather_indexes = [[padding_lengths[idx] + i for i in selected_index] + [0] * (max_length_of_sentences - word_lengths[idx]) for idx,selected_index in enumerate(selected_indexes)]
        elif alignment in {'ori', 'avg'}:
            #remove_cls_seq_gather_indexes = [[0] * padding_lengths[idx] + [padding_lengths[idx] + i for i in remove_cls_seq_index] for idx,remove_cls_seq_index in enumerate(remove_cls_seq_gather_indexes)]
            remove_cls_seq_gather_indexes = [[padding_lengths[idx] + i for i in remove_cls_seq_index] + [0] * padding_lengths[idx] for idx,remove_cls_seq_index in enumerate(remove_cls_seq_gather_indexes)]
    else:
        input_mask = [[1] * len(tokenized_text) + [0] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_text) + [pad_token] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        segments_ids = [si + [pad_token_segment_id] * padding_lengths[idx] for idx,si in enumerate(segment_ids)]
        if alignment == 'first':
            gather_indexes = [selected_index + [max_length_of_tokens - 1] * (max_length_of_sentences - word_lengths[idx]) for idx,selected_index in enumerate(selected_indexes)]
        elif alignment in {'ori', 'avg'}:
            remove_cls_seq_gather_indexes = [remove_cls_seq_index + [max_length_of_tokens - 1] * padding_lengths[idx] for idx,remove_cls_seq_index in enumerate(remove_cls_seq_gather_indexes)]
    if alignment == 'avg':
        ## output of hiddens should be masked to zero in positions of padding
        aggregated_indexes = [aggregated_index + [aggregated_index[-1]] * padding_lengths[idx] for idx,aggregated_index in enumerate(aggregated_indexes)]
        aggregated_counts = [aggregated_count + [1] * (max_length_of_sentences - word_lengths[idx]) for idx,aggregated_count in enumerate(aggregated_counts)]

    input_mask = torch.tensor(input_mask, dtype=torch.float, device=device)
    tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long, device=device)
    segments_tensor = torch.tensor(segments_ids, dtype=torch.long, device=device)
    output = {
            "input_ids": tokens_tensor,
            "segment_ids": segments_tensor,
            "attention_mask": input_mask
            }
    output["lengths"] = [len(seq) for seq in output_tokens]
    if alignment == 'first':
        gather_indexes = torch.tensor(gather_indexes, dtype=torch.long, device=device)
        output["gather_index"] = gather_indexes
    elif alignment == 'avg':
        remove_cls_seq_gather_indexes = torch.tensor(remove_cls_seq_gather_indexes, dtype=torch.long, device=device)
        aggregated_indexes = torch.tensor(aggregated_indexes, dtype=torch.long, device=device)
        aggregated_counts = torch.tensor(aggregated_counts, dtype=torch.float, device=device)
        output["remove_cls_seq_gather_index"] = remove_cls_seq_gather_indexes
        output["aggregated_index"] = aggregated_indexes
        output["aggregated_count"] = aggregated_counts
        output["max_word_seq_length"] = max_length_of_sentences
    elif alignment == 'ori':
        remove_cls_seq_gather_indexes = torch.tensor(remove_cls_seq_gather_indexes, dtype=torch.long, device=device)
        output["remove_cls_seq_gather_index"] = remove_cls_seq_gather_indexes

    return output #, tokens, output_tokens, [len(seq) for seq in output_to]

def transformer_forward_by_ignoring_suffix_2(transformer, input_ids, segment_ids, attention_mask, gather_index=None, remove_cls_seq_gather_index=None, aggregated_index=None, aggregated_count=None, max_word_seq_length=None, device=None, alignment='first'):
    '''
    ['first', 'avg']: Ignore hidden states of all suffixes: [CLS] from ... to de ##n ##ver [SEP] => from ... to de
    ['ori']: Ignore hidden states of [CLS] and [SEP]: [CLS] from ... to de ##n ##ver [SEP] => from ... to de ##n ##ver
    !!! and padding on the right side !!!
    '''
    
    #assert alignment in {'ori', 'first', 'avg', None}
    
    outputs = transformer(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
    pretrained_top_hiddens = outputs[0]
    #print(pretrained_top_hiddens[-1])
    batch_size, token_seq_length, hidden_size = pretrained_top_hiddens.shape
    pretrained_top_hiddens.masked_fill_((1 - attention_mask).to(torch.bool)[:, :, None], 0)
    
    if alignment == 'first':
        embeds = torch.gather(pretrained_top_hiddens, 1, gather_index[:, :, None].expand(-1, -1, hidden_size)) # expand does not allocate new memory
    elif alignment == 'avg':
        pretrained_top_hiddens_without_cls_seq = torch.gather(pretrained_top_hiddens, 1, remove_cls_seq_gather_index[:, :, None].expand(-1, -1, hidden_size))

        aggregated_embeds = torch.zeros(batch_size, max_word_seq_length, hidden_size, device=device)
        aggregated_embeds.scatter_add_(1, aggregated_index[:, :, None].expand(-1, -1, hidden_size), pretrained_top_hiddens_without_cls_seq)
        embeds = aggregated_embeds / aggregated_count[:, :, None]
    elif alignment == 'ori':
        pretrained_top_hiddens_without_cls_seq = torch.gather(pretrained_top_hiddens, 1, remove_cls_seq_gather_index[:, :, None].expand(-1, -1, hidden_size))
        embeds = pretrained_top_hiddens_without_cls_seq
    else:
        embeds = pretrained_top_hiddens
    #print(embeds[-1])
    return embeds
