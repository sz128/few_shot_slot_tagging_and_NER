#!/usr/bin/env python3

'''
@Time   : 2020-04-20 21:39:42
@Author : su.zhu
@Desc   : 
'''

import json
import collections

from utils.basic_vocab_reader import *

# re-define read_input_vocab_from_data_file
def read_input_vocab_from_data_file_in_HIT_form(file_name, word_tokenizer=None, char_tokenizer=None, lowercase=False, mini_word_freq=1, with_tag=True, separator=':'):
    """
    json file
    """
    assert word_tokenizer is not None or char_tokenizer is not None
    print('Constructing input vocabulary from ', file_name, ' ...')
    all_tokens = {}
    with open(file_name, 'r') as f:
        data = json.load(f)
        for domain_name in data:
            for batch in data[domain_name]:
                #print(list(batch['support'])) # ['seq_ins', 'labels', 'seq_outs', 'word_piece_marks', 'tokenized_texts', 'word_piece_labels']
                #print(list(batch['batch'])) # ['seq_ins', 'labels', 'seq_outs', 'word_piece_marks', 'tokenized_texts', 'word_piece_labels']
                for data_type in ['support', 'batch']:
                    for words in batch[data_type]['seq_ins']:
                        for word in words:
                            if lowercase:
                                word = word.lower()

                            if word not in all_tokens:
                                all_tokens[word] = 1
                            else:
                                all_tokens[word] += 1

    if word_tokenizer is not None:
        sorted_all_tokens = sorted(all_tokens.items(), key=lambda x:x[1], reverse=True)
        selected_tokens = [x[0] for x in sorted_all_tokens if x[1] >= mini_word_freq]
        for token in selected_tokens:
            if token not in word_tokenizer.token_to_id:
                word_tokenizer.token_to_id[token] = word_tokenizer.vocab_size
                word_tokenizer.vocab_size += 1
        word_tokenizer.id_to_token = collections.OrderedDict([(ids, tok) for tok, ids in word_tokenizer.token_to_id.items()])

    if char_tokenizer is not None:
        for word in all_tokens:
            for char in word:
                if char not in char_tokenizer.token_to_id:
                    char_tokenizer.token_to_id[char] = char_tokenizer.vocab_size
                    char_tokenizer.vocab_size += 1
        char_tokenizer.id_to_token = collections.OrderedDict([(ids, tok) for tok, ids in char_tokenizer.token_to_id.items()])

class FewShotSlotVocab(Tokenizer):

    def __init__(self, vocab_data_storage, slot_embedding_type='with_BIO', no_special_labels=False, bos_eos=False, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>'):
        '''
        slot_embedding_type:
          1. with_BIO : all slot-tag embeddings are calculated from support set;
          2. with_BI : except that 'O' is ranomly initilized as an embedding vector with the same dimension;
          3. without_BIO : all slot embeddings are calculated from support set except for 'O', and 'B/I' are ranomly initilized as embedding vectors which will be added into slot embeddings to represent slot-tags.
        '''
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token)
        
        self.label_to_id = collections.OrderedDict()
        self.special_labels = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        if no_special_labels:
            self.vocab_size = 0
        else:
            self.label_to_id[self.pad_token] = 0
            self.label_to_id[self.unk_token] = 1
            if bos_eos:
                self.label_to_id[self.bos_token] = 2
                self.label_to_id[self.eos_token] = 3
                self.vocab_size = 4
            else:
                self.vocab_size = 2
        
        self.init_token_to_id = {}
        for tok in list(self.label_to_id) + ['B', 'I', 'O']:
            self.init_token_to_id[tok] = len(self.init_token_to_id)
        assert slot_embedding_type in {'with_BIO', 'with_BI', 'without_BIO'}
        self.slot_embedding_type = slot_embedding_type
        if self.slot_embedding_type == 'with_BIO':
            self.head_labels_token_ids = [self.init_token_to_id[tok] for tok in self.label_to_id]
            self.other_labels = ['O']
            ## NOTE: 'O' is put in the head
            self.label_to_id['O'] = self.vocab_size
            self.vocab_size += 1
        else:
            ## NOTE: 'O' is put in the head
            self.label_to_id['O'] = self.vocab_size
            self.vocab_size += 1
            self.head_labels_token_ids = [self.init_token_to_id[tok] for tok in self.label_to_id]
            self.other_labels = []
            #self.other_labels_seg_ids =

        assert type(vocab_data_storage) in {str, list, tuple, dict} # set is not deterministic
        if type(vocab_data_storage) is str:
            vocab = read_vocab_file(vocab_data_storage)
        else:
            vocab = vocab_data_storage
        for label in vocab:
            if label not in self.label_to_id:
                self.label_to_id[label] = self.vocab_size
                self.vocab_size += 1
                self.other_labels.append(label)
        
        self.head_labels_token_ids = torch.tensor(self.head_labels_token_ids, dtype=torch.long)
        if self.slot_embedding_type == 'without_BIO':
            self.get_bio_tensor_and_selected_slot_indexes()

        self.id_to_label = collections.OrderedDict([(ids, lab) for lab, ids in self.label_to_id.items()])

    def get_bio_tensor_and_selected_slot_indexes(self):
        self.other_labels_seg_ids = []
        self.other_labels_slot_names = []
        self.other_labels_selected_slot_ids = []
        for label in self.other_labels:
            bio, slot_name = label.split('-', 1)
            self.other_labels_seg_ids.append(self.init_token_to_id[bio])
            if len(self.other_labels_slot_names) == 0 or slot_name != self.other_labels_slot_names[-1]:
                self.other_labels_slot_names.append(slot_name)
            self.other_labels_selected_slot_ids.append(len(self.other_labels_slot_names) - 1)
        
        self.other_labels_seg_ids = torch.tensor(self.other_labels_seg_ids, dtype=torch.long)
        self.other_labels_selected_slot_ids = torch.tensor(self.other_labels_selected_slot_ids, dtype=torch.long)

    def get_elements_used_in_label_embeddings(self, device=None):
        head_labels_token_ids = self.head_labels_token_ids.to(device=device)
        if self.slot_embedding_type == 'without_BIO':
            other_labels_seg_ids = self.other_labels_seg_ids.to(device=device)
            other_labels_selected_slot_ids = self.other_labels_selected_slot_ids.to(device=device)
        else:
            other_labels_seg_ids = None
            other_labels_selected_slot_ids = None
        return head_labels_token_ids, other_labels_seg_ids, other_labels_selected_slot_ids

    def get_label_description_for_HIT_data(self, slot_to_desc):
        label_descriptions = []
        for label in self.other_labels:
            if label == 'O':
                words = ['ordinary', '-']
            else:
                bio, slot = label.split('-', 1)
                if bio == 'B':
                    bio = 'begin'
                elif bio == 'I':
                    bio = 'inner'
                elif bio == 'S':
                    bio = 'single'
                elif bio == 'M':
                    bio = 'middle'
                elif bio == 'E':
                    bio = 'end'
                else:
                    exit()
                words = [bio, '-'] + slot_to_desc[slot]
            label_descriptions.append(words)
        return label_descriptions

    def get_vocab_size(self,):
        return self.vocab_size

    def save_vocab(self, vocab_file):
        save_vocab_file(self.label_to_id, vocab_file)

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.label_to_id.get(token, self.label_to_id.get(self.unk_token))
    
    def _convert_id_to_token(self, index):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.id_to_label.get(index, self.unk_token)
